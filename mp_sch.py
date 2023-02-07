from multiprocessing import Process
#from multiprocessing import Manager
import time
import multiprocess_simulation_sch


if __name__ == '__main__':
    processes = []
    #results = Manager().dict()
    num_processes = 5
    iters = 100
    print('started')
    start = time.time()

    for n in range(num_processes):
        process = Process(target=multiprocess_simulation_sch.mp_simulate, args=(n,))

        processes.append(process)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    print('finished')
    print(time.time()-start)
    #print(len(results))
