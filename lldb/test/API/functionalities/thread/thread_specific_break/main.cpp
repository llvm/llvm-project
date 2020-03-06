#include <chrono>
#include <thread>

void
thread_function ()
{
    // Set thread-specific breakpoint here.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    // On Windows, a sleep_for of less than about 16 ms effectively calls
    // Sleep(0).  The MS standard thread implementation uses a system thread
    // pool, which can deadlock on a Sleep(0), hanging not only the secondary
    // thread but the entire test.  I increased the delay to 20 ms to ensure
    // Sleep is called with a delay greater than 0.  The deadlock potential
    // is described here:
    // https://docs.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-sleep#remarks
}

int 
main ()
{
    // Set main breakpoint here.

    #ifdef __APPLE__
    pthread_setname_np("main-thread");
    #endif

    std::thread t(thread_function);
    t.join();

    thread_function();
    return 0;
}
