#include <stdio.h>
#include <thread>
#include <mutex>
#include <condition_variable>

std::mutex contended_mutex;

std::mutex control_mutex;
std::condition_variable  control_condition;

std::mutex thread_started_mutex;
std::condition_variable  thread_started_condition;

// This function runs in a thread.  The locking dance is to make sure that 
// by the time the main thread reaches the pthread_join below, this thread
// has for sure acquired the contended_mutex.  So then the call_me_to_get_lock
// function will block trying to get the mutex, and only succeed once it
// signals this thread, then lets it run to wake up from the cond_wait and
// release the mutex.

void
lock_acquirer_1 (void)
{
  std::unique_lock<std::mutex> contended_lock(contended_mutex);
  
  // Grab this mutex, that will ensure that the main thread
  // is in its cond_wait for it (since that's when it drops the mutex.

  thread_started_mutex.lock();
  thread_started_mutex.unlock();

  // Now signal the main thread that it can continue, we have the contended lock
  // so the call to call_me_to_get_lock won't make any progress till  this
  // thread gets a chance to run.

  std::unique_lock<std::mutex> control_lock(control_mutex);

  thread_started_condition.notify_all();

  control_condition.wait(control_lock);

}

int
call_me_to_get_lock (int ret_val)
{
  control_condition.notify_all();
  contended_mutex.lock();
  return ret_val;
}

int
get_int() {
  return 567;
}

int main ()
{
  std::unique_lock<std::mutex> thread_started_lock(thread_started_mutex);

  std::thread thread_1(lock_acquirer_1);

  thread_started_condition.wait(thread_started_lock);

  control_mutex.lock();
  control_mutex.unlock();

  // Break here.  At this point the other thread will have the contended_mutex,
  // and be sitting in its cond_wait for the control condition.  So there is
  // no way that our by-hand calling of call_me_to_get_lock will proceed
  // without running the first thread at least somewhat.

  int result = call_me_to_get_lock(get_int());
  thread_1.join();

  return 0;

}
