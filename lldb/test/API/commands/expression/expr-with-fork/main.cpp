#include <pthread.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

int fork_and_return(int value, bool use_vfork) {
  pid_t pid = use_vfork ? vfork() : fork();
  if (pid == -1)
    return -1;
  if (pid == 0) {
    // child
    _exit(value);
  }
  // parent
  int status;
  waitpid(pid, &status, 0);
  return WEXITSTATUS(status);
}

int fork_and_return_trap(int value) {
  pid_t pid = fork();
  if (pid == -1)
    return -1;
  if (pid == 0) {
    // child returning from the JITed function wrapper will hit a trap
    // instruction and terminate with SIGTRAP.
    return value;
  }
  // parent
  int status;
  waitpid(pid, &status, 0);
  if (WIFSIGNALED(status) && WTERMSIG(status) == SIGTRAP) {
    return 1; // Success: child terminated with SIGTRAP
  }
  return 0; // Failure
}

// Mutex-based synchronization for concurrent fork test.
// The main thread holds the mutex, starts a helper thread that waits
// on the mutex then forks, and the expression releases the mutex and
// waits to reacquire it - forcing the fork to happen on a different
// thread while the expression is running.
static pthread_mutex_t g_fork_mutex = PTHREAD_MUTEX_INITIALIZER;
static pid_t g_child_pid = -1;

static void *concurrent_fork_thread(void *arg) {
  // Wait until the expression releases the mutex.
  pthread_mutex_lock(&g_fork_mutex);
  g_child_pid = fork();
  if (g_child_pid == 0) {
    // child
    _exit(42);
  }
  // parent - release mutex so expression can reacquire.
  pthread_mutex_unlock(&g_fork_mutex);
  return nullptr;
}

// Called as an expression while another thread forks.
// The caller must hold g_fork_mutex before evaluating this expression.
int expr_with_concurrent_fork() {
  pthread_t t;
  pthread_create(&t, nullptr, concurrent_fork_thread, nullptr);

  // Release mutex - lets the helper thread proceed to fork.
  pthread_mutex_unlock(&g_fork_mutex);

  // Wait for the helper thread to finish (it forks and unlocks).
  pthread_join(t, nullptr);

  // Reacquire mutex to synchronize.
  pthread_mutex_lock(&g_fork_mutex);

  // Return the child PID so the test can verify fork happened.
  return (int)g_child_pid;
}

int main() {
  pthread_mutex_lock(&g_fork_mutex);
  int x = 42;
  return 0; // break here
}
