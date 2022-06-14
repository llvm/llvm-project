#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <future>
#include <inttypes.h>
#include <memory>
#include <mutex>
#if !defined(_WIN32)
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#endif
#include "thread.h"
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <thread>
#include <time.h>
#include <vector>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

static const char *const PRINT_PID_COMMAND = "print-pid";

static bool g_print_thread_ids = false;
static std::mutex g_print_mutex;
static bool g_threads_do_segfault = false;

static std::mutex g_jump_buffer_mutex;
static jmp_buf g_jump_buffer;
static bool g_is_segfaulting = false;

static char g_message[256];

static volatile char g_c1 = '0';
static volatile char g_c2 = '1';

static void print_pid() {
#if defined(_WIN32)
  fprintf(stderr, "PID: %d\n", ::GetCurrentProcessId());
#else
  fprintf(stderr, "PID: %d\n", getpid());
#endif
}

static void signal_handler(int signo) {
#if defined(_WIN32)
  // No signal support on Windows.
#else
  const char *signal_name = nullptr;
  switch (signo) {
  case SIGUSR1:
    signal_name = "SIGUSR1";
    break;
  case SIGSEGV:
    signal_name = "SIGSEGV";
    break;
  default:
    signal_name = nullptr;
  }

  // Print notice that we received the signal on a given thread.
  char buf[100];
  if (signal_name)
    snprintf(buf, sizeof(buf), "received %s on thread id: %" PRIx64 "\n", signal_name, get_thread_id());
  else
    snprintf(buf, sizeof(buf), "received signo %d (%s) on thread id: %" PRIx64 "\n", signo, strsignal(signo), get_thread_id());
  write(STDOUT_FILENO, buf, strlen(buf));

  // Reset the signal handler if we're one of the expected signal handlers.
  switch (signo) {
  case SIGSEGV:
    if (g_is_segfaulting) {
      // Fix up the pointer we're writing to.  This needs to happen if nothing
      // intercepts the SIGSEGV (i.e. if somebody runs this from the command
      // line).
      longjmp(g_jump_buffer, 1);
    }
    break;
  case SIGUSR1:
    if (g_is_segfaulting) {
      // Fix up the pointer we're writing to.  This is used to test gdb remote
      // signal delivery. A SIGSEGV will be raised when the thread is created,
      // switched out for a SIGUSR1, and then this code still needs to fix the
      // seg fault. (i.e. if somebody runs this from the command line).
      longjmp(g_jump_buffer, 1);
    }
    break;
  }

  // Reset the signal handler.
  sig_t sig_result = signal(signo, signal_handler);
  if (sig_result == SIG_ERR) {
    fprintf(stderr, "failed to set signal handler: errno=%d\n", errno);
    exit(1);
  }
#endif
}

static void swap_chars() {
#if defined(__x86_64__) || defined(__i386__)
  asm volatile("movb %1, (%2)\n\t"
               "movb %0, (%3)\n\t"
               "movb %0, (%2)\n\t"
               "movb %1, (%3)\n\t"
               :
               : "i"('0'), "i"('1'), "r"(&g_c1), "r"(&g_c2)
               : "memory");
#elif defined(__aarch64__)
  asm volatile("strb %w1, [%2]\n\t"
               "strb %w0, [%3]\n\t"
               "strb %w0, [%2]\n\t"
               "strb %w1, [%3]\n\t"
               :
               : "r"('0'), "r"('1'), "r"(&g_c1), "r"(&g_c2)
               : "memory");
#elif defined(__arm__)
  asm volatile("strb %1, [%2]\n\t"
               "strb %0, [%3]\n\t"
               "strb %0, [%2]\n\t"
               "strb %1, [%3]\n\t"
               :
               : "r"('0'), "r"('1'), "r"(&g_c1), "r"(&g_c2)
               : "memory");
#else
#warning This may generate unpredictible assembly and cause the single-stepping test to fail.
#warning Please add appropriate assembly for your target.
  g_c1 = '1';
  g_c2 = '0';

  g_c1 = '0';
  g_c2 = '1';
#endif
}

static void trap() {
#if defined(__x86_64__) || defined(__i386__)
  asm volatile("int3");
#elif defined(__aarch64__)
  asm volatile("brk #0xf000");
#elif defined(__arm__)
  asm volatile("udf #254");
#elif defined(__powerpc__)
  asm volatile("trap");
#elif __has_builtin(__builtin_debugtrap())
  __builtin_debugtrap();
#else
#warning Don't know how to generate a trap. Some tests may fail.
#endif
}

static void hello() {
  std::lock_guard<std::mutex> lock(g_print_mutex);
  printf("hello, world\n");
}

static void *thread_func(std::promise<void> ready) {
  ready.set_value();
  static std::atomic<int> s_thread_index(1);
  const int this_thread_index = s_thread_index++;
  if (g_print_thread_ids) {
    std::lock_guard<std::mutex> lock(g_print_mutex);
    printf("thread %d id: %" PRIx64 "\n", this_thread_index, get_thread_id());
  }

  if (g_threads_do_segfault) {
    // Sleep for a number of seconds based on the thread index.
    // TODO add ability to send commands to test exe so we can
    // handle timing more precisely.  This is clunky.  All we're
    // trying to do is add predictability as to the timing of
    // signal generation by created threads.
    int sleep_seconds = 2 * (this_thread_index - 1);
    std::this_thread::sleep_for(std::chrono::seconds(sleep_seconds));

    // Test creating a SEGV.
    {
      std::lock_guard<std::mutex> lock(g_jump_buffer_mutex);
      g_is_segfaulting = true;
      int *bad_p = nullptr;
      if (setjmp(g_jump_buffer) == 0) {
        // Force a seg fault signal on this thread.
        *bad_p = 0;
      } else {
        // Tell the system we're no longer seg faulting.
        // Used by the SIGUSR1 signal handler that we inject
        // in place of the SIGSEGV so it only tries to
        // recover from the SIGSEGV if this seg fault code
        // was in play.
        g_is_segfaulting = false;
      }
    }

    {
      std::lock_guard<std::mutex> lock(g_print_mutex);
      printf("thread %" PRIx64 ": past SIGSEGV\n", get_thread_id());
    }
  }

  int sleep_seconds_remaining = 60;
  std::this_thread::sleep_for(std::chrono::seconds(sleep_seconds_remaining));

  return nullptr;
}

static bool consume_front(std::string &str, const std::string &front) {
  if (str.find(front) != 0)
    return false;

  str = str.substr(front.size());
  return true;
}

int main(int argc, char **argv) {
  lldb_enable_attach();

  std::vector<std::thread> threads;
  std::unique_ptr<uint8_t[]> heap_array_up;
  int return_value = 0;

#if !defined(_WIN32)
  // Set the signal handler.
  sig_t sig_result = signal(SIGALRM, signal_handler);
  if (sig_result == SIG_ERR) {
    fprintf(stderr, "failed to set SIGALRM signal handler: errno=%d\n", errno);
    exit(1);
  }

  sig_result = signal(SIGUSR1, signal_handler);
  if (sig_result == SIG_ERR) {
    fprintf(stderr, "failed to set SIGUSR1 handler: errno=%d\n", errno);
    exit(1);
  }

  sig_result = signal(SIGSEGV, signal_handler);
  if (sig_result == SIG_ERR) {
    fprintf(stderr, "failed to set SIGSEGV handler: errno=%d\n", errno);
    exit(1);
  }

  sig_result = signal(SIGCHLD, SIG_IGN);
  if (sig_result == SIG_ERR) {
    fprintf(stderr, "failed to set SIGCHLD handler: errno=%d\n", errno);
    exit(1);
  }
#endif

  // Process command line args.
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (consume_front(arg, "stderr:")) {
      // Treat remainder as text to go to stderr.
      fprintf(stderr, "%s\n", arg.c_str());
    } else if (consume_front(arg, "retval:")) {
      // Treat as the return value for the program.
      return_value = std::atoi(arg.c_str());
    } else if (consume_front(arg, "sleep:")) {
      // Treat as the amount of time to have this process sleep (in seconds).
      int sleep_seconds_remaining = std::atoi(arg.c_str());

      // Loop around, sleeping until all sleep time is used up.  Note that
      // signals will cause sleep to end early with the number of seconds
      // remaining.
      std::this_thread::sleep_for(
          std::chrono::seconds(sleep_seconds_remaining));

    } else if (consume_front(arg, "set-message:")) {
      // Copy the contents after "set-message:" to the g_message buffer.
      // Used for reading inferior memory and verifying contents match
      // expectations.
      strncpy(g_message, arg.c_str(), sizeof(g_message));

      // Ensure we're null terminated.
      g_message[sizeof(g_message) - 1] = '\0';

    } else if (consume_front(arg, "print-message:")) {
      std::lock_guard<std::mutex> lock(g_print_mutex);
      printf("message: %s\n", g_message);
    } else if (consume_front(arg, "get-data-address-hex:")) {
      volatile void *data_p = nullptr;

      if (arg == "g_message")
        data_p = &g_message[0];
      else if (arg == "g_c1")
        data_p = &g_c1;
      else if (arg == "g_c2")
        data_p = &g_c2;

      std::lock_guard<std::mutex> lock(g_print_mutex);
      printf("data address: %p\n", data_p);
    } else if (consume_front(arg, "get-heap-address-hex:")) {
      // Create a byte array if not already present.
      if (!heap_array_up)
        heap_array_up.reset(new uint8_t[32]);

      std::lock_guard<std::mutex> lock(g_print_mutex);
      printf("heap address: %p\n", heap_array_up.get());

    } else if (consume_front(arg, "get-stack-address-hex:")) {
      std::lock_guard<std::mutex> lock(g_print_mutex);
      printf("stack address: %p\n", &return_value);
    } else if (consume_front(arg, "get-code-address-hex:")) {
      void (*func_p)() = nullptr;

      if (arg == "hello")
        func_p = hello;
      else if (arg == "swap_chars")
        func_p = swap_chars;

      std::lock_guard<std::mutex> lock(g_print_mutex);
      printf("code address: %p\n", func_p);
    } else if (consume_front(arg, "call-function:")) {
      void (*func_p)() = nullptr;

      if (arg == "hello")
        func_p = hello;
      else if (arg == "swap_chars")
        func_p = swap_chars;
      func_p();
#if !defined(_WIN32) && !defined(TARGET_OS_WATCH) && !defined(TARGET_OS_TV)
    } else if (arg == "fork") {
      assert (fork() != -1);
    } else if (arg == "vfork") {
      if (vfork() == 0)
        _exit(0);
#endif
    } else if (consume_front(arg, "thread:new")) {
      std::promise<void> promise;
      std::future<void> ready = promise.get_future();
      threads.push_back(std::thread(thread_func, std::move(promise)));
      ready.wait();
    } else if (consume_front(arg, "thread:print-ids")) {
      // Turn on thread id announcing.
      g_print_thread_ids = true;

      // And announce us.
      {
        std::lock_guard<std::mutex> lock(g_print_mutex);
        printf("thread 0 id: %" PRIx64 "\n", get_thread_id());
      }
    } else if (consume_front(arg, "thread:segfault")) {
      g_threads_do_segfault = true;
    } else if (consume_front(arg, "print-pid")) {
      print_pid();
    } else if (consume_front(arg, "print-env:")) {
      // Print the value of specified envvar to stdout.
      const char *value = getenv(arg.c_str());
      printf("%s\n", value ? value : "__unset__");
    } else if (consume_front(arg, "trap")) {
      trap();
    } else {
      // Treat the argument as text for stdout.
      printf("%s\n", argv[i]);
    }
  }

  // If we launched any threads, join them
  for (std::vector<std::thread>::iterator it = threads.begin();
       it != threads.end(); ++it)
    it->join();

  return return_value;
}
