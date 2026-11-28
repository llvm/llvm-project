#include <atomic>
#include <cstddef>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

// Carve every worker thread's stack out of one mapping so that the stack
// pointers of all the workers fall into the same memory region.
constexpr int num_threads = 3;
constexpr size_t stack_size = 1024 * 1024;

std::atomic<int> started{0};
std::atomic<bool> done{false};

void *thread_func(void *) {
  started++;
  while (!done)
    usleep(1000);
  return nullptr;
}

int main() {
  void *stacks = mmap(nullptr, num_threads * stack_size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (stacks == MAP_FAILED)
    return 1;

  pthread_t threads[num_threads];
  for (int i = 0; i < num_threads; ++i) {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstack(&attr, static_cast<char *>(stacks) + i * stack_size,
                          stack_size);
    if (pthread_create(&threads[i], &attr, thread_func, nullptr) != 0)
      return 1;
    pthread_attr_destroy(&attr);
  }

  while (started != num_threads)
    usleep(1000);

  done = true; // Set a breakpoint here

  for (int i = 0; i < num_threads; ++i)
    pthread_join(threads[i], nullptr);
  munmap(stacks, num_threads * stack_size);
  return 0;
}
