#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

extern uint8_t __start_target_section[];
extern uint8_t __stop_target_section[];

__attribute__((used, section("target_section"))) int target_function(void) {
  return 42;
}

typedef int (*target_function_t)(void);

int main(void) {
  size_t target_function_size = __stop_target_section - __start_target_section;
  size_t page_size = sysconf(_SC_PAGESIZE);
  size_t page_aligned_size =
      (target_function_size + page_size - 1) & ~(page_size - 1);

  void *executable_memory =
      mmap(NULL, page_aligned_size, PROT_READ | PROT_WRITE | PROT_EXEC,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (executable_memory == MAP_FAILED) {
    perror("mmap");
    return 1;
  }

  memcpy(executable_memory, __start_target_section, target_function_size);

  target_function_t func = (target_function_t)executable_memory;
  int result = func(); // Break here
  printf("Result from target function: %d\n", result);

  return 0;
}
