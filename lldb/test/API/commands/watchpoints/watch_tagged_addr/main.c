#include <stdint.h>

uint32_t global_var = 0; // Watchpoint variable declaration.

int main(int argc, char **argv) {
  int dummy = 0;
  // Move address of global variable into tagged_ptr after tagging
  // Simple tagging scheme where 62nd bit of tagged address is set
  uint32_t *tagged_ptr = (uint32_t *)((uint64_t)&global_var | (1ULL << 62));

  // pacdza computes and inserts a pointer authentication code for address
  // stored in tagged_ptr using PAC key A.
  __asm__ __volatile__("pacdza %0" : "=r"(tagged_ptr) : "r"(tagged_ptr));

  ++dummy; // Set break point at this line.

  // Increment global_var
  ++global_var;

  ++dummy;

  // autdza authenticates tagged_ptr using PAC key A.
  __asm__ __volatile__("autdza %0" : "=r"(tagged_ptr) : "r"(tagged_ptr));

  // Increment global_var using tagged_ptr
  ++*tagged_ptr;

  return 0;
}
