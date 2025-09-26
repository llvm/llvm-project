#include <malloc/malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Produce some names on the trace
const size_t tag_granule = 16;
uint8_t *my_malloc(void) { return malloc(2 * tag_granule); }
uint8_t *allocate(void) { return my_malloc(); }

void my_free(void *ptr) { free(ptr); }
void deallocate(void *ptr) { my_free(ptr); }

void touch_memory(uint8_t *ptr) { ptr[7] = 1; } // invalid access
void modify(uint8_t *ptr) { touch_memory(ptr); }

int main() {
  uint8_t *ptr = allocate();
  printf("ptr: %p\n", ptr);

  strcpy((char *)ptr, "Hello");
  strcpy((char *)ptr + 16, "World");

  deallocate(ptr); // before free

  modify(ptr); // use-after-free

  return 0;
}
