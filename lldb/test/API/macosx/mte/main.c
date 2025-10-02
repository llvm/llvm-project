#include <malloc/malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Produce some names on the trace
const size_t tag_granule = 16;
static uint8_t *my_malloc(void) { return malloc(2 * tag_granule); }
static uint8_t *allocate(void) { return my_malloc(); }

static void my_free(void *ptr) { free(ptr); }
static void deallocate(void *ptr) { my_free(ptr); }

static void touch_memory(uint8_t *ptr) { ptr[7] = 1; } // invalid access
static void modify(uint8_t *ptr) { touch_memory(ptr); }

int main() {
  uint8_t *ptr = allocate();

  strncpy((char *)ptr, "Hello", 16);
  strncpy((char *)ptr + 16, "World", 16);

  deallocate(ptr); // before free

  modify(ptr); // use-after-free

  return 0;
}
