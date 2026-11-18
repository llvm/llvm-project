#include <stdlib.h>

struct Payload {
  long values[8];
  char name[16];
};

#define DISTANT_COUNT 512

int main(void) {
  struct Payload *heap = (struct Payload *)calloc(1, sizeof(struct Payload));
  for (int i = 0; i < 8; ++i)
    heap->values[i] = i * 100;

  long distant[DISTANT_COUNT];
  for (int i = 0; i < DISTANT_COUNT; ++i)
    distant[i] = i * 7 + 1;

  long total = 0;
  for (int i = 0; i < 4; ++i) {
    total += heap->values[i] + distant[i]; // break here
  }
  free(heap);
  return (int)(total & 0x7f);
}
