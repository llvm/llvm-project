#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <stdlib.h>
#else
#include <unistd.h>
#endif

int main(int argc, char *argv[]) {
  const char *foo = getenv("FOO");
  const char *nodebugheap = getenv("_NO_DEBUG_HEAP");
  int counter = 1;

  return 0; // breakpoint
}
