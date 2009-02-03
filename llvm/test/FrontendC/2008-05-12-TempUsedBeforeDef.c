// RUN: %llvmgcc -w -S -o /dev/null %s
// PR2264.
unsigned foo = 8L;
unsigned bar = 0L;
volatile unsigned char baz = 6L;
int test() {
  char qux = 1L;
  for (; baz >= -29; baz--)
    bork(bar && foo, qux);
}
