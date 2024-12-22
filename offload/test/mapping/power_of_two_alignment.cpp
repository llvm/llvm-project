// RUN: %libomptarget-compilexx-run-and-check-generic

// Assuming the stack is allocated on the host starting at high addresses, the
// host memory layout for the following program looks like this:
//
//   low addr <----------------------------------------------------- high addr
//              |   16 bytes  | 16 bytes  |  16 bytes  | ? bytes  |
//              | collidePost |     s     | collidePre | stackPad |
//              |             | x | y | z |            |          |
//              `-------------'
//                    ^  `--------'
//                    |      ^
//                    |      |
//                    |      `-- too much padding (< 16 bytes) for s maps here
//                    |
//                    `------------------array extension error maps here
//
// libomptarget used to add too much padding to the device allocation of s and
// map it back to the host at the location indicated above when all of the
// following conditions were true:
// - Multiple members (s.y and s.z below) were mapped.  In this case, initial
//   padding might be needed to ensure later mapped members (s.z) are aligned
//   properly on the device.  (If the first member in the struct, s.x, were also
//   mapped, then the correct initial padding would always be zero.)
// - mod16 = &s % 16 was not a power of 2 (e.g., 0x7ffcce2b584e % 16 = 14).
//   libomptarget then incorrectly assumed mod16 was the existing host memory
//   alignment of s.  (The fix was to only look for alignments that are powers
//   of 2.)
// - &s.y % mod16 was > 1 (e.g., 0x7ffcce2b584f % 14 = 11).  libomptarget added
//   padding of that size for s, but at most 1 byte is ever actually needed.
//
// Below, we try many sizes of stackPad to try to produce those conditions.
//
// When collidePost was then mapped to the same host memory as the unnecessary
// padding for s, libomptarget reported an array extension error.  collidePost
// is never fully contained within that padding (which would avoid the extension
// error) because collidePost is 16 bytes while the padding is always less than
// 16 bytes due to the modulo operations.  (Later, libomptarget was changed not
// to consider padding to be mapped to the host, so it cannot be involved in
// array extension errors.)

#include <stdint.h>
#include <stdio.h>

template <typename StackPad>
void test() {
  StackPad stackPad;
  struct S { char x; char y[7]; char z[8]; };
  struct S collidePre, s, collidePost;
  uintptr_t mod16 = (uintptr_t)&s % 16;
  fprintf(stderr, "&s = %p\n", &s);
  fprintf(stderr, "&s %% 16 = %lu\n", mod16);
  if (mod16) {
    fprintf(stderr, "&s.y = %p\n", &s.y);
    fprintf(stderr, "&s.y %% %lu = %lu\n", mod16, (uintptr_t)&s.y % mod16);
  }
  fprintf(stderr, "&collidePre = %p\n", &collidePre);
  fprintf(stderr, "&collidePost = %p\n", &collidePost);
  #pragma omp target data map(to:s.y, s.z)
  #pragma omp target data map(to:collidePre, collidePost)
  ;
}

#define TEST(StackPad)                                                         \
  fprintf(stderr, "-------------------------------------\n");                  \
  fprintf(stderr, "StackPad=%s\n", #StackPad);                                 \
  test<StackPad>()

int main() {
  TEST(char[1]);
  TEST(char[2]);
  TEST(char[3]);
  TEST(char[4]);
  TEST(char[5]);
  TEST(char[6]);
  TEST(char[7]);
  TEST(char[8]);
  TEST(char[9]);
  TEST(char[10]);
  TEST(char[11]);
  TEST(char[12]);
  TEST(char[13]);
  TEST(char[14]);
  TEST(char[15]);
  TEST(char[16]);
  // CHECK: pass
  printf("pass\n");
  return 0;
}
