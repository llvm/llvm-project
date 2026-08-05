// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// A close mapping of a pointee whose pointer is a structure member, where the
// structure itself stays on the unified-shared-memory host path.
//
// Pointer attachment writes the device pointee address into the corresponding
// pointer, which for a member is an interior address of the structure's own
// storage: the device address of s.p is defined as the device address of s plus
// the member offset. When s shares storage with the original, that write lands
// in the original s.p, and device code that reaches the member through the
// structure base -- as bar() does below -- observes it.
//
// The close pointee therefore gets a device buffer that the kernel writes,
// while nothing copies it back to the original storage, so the write is lost.
//
// FIXME: the values checked below are the ones produced today; the expected
// value is given alongside each. Note that the pointer cannot be given storage
// of its own to attach into, precisely because a member's device address is
// derived from the structure's: see the comment above.

#include <stdio.h>

#pragma omp requires unified_shared_memory

int arr[10] = {0};

struct S {
  int a;
  int *p;
  int b;
};

struct S s = {1, &arr[0], 2};

int *p_device;

#pragma omp begin declare target
// Reaches the member through the structure base, so it must see the attached
// value at the device address of s plus the offset of p.
void bar(struct S *ps) { ps->p[0] = 77; }
#pragma omp end declare target

int main() {
  // CHECK: before: s.p == &arr[0]
  printf("before: s.p %s &arr[0]\n", s.p == &arr[0] ? "==" : "!=");

#pragma omp target data map(tofrom : s)
  {
#pragma omp target enter data map(close, to : s.p[0 : 10])

#pragma omp target map(present, alloc : s) map(from : p_device)
    {
      p_device = s.p;
      bar(&s);
    }
  }

  // The pointee was kept on the host path, since attaching the member pointer
  // would otherwise have written a device address into the original s.p.
  // CHECK: in tgt: s.p == &arr[0]
  printf("in tgt: s.p %s &arr[0]\n", p_device == &arr[0] ? "==" : "!=");

  // The original member pointer is intact afterwards.
  // CHECK: after: s.p == &arr[0]
  printf("after: s.p %s &arr[0]\n", s.p == &arr[0] ? "==" : "!=");

  // bar() wrote through the member pointer, which designates arr itself.
  // CHECK: arr[0] = 77
  printf("arr[0] = %d\n", arr[0]);

  // CHECK: Done!
  printf("Done!\n");
  return 0;
}
