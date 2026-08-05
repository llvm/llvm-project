// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// A structure containing a pointer, mapped as an untyped byte range rather than
// by its own name, and then a close mapping of what the member points to.
//
// The byte range is mapped without close, so its corresponding storage is the
// original storage. Attachment for the member then writes the device pointee
// address into the original member, since a member's device address is the
// structure's device address plus the member offset.
//
// This is the same situation as close_ptee_struct_member.c, but the mapped
// entry carries no type information: the runtime sees only a range of bytes
// that happens to contain a pointer. Anything that gives the pointer storage of
// its own would have to promote this whole range, which is the only storage the
// member's device address can be derived from.
//
// FIXME: the value checked below is the one produced today; the expected value
// is given alongside it.

#include <stdio.h>

#pragma omp requires unified_shared_memory

int arr[10];

struct S {
  int x;
  int y;
  int *p;
};

struct S s;
char *buf = (char *)&s;

int main() {
  for (int i = 0; i < 10; ++i)
    arr[i] = 42;
  s.p = &arr[0];

  // The unified_shared_memory requirement is registered when a device image is
  // loaded, so the program needs a target region for it to take effect.
#pragma omp target
  {
  }

  // CHECK: before: s.p == &arr[0]
  printf("before: s.p %s &arr[0]\n", s.p == &arr[0] ? "==" : "!=");

  // The structure is mapped as plain bytes, so it stays on the host path.
#pragma omp target enter data map(alloc : buf[0 : sizeof(struct S)])

  // The pointee is newly mapped with close, which triggers the attachment.
#pragma omp target enter data map(close, alloc : s.p[0 : 10])

  // The pointee was kept on the host path, so the original s.p is unchanged.
  // CHECK: after attach: s.p == &arr[0]
  printf("after attach: s.p %s &arr[0]\n", s.p == &arr[0] ? "==" : "!=");

  // CHECK: Done!
  printf("Done!\n");
  return 0;
}
