// Show that the ALWAYS map-type modifier on the outer map clause should be
// propagated to the entries pushed by a user-defined mapper, using no target
// construct at all: the device data is inspected with omp_get_mapped_ptr() and
// omp_target_memcpy(), so what is checked is purely the data-motion done by
// `target enter data`.
//
// The mapper transfers s.y. We pre-map s.y so that it already has a device copy
// with a nonzero reference count, and set that copy to a known value. Without
// ALWAYS, the `to` of the second `enter data` is suppressed (a present,
// ref-counted entry is not copied), so the host's 111 does not reach the
// device. ALWAYS must force the copy -- but only if ALWAYS is propagated from
// the outer clause to the mapper's s.y entry.
//
// FIXME: ALWAYS is not propagated to the mapper's entries yet, so the transfer
// is currently suppressed and the device copy still reads as 0 (its pre-set
// value). Once ALWAYS is propagated:
//   EXPECTED: device s.y = 111

// RUN: %libomptarget-compile-run-and-check-generic

#include <omp.h>
#include <stdio.h>

typedef struct {
  int x;
  int y;
  int z;
} S;

#pragma omp declare mapper(default : S s) map(tofrom : s.y)

S s;

int main() {
  int dev = omp_get_default_device();
  int host = omp_get_initial_device();

  s.y = 0;

#pragma omp target enter data map(alloc : s.y)

  // The device copy of s.y is uninitialized after `alloc`; set it to 0 so that
  // a missing transfer below is distinguishable from garbage.
  int *dev_y = (int *)omp_get_mapped_ptr(&s.y, dev);
  omp_target_memcpy(dev_y, &s.y, sizeof(int), 0, 0, dev, host);

  s.y = 111;

#pragma omp target enter data map(always, to : s)

  // ALWAYS should force s.y to the device even though it is already mapped
  // (ref count > 0), but the modifier is not propagated yet, so the transfer
  // does not happen.
  int dev_y_val = -1;
  omp_target_memcpy(&dev_y_val, dev_y, sizeof(int), 0, 0, host, dev);

  printf("device s.y = %d\n", dev_y_val); // CHECK: device s.y = 0

#pragma omp target exit data map(delete : s.y)
}
