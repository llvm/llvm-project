// clang-format off
// RUN: %libomptarget-compilexx-generic && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 | %fcheck-generic
// clang-format on

// REQUIRES: libomptarget-debug

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// XFAIL: intelgpu

#include <stdio.h>
#include <stdlib.h>

struct Descriptor {
  int *datum;
  long int x;
  int *more_datum;
  int xi;
  int val_datum, val_more_datum;
  long int arr[1][30];
  int val_arr;
};

int main() {
  Descriptor dat = Descriptor();
  dat.datum = (int *)malloc(sizeof(int) * 10);
  dat.more_datum = (int *)malloc(sizeof(int) * 20);
  dat.xi = 3;
  dat.arr[0][0] = 1;

  dat.datum[7] = 7;
  dat.more_datum[17] = 17;
  dat.datum[dat.arr[0][0]] = 0;

  /// The struct is mapped with type 0x0 when the pointer fields are mapped.
  /// The struct is also map explicitly by the user. The second mapping by
  /// the user must not overwrite the mapping set up for the pointer fields
  /// when mapping the struct happens after the mapping of the pointers.

  // clang-format off
  // CHECK: omptarget --> Entry  0: Base=0x{{0*}}[[DAT_HST_PTR_BASE:.*]], Begin=0x{{0*}}[[DAT_HST_PTR_BASE]], Size=288, Type=0x{{0*}}1, Name=unknown
  // CHECK: omptarget --> Entry  1: Base=0x{{0*}}[[DATUM_HST_PTEE_BASE:.*]], Begin=0x{{0*}}[[DATUM_HST_PTEE_BASE]], Size=40, Type=0x{{0*}}1, Name=unknown
  // CHECK: omptarget --> Entry  2: Base=0x{{0*}}[[DAT_HST_PTR_BASE]], Begin=0x{{0*}}[[DATUM_HST_PTEE_BASE]], Size=8, Type=0x{{0*}}4000, Name=unknown
  // CHECK: omptarget --> Entry  3: Base=0x{{0*}}[[MORE_DATUM_HST_PTEE_BASE:.*]], Begin=0x{{0*}}[[MORE_DATUM_HST_PTEE_BASE]], Size=80, Type=0x{{0*}}1, Name=unknown
  // CHECK: omptarget --> Entry  4: Base=0x{{0*}}[[MORE_DATUM_HST_PTR_BASE:.*]], Begin=0x{{0*}}[[MORE_DATUM_HST_PTEE_BASE]], Size=8, Type=0x{{0*}}4000, Name=unknown
  // clang-format on

  /// The struct will be mapped in the same order as the above entries.

  /// First argument is the struct itself and it will be mapped once.

  // clang-format off
  // CHECK: omptarget --> Looking up mapping(HstPtrBegin=0x{{0*}}[[DAT_HST_PTR_BASE]], Size=288)...
  // CHECK: PluginInterface --> MemoryManagerTy::allocate: size 288 with host pointer 0x{{0*}}[[DAT_HST_PTR_BASE]].
  // CHECK: omptarget --> Creating new map entry with HstPtrBase=0x{{0*}}[[DAT_HST_PTR_BASE]], HstPtrBegin=0x{{0*}}[[DAT_HST_PTR_BASE]], TgtAllocBegin=0x{{0*}}[[DAT_DEVICE_PTR_BASE:.*]], TgtPtrBegin=0x{{0*}}[[DAT_DEVICE_PTR_BASE]], Size=288, DynRefCount=1, HoldRefCount=0, Name=unknown
  // CHECK: omptarget --> Moving 288 bytes (hst:0x{{0*}}[[DAT_HST_PTR_BASE]]) -> (tgt:0x{{0*}}[[DAT_DEVICE_PTR_BASE]])
  // clang-format on

  /// Second argument is dat.datum[ : 10]:
  // clang-format off
  // CHECK: omptarget --> Looking up mapping(HstPtrBegin=0x{{0*}}[[DATUM_HST_PTEE_BASE]], Size=40)...
  // CHECK: PluginInterface --> MemoryManagerTy::allocate: size 40 with host pointer 0x{{0*}}[[DATUM_HST_PTEE_BASE]].
  // CHECK: omptarget --> Creating new map entry with HstPtrBase=0x{{0*}}[[DATUM_HST_PTEE_BASE]], HstPtrBegin=0x{{0*}}[[DATUM_HST_PTEE_BASE]], TgtAllocBegin=0x[[DATUM_DEVICE_PTR_BASE:.*]], TgtPtrBegin=0x{{0*}}[[DATUM_DEVICE_PTR_BASE]], Size=40, DynRefCount=1, HoldRefCount=0, Name=unknown
  // CHECK: omptarget --> Moving 40 bytes (hst:0x{{0*}}[[DATUM_HST_PTEE_BASE]]) -> (tgt:0x{{0*}}[[DATUM_DEVICE_PTR_BASE]])
  // clang-format on

  /// Third argument conditionally attaches data.datum -> dat.datum[:]
  // CHECK: omptarget --> Deferring ATTACH map-type processing for argument 2

  /// Fourth argument is dat.more_datum[ : 10]:
  // clang-format off
  // CHECK: omptarget --> Looking up mapping(HstPtrBegin=0x{{0*}}[[MORE_DATUM_HST_PTEE_BASE]], Size=80)...
  // CHECK: omptarget --> Creating new map entry with HstPtrBase=0x{{0*}}[[MORE_DATUM_HST_PTEE_BASE]], HstPtrBegin=0x{{0*}}[[MORE_DATUM_HST_PTEE_BASE]], TgtAllocBegin=0x{{0*}}[[MORE_DATUM_DEVICE_PTR_BASE:.*]], TgtPtrBegin=0x{{0*}}[[MORE_DATUM_DEVICE_PTR_BASE]], Size=80, DynRefCount=1, HoldRefCount=0, Name=unknown
  // CHECK: omptarget --> Moving 80 bytes (hst:0x{{0*}}[[MORE_DATUM_HST_PTEE_BASE]]) -> (tgt:0x{{0*}}[[MORE_DATUM_DEVICE_PTR_BASE]])
  // clang-format on

  /// Fifth argument conditionally attaches data.more_datum -> dat.more_datum[:]
  // clang-format off
  // CHECK: omptarget --> Deferring ATTACH map-type processing for argument 4
  // clang-format on

  /// Attach entries are handled at the end
  // clang-format off
  // CHECK: omptarget --> Processing 2 deferred ATTACH map entries
  // CHECK: omptarget --> Processing ATTACH entry 0: HstPtr=0x{{0*}}[[DAT_HST_PTR_BASE]], HstPteeBegin=0x{{0*}}[[DATUM_HST_PTEE_BASE]], PtrSize=8, MapType=0x{{0*}}4000
  // CHECK: omptarget --> Attach pointee 0x{{0*}}[[DATUM_HST_PTEE_BASE]] was newly allocated: yes
  // CHECK: omptarget --> Update pointer (0x{{.*}}) -> [0x{{.*}}]

  // CHECK: omptarget --> Processing ATTACH entry 1: HstPtr=0x{{0*}}[[MORE_DATUM_HST_PTR_BASE]], HstPteeBegin=0x{{0*}}[[MORE_DATUM_HST_PTEE_BASE]], PtrSize=8, MapType=0x{{0*}}4000
  // CHECK: omptarget --> Attach pointee 0x{{0*}}[[MORE_DATUM_HST_PTEE_BASE]] was newly allocated: yes
  // CHECK: omptarget --> Update pointer (0x{{.*}}) -> [0x{{.*}}]
  // clang-format on

#pragma omp target enter data map(to : dat.datum[ : 10])                       \
    map(to : dat.more_datum[ : 20]) map(to : dat)

  /// Checks induced by having a target region:
  // clang-format off
  // CHECK: omptarget --> Entry  0: Base=0x{{0*}}[[DAT_HST_PTR_BASE]], Begin=0x{{0*}}[[DAT_HST_PTR_BASE]], Size=288, Type=0x{{0*}}223, Name=unknown
  // CHECK: omptarget --> Mapping exists (implicit) with HstPtrBegin=0x{{0*}}[[DAT_HST_PTR_BASE]], TgtPtrBegin=0x{{0*}}[[DAT_DEVICE_PTR_BASE]], Size=288, DynRefCount=2 (incremented), HoldRefCount=0, Name=unknown
  // CHECK: omptarget --> Obtained target argument 0x{{0*}}[[DAT_DEVICE_PTR_BASE]] from host pointer 0x{{0*}}[[DAT_HST_PTR_BASE]]
  // clang-format on

#pragma omp target
  {
    dat.xi = 4;
    dat.datum[7]++;
    dat.more_datum[17]++;
    dat.val_datum = dat.datum[7];
    dat.val_more_datum = dat.more_datum[17];
    dat.datum[dat.arr[0][0]] = dat.xi;
    dat.val_arr = dat.datum[dat.arr[0][0]];
  }

  /// Post-target region checks:
  // clang-format off
  // CHECK: omptarget --> Mapping exists with HstPtrBegin=0x{{0*}}[[DAT_HST_PTR_BASE]], TgtPtrBegin=0x{{0*}}[[DAT_DEVICE_PTR_BASE]], Size=288, DynRefCount=1 (decremented), HoldRefCount=0
  // clang-format on

#pragma omp target exit data map(from : dat)

  /// Target data end checks:
  // clang-format off
  // CHECK: omptarget --> Mapping exists with HstPtrBegin=0x{{0*}}[[DAT_HST_PTR_BASE]], TgtPtrBegin=0x{{0*}}[[DAT_DEVICE_PTR_BASE]], Size=288, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
  // CHECK: omptarget --> Moving 288 bytes (tgt:0x{{0*}}[[DAT_DEVICE_PTR_BASE]]) -> (hst:0x{{0*}}[[DAT_HST_PTR_BASE]])
  // clang-format on

  // CHECK: dat.xi = 4
  // CHECK: dat.val_datum = 8
  // CHECK: dat.val_more_datum = 18
  // CHECK: dat.datum[dat.arr[0][0]] = 0
  // CHECK: dat.val_arr = 4

  printf("dat.xi = %d\n", dat.xi);
  printf("dat.val_datum = %d\n", dat.val_datum);
  printf("dat.val_more_datum = %d\n", dat.val_more_datum);
  printf("dat.datum[dat.arr[0][0]] = %d\n", dat.datum[dat.arr[0][0]]);
  printf("dat.val_arr = %d\n", dat.val_arr);

  return 0;
}
