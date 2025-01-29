// clang-format off
// RUN: %libomptarget-compile-generic -DREQ=1 && %libomptarget-run-generic 2>&1 | %fcheck-generic -check-prefix=GOOD
// RUN: %libomptarget-compile-generic -DREQ=2 && not %libomptarget-run-generic 2>&1 | %fcheck-generic -check-prefix=BAD
// clang-format on

/*
  Test for the 'requires' clause check.
  When a target region is used, the requires flags are set in the
  runtime for the entire compilation unit. If the flags are set again,
  (for whatever reason) the set must be consistent with previously
  set values.
*/
#include <omp.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Various definitions copied from OpenMP RTL

typedef struct {
  uint64_t Reserved;
  uint16_t Version;
  uint16_t Kind;
  uint32_t Flags;
  void *Address;
  char *SymbolName;
  uint64_t Size;
  uint64_t Data;
  void *AuxAddr;
} __tgt_offload_entry;

enum Flags {
  OMP_REGISTER_REQUIRES = 0x10,
};

typedef struct {
  void *ImageStart;
  void *ImageEnd;
  __tgt_offload_entry *EntriesBegin;
  __tgt_offload_entry *EntriesEnd;
} __tgt_device_image;

typedef struct {
  int32_t NumDeviceImages;
  __tgt_device_image *DeviceImages;
  __tgt_offload_entry *HostEntriesBegin;
  __tgt_offload_entry *HostEntriesEnd;
} __tgt_bin_desc;

void __tgt_register_lib(__tgt_bin_desc *Desc);
void __tgt_unregister_lib(__tgt_bin_desc *Desc);

// End of definitions copied from OpenMP RTL.
// ---------------------------------------------------------------------------

void run_reg_requires() {
  // Before the target region is registered, the requires registers the status
  // of the requires clauses. Since there are no requires clauses in this file
  // the flags state can only be OMP_REQ_NONE i.e. 1.

  // This is the 2nd time this function is called so it should print SUCCESS if
  // REQ is compatible with `1` and otherwise cause an error.
  __tgt_offload_entry entries[] = {
      {0, 0, 1, OMP_REGISTER_REQUIRES, NULL, "", 0, 1, NULL},
      {0, 0, 1, OMP_REGISTER_REQUIRES, NULL, "", 0, REQ, NULL}};
  __tgt_device_image image = {NULL, NULL, &entries[0], &entries[1] + 1};
  __tgt_bin_desc bin = {1, &image, &entries[0], &entries[1] + 1};

  __tgt_register_lib(&bin);

  printf("SUCCESS");

  __tgt_unregister_lib(&bin);

  // clang-format off
  // GOOD: SUCCESS
  // BAD: omptarget fatal error 2: '#pragma omp requires reverse_offload' not used consistently!
  // clang-format on
}

// ---------------------------------------------------------------------------
int main() {
  run_reg_requires();

// This also runs reg requires for the first time.
#pragma omp target
  {
  }

  return 0;
}
