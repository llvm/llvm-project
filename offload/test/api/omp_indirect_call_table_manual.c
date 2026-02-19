// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: intelgpu
#include <assert.h>
#include <omp.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Various definitions copied from OpenMP RTL

typedef struct {
  uint64_t Reserved;
  uint16_t Version;
  uint16_t Kind; // OpenMP==1
  uint32_t Flags;
  void *Address;
  char *SymbolName;
  uint64_t Size;
  uint64_t Data;
  void *AuxAddr;
} __tgt_offload_entry;

enum OpenMPOffloadingDeclareTargetFlags {
  /// Mark the entry global as having a 'link' attribute.
  OMP_DECLARE_TARGET_LINK = 0x01,
  /// Mark the entry global as being an indirectly callable function.
  OMP_DECLARE_TARGET_INDIRECT = 0x08,
  /// This is an entry corresponding to a requirement to be registered.
  OMP_REGISTER_REQUIRES = 0x10,
  /// Mark the entry global as being an indirect vtable.
  OMP_DECLARE_TARGET_INDIRECT_VTABLE = 0x20,
};

#pragma omp begin declare variant match(device = {kind(gpu)})
// Provided by the runtime.
void *__llvm_omp_indirect_call_lookup(void *host_ptr);
#pragma omp declare target to(__llvm_omp_indirect_call_lookup)                 \
    device_type(nohost)
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {kind(cpu)})
// We assume unified addressing on the CPU target.
void *__llvm_omp_indirect_call_lookup(void *host_ptr) { return host_ptr; }
#pragma omp end declare variant

#pragma omp begin declare target
void foo(int *i) { *i += 1; }
void bar(int *i) { *i += 10; }
void baz(int *i) { *i += 100; }
#pragma omp end declare target

typedef void (*fptr_t)(int *i);

// Dispatch Table - declare separately on host and device to avoid
// registering with the library; this also allows us to use separate
// names, which is convenient for debugging.  This dispatchTable is
// intended to mimic what Clang emits for C++ vtables.
fptr_t dispatchTable[] = {foo, bar, baz};
#pragma omp begin declare target device_type(nohost)
fptr_t GPUdispatchTable[] = {foo, bar, baz};
fptr_t *GPUdispatchTablePtr = GPUdispatchTable;
#pragma omp end declare target

// Define "manual" OpenMP offload entries, where we emit Clang
// offloading entry structure definitions in the appropriate ELF
// section.  This allows us to  emulate the offloading entries that Clang would
// normally emit for us

__attribute__((weak, section("llvm_offload_entries"), aligned(8)))
const __tgt_offload_entry __offloading_entry[] = {{
    0ULL,                               // Reserved
    1,                                  // Version
    1,                                  // Kind
    OMP_DECLARE_TARGET_INDIRECT_VTABLE, // Flags
    &dispatchTable,                     // Address
    "GPUdispatchTablePtr",              // SymbolName
    (size_t)(sizeof(dispatchTable)),    // Size
    0ULL,                               // Data
    NULL                                // AuxAddr
}};

// Mimic how Clang emits vtable pointers for C++ classes
typedef struct {
  fptr_t *dispatchPtr;
} myClass;

// ---------------------------------------------------------------------------
int main() {
  myClass obj_foo = {dispatchTable + 0};
  myClass obj_bar = {dispatchTable + 1};
  myClass obj_baz = {dispatchTable + 2};
  int aaa = 0;

#pragma omp target map(aaa) map(to : obj_foo, obj_bar, obj_baz)
  {
    // Lookup
    fptr_t *foo_ptr = __llvm_omp_indirect_call_lookup(obj_foo.dispatchPtr);
    fptr_t *bar_ptr = __llvm_omp_indirect_call_lookup(obj_bar.dispatchPtr);
    fptr_t *baz_ptr = __llvm_omp_indirect_call_lookup(obj_baz.dispatchPtr);
    foo_ptr[0](&aaa);
    bar_ptr[0](&aaa);
    baz_ptr[0](&aaa);
  }

  assert(aaa == 111);
  // CHECK: PASS
  printf("PASS\n");
  return 0;
}
