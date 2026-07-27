// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s 2>&1 | FileCheck %s
// Verify that ProcessorAlias entries are emitted into the CPU subtype table
// alongside real processors, resolving to their canonical processor's
// features and scheduling model, and that the table stays sorted by key.

include "llvm/Target/Target.td"

def MyTarget : Target;

def FeatureA : SubtargetFeature<"feature-a", "HasA", "true", "">;

def ProcA : ProcessorModel<"cpu-a", NoSchedModel, [FeatureA]>;
def ProcB : ProcessorModel<"cpu-b", NoSchedModel, []>;

// An alias resolves to an existing processor and is emitted as its own entry.
def : ProcessorAlias<"alias-of-a", "cpu-a">;

// The subtype table has 3 entries (2 processors + 1 alias) sorted by key:
// alias-of-a, cpu-a, cpu-b. The alias carries cpu-a's feature mask (bit 0 set).
// CHECK: extern const llvm::SubtargetSubTypeKVStorage< 3,
// CHECK:      { sizeof(SubtargetSubTypeKV) * 3 + {{[0-9]+}}, { { { 0x1ULL,{{.*}} }, {{.*}}, 0 },
// CHECK-NEXT: { sizeof(SubtargetSubTypeKV) * 2 + {{[0-9]+}}, { { { 0x1ULL,{{.*}} }, {{.*}}, 0 },
// CHECK-NEXT: { sizeof(SubtargetSubTypeKV) * 1 + {{[0-9]+}}, { { { 0x0ULL,{{.*}} }, {{.*}}, 0 },
// CHECK: "\000alias-of-a\000cpu-a\000cpu-b\000"
