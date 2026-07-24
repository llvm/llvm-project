// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s 2>&1 | FileCheck %s
// Verify that ProcessorAlias entries are emitted into a separate, compact alias
// table (mapping alias name -> canonical processor index) rather than as full
// SubtargetSubTypeKV entries, and that both tables stay sorted by key.

include "llvm/Target/Target.td"

def MyTarget : Target;

def FeatureA : SubtargetFeature<"feature-a", "HasA", "true", "">;

def ProcA : ProcessorModel<"cpu-a", NoSchedModel, [FeatureA]>;
def ProcB : ProcessorModel<"cpu-b", NoSchedModel, []>;

// Two aliases pointing at cpu-a (index 0). One sorts before its canonical name,
// the other after, so the alias table order is independent of cpu-a's position.
def : ProcessorAlias<"alias-of-a", "cpu-a">;
def : ProcessorAlias<"zzz-of-a", "cpu-a">;

// The subtype table holds only the 2 real processors; the storage template
// records 2 subtypes and 2 aliases.
// CHECK: static constexpr size_t MyTargetCPUAliasArraySize = sizeof(std::array<SubtargetSubTypeAliasKV, 2>);
// CHECK: extern const llvm::SubtargetSubTypeKVStorage< 2, 2, {{[0-9]+}}> MyTargetSubTypeKVStorage
// CHECK:      { sizeof(SubtargetSubTypeKV) * 2 + MyTargetCPUAliasArraySize + {{[0-9]+}}, { { { 0x1ULL,{{.*}} }, {{.*}}, 0 },
// CHECK-NEXT: { sizeof(SubtargetSubTypeKV) * 1 + MyTargetCPUAliasArraySize + {{[0-9]+}}, { { { 0x0ULL,{{.*}} }, {{.*}}, 0 },

// The alias table is sorted by alias name (alias-of-a, zzz-of-a); both resolve
// to subtype index 0 (cpu-a).
// CHECK:      { sizeof(SubtargetSubTypeAliasKV) * 2 + {{[0-9]+}}, 0 },
// CHECK-NEXT: { sizeof(SubtargetSubTypeAliasKV) * 1 + {{[0-9]+}}, 0 },
// CHECK: "\000cpu-a\000cpu-b\000alias-of-a\000zzz-of-a\000"
