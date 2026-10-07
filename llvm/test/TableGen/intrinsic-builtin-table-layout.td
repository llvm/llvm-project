// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS | FileCheck %s

include "llvm/IR/Intrinsics.td"

def int_zeta : Intrinsic<[]>, ClangBuiltin<"__builtin_alpha">;
def int_alpha : Intrinsic<[]>, ClangBuiltin<"__builtin_beta">;

let TargetPrefix = "other" in
def int_other_zeta : Intrinsic<[]>, ClangBuiltin<"__builtin_other_one">;

let TargetPrefix = "target" in {
  def int_target_zeta : Intrinsic<[]>, ClangBuiltin<"__builtin_target_one">;
  def int_target_alpha : Intrinsic<[]>, ClangBuiltin<"__builtin_target_two">;
}

// CHECK-LABEL: struct BuiltinNameEntry {
// CHECK-NEXT: uint32_t StrTabOffset;
// CHECK-NOT: IntrinsicID
// CHECK: };
// CHECK: static_assert(sizeof(BuiltinNameEntry) == sizeof(uint32_t),
// CHECK: static_assert(sizeof(BuiltinNamesStorage) <=
// CHECK-LABEL: static constexpr BuiltinNameEntry BuiltinNameEntries[] = {
// CHECK-NEXT: // Target independent builtins.
// CHECK-NEXT: {{    \{[0-9]+\}, // __builtin_alpha}}
// CHECK-NEXT: {{    \{[0-9]+\}, // __builtin_beta}}
// CHECK-NEXT: // Builtins for other.
// CHECK-NEXT: {{    \{[0-9]+\}, // __builtin_other_one}}
// CHECK-NEXT: // Builtins for target.
// CHECK-NEXT: {{    \{[0-9]+\}, // __builtin_target_one}}
// CHECK-NEXT: {{    \{[0-9]+\}, // __builtin_target_two}}
// CHECK-NEXT: };
// CHECK: static_assert(2 <= uint16_t(-1),
// CHECK-LABEL: static constexpr uint16_t BuiltinIntrinsicIDs[] = {
// CHECK-NEXT: zeta, // __builtin_alpha
// CHECK-NEXT: alpha, // __builtin_beta
// CHECK-NEXT: other_zeta, // __builtin_other_one
// CHECK-NEXT: target_zeta, // __builtin_target_one
// CHECK-NEXT: target_alpha, // __builtin_target_two
// CHECK-NEXT: };
// CHECK: static_assert(target_zeta <= uint16_t(-1),
// CHECK: static_assert(std::size(BuiltinNameEntries) ==
// CHECK: static_assert(std::size(BuiltinIntrinsicIDs) <=
// CHECK-LABEL: struct TargetEntry {
// CHECK-NEXT: uint32_t TargetPrefixOffset;
// CHECK-NEXT: uint16_t EntryOffset;
// CHECK-NEXT: uint16_t NumEntries;
// CHECK-NEXT: uint32_t CommonPrefixOffset;
// CHECK: StringRef getTargetPrefix() const {
// CHECK-NEXT: return BuiltinNames[TargetPrefixOffset];
// CHECK: StringRef getCommonPrefix() const {
// CHECK-NEXT: return BuiltinNames[CommonPrefixOffset];
// CHECK: static_assert(sizeof(TargetEntry) ==
// CHECK-SAME: 3 * sizeof(uint32_t),
// CHECK: {{\{[0-9]+, 2, 1, [0-9]+\},}}
// CHECK: {{\{[0-9]+, 3, 2, [0-9]+\},}}
// CHECK: ArrayRef<BuiltinNameEntry> Names(BuiltinNameEntries + 0, 2);
// CHECK: auto II = llvm::lower_bound(Names, Suffix);
// CHECK: return static_cast<ID>(BuiltinIntrinsicIDs[0 +
// CHECK-NEXT: (II - Names.begin())]);
// CHECK: TI->getTargetPrefix() != TargetPrefix
// CHECK: BuiltinName.consume_front(TI->getCommonPrefix())
// CHECK: ArrayRef<BuiltinNameEntry> TargetNames(BuiltinNameEntries + TI->EntryOffset,
// CHECK-NEXT: TI->NumEntries);
// CHECK: auto II = llvm::lower_bound(TargetNames, BuiltinName);
// CHECK: BuiltinIntrinsicIDs[TI->EntryOffset +
// CHECK-NEXT: (II - TargetNames.begin())]);
