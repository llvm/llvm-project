// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include -DTEST_INTRINSICS_SUPPRESS_DEFS %s | FileCheck %s

include "llvm/IR/Intrinsics.td"

def int_no_feature : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

let TargetFeatures = "$custom" in
def int_custom_feature : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

let TargetFeatures = "feat-a,$custom" in
def int_mixed_custom_feature : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

let TargetFeatures = "$custom,feat-b" in
def int_leading_custom_feature : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

let TargetFeatures = "feat-c,($custom|feat-d)" in
def int_middle_custom_feature : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_requires_feature : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
  RequiresTargetFeatures<"feat-a,(feat-b|feat-c)">;

let TargetFeatures = "feat-let-a,feat-let-b" in
def int_let_feature : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

// CHECK:      #ifdef GET_INTRINSIC_TARGET_FEATURES_TABLE
// CHECK-NEXT: #undef GET_INTRINSIC_TARGET_FEATURES_TABLE
// CHECK:      // Intrinsic ID to required target features table.
// CHECK:      static constexpr char IntrinsicTargetFeaturesTableStorage[] =
// CHECK-NEXT: "\0"
// CHECK-NEXT: "$custom\0"
// CHECK-NEXT: "$custom,feat-b\0"
// CHECK-NEXT: "feat-let-a,feat-let-b\0"
// CHECK-NEXT: "feat-c,($custom|feat-d)\0"
// CHECK-NEXT: "feat-a,$custom\0"
// CHECK-NEXT: "feat-a,(feat-b|feat-c)\0"
// CHECK:      static constexpr llvm::StringTable
// CHECK-NEXT: IntrinsicTargetFeaturesTable = IntrinsicTargetFeaturesTableStorage;
// CHECK:      static constexpr unsigned IntrinsicTargetFeaturesOffsetTable[] = {
// CHECK-NEXT:   0, // not_intrinsic
// CHECK-NEXT:   {{[0-9]+}}, // llvm.custom.feature
// CHECK-NEXT:   {{[0-9]+}}, // llvm.leading.custom.feature
// CHECK-NEXT:   {{[0-9]+}}, // llvm.let.feature
// CHECK-NEXT:   {{[0-9]+}}, // llvm.middle.custom.feature
// CHECK-NEXT:   {{[0-9]+}}, // llvm.mixed.custom.feature
// CHECK-NEXT:   0, // llvm.no.feature
// CHECK-NEXT:   {{[0-9]+}}, // llvm.requires.feature
// CHECK:      }; // IntrinsicTargetFeaturesOffsetTable
// CHECK:      #endif // GET_INTRINSIC_TARGET_FEATURES_TABLE
