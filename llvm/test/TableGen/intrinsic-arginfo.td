// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s | FileCheck %s

// Test ArgInfo property for pretty-printing intrinsic arguments.
// This test verifies that TableGen generates the correct pretty-printing code
// for intrinsics that use the ArgInfo property.

include "llvm/IR/Intrinsics.td"

// Simple intrinsic with two arguments that have ArgInfo.
def int_dummy_foo_bar : DefaultAttrsIntrinsic<
    [llvm_i32_ty],
    [llvm_i32_ty,      // data
     llvm_i32_ty,      // mode
     llvm_i32_ty],     // stride
    [IntrNoMem,
     ImmArg<ArgIndex<1>>,
     ArgInfo<ArgIndex<1>, [ArgName<"mode">, ImmArgPrinter<"printDummyMode">]>,
     ArgInfo<ArgIndex<2>, [ArgName<"stride">]>]>;

// A custom floating point add with rounding and sat mode.
def int_my_fadd_f32 : DefaultAttrsIntrinsic<
    [llvm_float_ty],
    [llvm_float_ty,    // a
     llvm_float_ty,    // b
     llvm_i32_ty,      // rounding_mode
     llvm_i1_ty],      // saturation_mode
    [IntrNoMem,
     ImmArg<ArgIndex<2>>,
     ImmArg<ArgIndex<3>>,
     ArgInfo<ArgIndex<2>, [ArgName<"rounding_mode">, ImmArgPrinter<"printRoundingMode">]>,
     ArgInfo<ArgIndex<3>, [ArgName<"saturation_mode">]>]>;

// CHECK: #ifdef GET_INTRINSIC_PRETTY_PRINT_TABLE
// CHECK-NEXT: static constexpr uint8_t PPTable[] = {

// CHECK: #endif // GET_INTRINSIC_PRETTY_PRINT_TABLE

// CHECK: #ifdef GET_INTRINSIC_PRETTY_PRINT_ARGUMENTS
// CHECK: void Intrinsic::printImmArg(ID IID, unsigned ArgIdx, raw_ostream &OS, const Constant *ImmArgVal) {

// CHECK: case dummy_foo_bar:
// CHECK-NEXT: switch (ArgIdx) {

// CHECK-NEXT: case 1:
// CHECK-NEXT: OS << "mode=";
// CHECK-NEXT: printDummyMode(OS, ImmArgVal);
// CHECK-NEXT: return;

// CHECK-NEXT: case 2:
// CHECK-NEXT: OS << "stride=";
// CHECK-NEXT: return;

// CHECK-NEXT: }
// CHECK-NEXT: break;

// CHECK: case my_fadd_f32:
// CHECK-NEXT: switch (ArgIdx) {

// CHECK-NEXT: case 2:
// CHECK-NEXT: OS << "rounding_mode=";
// CHECK-NEXT: printRoundingMode(OS, ImmArgVal);
// CHECK-NEXT: return;

// CHECK-NEXT: case 3:
// CHECK-NEXT: OS << "saturation_mode=";
// CHECK-NEXT: return;

// CHECK-NEXT: }
// CHECK-NEXT: break;

// CHECK: #endif // GET_INTRINSIC_PRETTY_PRINT_ARGUMENTS
