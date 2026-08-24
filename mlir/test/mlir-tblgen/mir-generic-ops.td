// Verify the -gen-mir-generic-ops backend consumes LLVM's generic (G_*) opcode
// definitions and emits well-formed MLIR ODS.
//
// RUN: mlir-tblgen -gen-mir-generic-ops -I %S/../../../llvm/include %s | FileCheck %s
//
// It also checks that the emitted ODS is consumable by ODS itself: feeding it
// back into -gen-op-decls must produce C++ op class declarations.
//
// RUN: mlir-tblgen -gen-mir-generic-ops -I %S/../../../llvm/include %s \
// RUN: | mlir-tblgen -gen-op-decls -I %S/../../include \
// RUN: | FileCheck --check-prefix=ODS %s

include "llvm/Target/Target.td"

// A pure, commutative binary op with two LLT operands and one LLT result.
// CHECK: def MIR_G_ADD : MIR_GenericOp<"g_add", [Pure, Commutative]> {
// CHECK-NEXT: let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
// CHECK-NEXT: let results = (outs MIR_AnyLLT:$dst);

// A load is not pure (mayLoad) and takes a pointer-typed operand.
// CHECK: def MIR_G_LOAD : MIR_GenericOp<"g_load", []> {
// CHECK-NEXT: let arguments = (ins MIR_AnyLLT:$addr);
// CHECK-NEXT: let results = (outs MIR_AnyLLT:$dst);

// PHI carries a variadic operand list.
// CHECK: def MIR_G_PHI : MIR_GenericOp<"g_phi", [Pure]> {
// CHECK-NEXT: let arguments = (ins Variadic<MIR_AnyLLT>:$varargs);

// A store has no results.
// CHECK: def MIR_G_STORE : MIR_GenericOp<"g_store", []> {
// CHECK: let results = (outs);

// UNMERGE has variadic results.
// CHECK: def MIR_G_UNMERGE_VALUES : MIR_GenericOp<"g_unmerge_values", [Pure]> {
// CHECK: let results = (outs MIR_AnyLLT:$dst0, Variadic<MIR_AnyLLT>:$outs);

// A C++ keyword operand name is sanitized.
// CHECK: def MIR_G_WRITE_REGISTER : MIR_GenericOp<"g_write_register", []> {
// CHECK-NEXT: let arguments = (ins MIR_AnyLLT:$value, AnyAttr:$register_);

// All 279 generic opcodes are emitted with none skipped.
// CHECK: Emitted 279 ops, skipped 0.

// The emitted ODS produces C++ op classes.
// ODS: class G_ADD
// ODS: class G_LOAD
