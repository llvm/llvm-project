// Verify the -gen-mir-target-ops backend collapses AArch64's selected
// instructions into one op per mnemonic and emits well-formed MLIR ODS.
//
// This parses the whole AArch64 target, so it is only run when the AArch64
// target's TableGen is available under the source tree.
//
// RUN: mlir-tblgen -gen-mir-target-ops --mir-target=AArch64 \
// RUN:   -I %S/../../../llvm/include -I %S/../../../llvm/lib/Target/AArch64 %s \
// RUN: | FileCheck %s

include "AArch64.td"

// A pure ALU op collapsed from its many encoding variants.
// CHECK: def AArch64MIR_add : AArch64MIR_TargetOp<"add", [Pure]> {
// CHECK-NEXT: let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
// CHECK: OptionalAttr<StrAttr>:$variant);
// CHECK-NEXT: let results = (outs Variadic<MIR_AnyLLT>:$rets);
// CHECK: // opcodes:

// A C++ keyword mnemonic keeps the mnemonic but sanitizes the class/def name.
// CHECK: def AArch64MIR_and_ : AArch64MIR_TargetOp<"and", [Pure]> {

// The conditional-branch mnemonic (AsmString "b.$cond") is sanitized.
// CHECK: def AArch64MIR_b_cond : AArch64MIR_TargetOp<"b_cond",

// The collapse produces on the order of ~1.5k ops from ~7.5k instructions.
// CHECK: emitted 1551 ops across 1551 mnemonics.
