; RUN: llc < %s -mtriple=avr | FileCheck %s

; The avr-rust bug can be found here:
; https://github.com/avr-rust/rust/issues/112
;
; In this test, the codegen stage generates a FRMIDX
; instruction. Later in the pipeline, the frame index
; gets expanded into a 16-bit MOVWRdRr instruction.
;
; There was a bug in the FRMIDX->MOVWRdRr expansion logic
; that could leave the MOVW instruction with an extraneous
; operand, left over from the original FRMIDX.
;
; This would trigger an assertion:
;
;   Assertion failed: ((isImpReg || Op.isRegMask() || MCID->isVariadic() ||
;                       OpNo < MCID->getNumOperands() || isMetaDataOp) &&
;                       "Trying to add an operand to a machine instr that is already done!"),
;   function addOperand, file llvm/lib/CodeGen/MachineInstr.cpp
;
; The logic has since been fixed.

; CHECK-LABEL: "core::str::slice_error_fail"
define void @"core::str::slice_error_fail"(i16 %arg) personality ptr addrspace(1) @rust_eh_personality {
start:
  %char_range = alloca { i16, i16 }, align 1
  br i1 undef, label %"<core::option::Option<T>>::unwrap.exit.thread", label %bb11.i.i

"<core::option::Option<T>>::unwrap.exit.thread":
  br label %"core::char::methods::<impl char>::len_utf8.exit"

bb11.i.i:
  %tmp = bitcast ptr %char_range to ptr
  %tmp1 = icmp ult i32 undef, 65536
  %..i = select i1 %tmp1, i16 3, i16 4
  br label %"core::char::methods::<impl char>::len_utf8.exit"

"core::char::methods::<impl char>::len_utf8.exit":
  %tmp2 = phi ptr [ %tmp, %bb11.i.i ], [ undef, %"<core::option::Option<T>>::unwrap.exit.thread" ]
  %_0.0.i12 = phi i16 [ %..i, %bb11.i.i ], [ 1, %"<core::option::Option<T>>::unwrap.exit.thread" ]
  %tmp3 = add i16 %_0.0.i12, %arg
  store i16 %tmp3, ptr undef, align 1
  store ptr %tmp2, ptr undef, align 1
  unreachable
}

declare i32 @rust_eh_personality(...) addrspace(1)

