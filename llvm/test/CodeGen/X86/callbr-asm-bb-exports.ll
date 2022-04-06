; REQUIRES: asserts, abi_breaking_checks
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -o /dev/null -debug-only=isel 2>&1 | FileCheck %s

; Make sure we emit the basic block exports and the TokenFactor before the
; inlineasm_br. Not sure how to get a MachineIR change so this reads the debug
; output from SelectionDAG.

; CHECK: t0: ch = EntryToken
; CHECK-NEXT: t16: i64 = BlockAddress<@test, %fail> 0
; CHECK-NEXT: t4: i32,ch = CopyFromReg t0, Register:i32 %3
; CHECK-NEXT: t10: i32 = add t4, Constant:i32<1>
; CHECK-NEXT: t12: ch = CopyToReg t0, Register:i32 %0, t10
; CHECK-NEXT: t6: i32,ch = CopyFromReg t0, Register:i32 %4
; CHECK-NEXT: t13: i32 = add t6, Constant:i32<1>
; CHECK-NEXT: t15: ch = CopyToReg t0, Register:i32 %1, t13
; CHECK-NEXT: t17: ch = TokenFactor t12, t15
; CHECK-NEXT: t2: i32,ch = CopyFromReg t0, Register:i32 %2
; CHECK-NEXT: t8: i32 = add t2, Constant:i32<4>
; CHECK-NEXT: t22: ch,glue = CopyToReg t17, Register:i32 %5, t8
; CHECK-NEXT: t30: ch,glue = inlineasm_br t22, TargetExternalSymbol:i64'xorl $0, $0; jmp ${1:l}', MDNode:ch<null>, TargetConstant:i64<8>, TargetConstant:i32<2293769>, Register:i32 %5, TargetConstant:i64<13>, TargetBlockAddress:i64<@test, %fail> 0, TargetConstant:i32<12>, Register:i32 $df, TargetConstant:i32<12>, Register:i16 $fpsw, TargetConstant:i32<12>, Register:i32 $eflags, t22:1

define i32 @test(i32 %a, i32 %b, i32 %c) {
entry:
  %0 = add i32 %a, 4
  %1 = add i32 %b, 1
  %2 = add i32 %c, 1
  callbr void asm "xorl $0, $0; jmp ${1:l}", "r,i,~{dirflag},~{fpsr},~{flags}"(i32 %0, i8* blockaddress(@test, %fail)) to label %normal [label %fail]

normal:
  ret i32 %1

fail:
  ret i32 %2
}
