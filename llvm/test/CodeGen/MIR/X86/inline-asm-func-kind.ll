; RUN: llc -mtriple=x86_64-linux-gnu -global-isel=0 -stop-after=finalize-isel %s -o %t.mir
; RUN: FileCheck %s --check-prefix=MIR < %t.mir
; RUN: llc -mtriple=x86_64-linux-gnu -run-pass=none %t.mir -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=x86_64-linux-gnu -start-after=finalize-isel %t.mir -o - | FileCheck %s
; RUN: llc -mtriple=x86_64-linux-gnu -global-isel=0 %s -o - | FileCheck %s
; RUN: llc -mtriple=x86_64-linux-gnu -global-isel=0 -print-after=finalize-isel -filetype=null %s 2>&1 | FileCheck %s --check-prefix=PRINT

; Test that the "func" flag of a function operand of an inline asm call, and
; its constraint, survive a MIR round trip and are printed.

declare void @g()

define void @test_p() {
; MIR-LABEL: name: test_p
; MIR: INLINEASM &"call ${0:P}", sideeffect attdialect, func:p ops:5, target-flags(x86-plt) @g, 1, $noreg, 0, $noreg,
;
; CHECK-LABEL: test_p:
; CHECK: callq g@PLT
;
; PRINT-LABEL: # Machine code for function test_p:
; PRINT: INLINEASM &"call ${0:P}" [sideeffect] [attdialect], $0:[func:p], target-flags(x86-plt) @g, 1, $noreg, 0, $noreg,
  call void asm sideeffect "call ${0:P}", "p,~{dirflag},~{fpsr},~{flags}"(ptr @g)
  ret void
}

; A "*m" call target is turned into an address, like "p".
define void @test_indirect_m() {
; MIR-LABEL: name: test_indirect_m
; MIR: INLINEASM &"call ${0:P}", sideeffect mayload attdialect, func:m ops:5, target-flags(x86-plt) @g, 1, $noreg, 0, $noreg,
;
; CHECK-LABEL: test_indirect_m:
; CHECK: callq g@PLT
;
; PRINT-LABEL: # Machine code for function test_indirect_m:
; PRINT: INLINEASM &"call ${0:P}" [sideeffect] [mayload] [attdialect], $0:[func:m], target-flags(x86-plt) @g, 1, $noreg, 0, $noreg,
  call void asm sideeffect "call ${0:P}", "*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(void ()) @g)
  ret void
}
