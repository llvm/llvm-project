; RUN: opt -mtriple amdgpu7.00-- -passes='print<uniformity>' -disable-output < %s 2>&1 | FileCheck %s

; Making %c inreg makes it being passed in SGPR -> uniform Check that the
; callbr is still marked as divergent due to the call to the non-uniform kill
; intrinsic.
; CHECK-LABEL: 'test_callbr_divergent_call'
; CHECK-NOT: DIVERGENT: i1
; CHECK: DIVERGENT: callbr
define void @test_callbr_divergent_call(i1 inreg %c) {
  callbr void @llvm.amdgcn.kill(i1 %c) to label %cont [label %kill]
kill:
  unreachable
cont:
  ret void
}

; Although %id is divergent, the inline-asm/callbr isn't because it isn't
; actually doing anything.
; CHECK-LABEL: 'test_callbr_uniform_inline_asm'
; CHECK: DIVERGENT: %id
; CHECK-NOT: DIVERGENT: callbr
define i32 @test_callbr_uniform_inline_asm() {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  callbr void asm "", "r,!i"(i32 %id) to label %ret0 [label %ret1]
ret0:
  ret i32 0
ret1:
  ret i32 1
}

; CHECK-LABEL: 'test_callbr_divergent_inline_asm'
; CHECK-NOT: DIVERGENT: callbr
define i32 @test_callbr_divergent_inline_asm() {
  %val = callbr i32 asm "v_mov_b32 $0, 42","=v,!i"() to label %cont [label %ret]
cont:
  ret i32 %val
ret:
  ret i32 0
}

