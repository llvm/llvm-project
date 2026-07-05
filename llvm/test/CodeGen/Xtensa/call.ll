; RUN: llc --mtriple=xtensa < %s | FileCheck %s

declare i32 @external_function(i32)

define i32 @test_call_external(i32 %a) nounwind {
; CHECK-LABEL: test_call_external:
; CHECK:       # %bb.0:
; CHECK:       s32i  a0, a1, 0
; CHECK-NEXT:  l32r  a8, .LCPI0_0
; CHECK-NEXT:  callx0  a8
; CHECK-NEXT:  l32i  a0, a1, 0
; CHECK:       ret
  %1 = call i32 @external_function(i32 %a)
  ret i32 %1
}

define i32 @defined_function(i32 %a) nounwind {
; CHECK-LABEL: defined_function:
; CHECK:       # %bb.0:
; CHECK-NEXT:  addi  a2, a2, 1
; CHECK-NEXT:  ret
  %1 = add i32 %a, 1
  ret i32 %1
}

define i32 @test_call_defined(i32 %a) nounwind {
; CHECK-LABEL: test_call_defined:
; CHECK:       # %bb.0:
; CHECK:       s32i  a0, a1, 0
; CHECK-NEXT:  l32r  a8, .LCPI2_0
; CHECK-NEXT:  callx0  a8
; CHECK-NEXT:  l32i  a0, a1, 0
; CHECK:       ret
  %1 = call i32 @defined_function(i32 %a) nounwind
  ret i32 %1
}

define i32 @test_call_indirect(ptr %a, i32 %b) nounwind {
; CHECK-LABEL: test_call_indirect:
; CHECK:       # %bb.0:
; CHECK:       s32i  a0, a1, 0
; CHECK-NEXT:  or  a8, a2, a2
; CHECK-NEXT:  or  a2, a3, a3
; CHECK-NEXT:  callx0  a8
; CHECK-NEXT:  l32i  a0, a1, 0
; CHECK:       ret
  %1 = call i32 %a(i32 %b)
  ret i32 %1
}
