; RUN: split-file %s %t
; RUN: llvm-as -o %t1.bc %t/f01.ll
; RUN: llvm-as -o %t2.bc %t/f02.ll
; RUN: llvm-link %t1.bc %t2.bc -o %t3.bc
; RUN: llvm-dis -o - %t3.bc | FileCheck %s

; Make sure we can link files with clashing intrinsic names using unnamed types.

;--- f01.ll
%1 = type opaque
%0 = type opaque

; CHECK-LABEL: @test01(
; CHECK:       %c1 = call %0 @llvm.ssa.copy.s_s.0(%0 %arg)
; CHECK-NEXT:  %c2 = call %1 @llvm.ssa.copy.s_s.1(%1 %tmp)

define void @test01(%0 %arg, %1 %tmp) {
bb:
  %c1 = call %0 @llvm.ssa.copy.s_s.0(%0 %arg)
  %c2 = call %1 @llvm.ssa.copy.s_s.1(%1 %tmp)
  ret void
}

declare %0 @llvm.ssa.copy.s_s.0(%0 returned)

declare %1 @llvm.ssa.copy.s_s.1(%1 returned)

; now with recycling of previous declarations:
; CHECK-LABEL: @test02(
; CHECK:       %c1 = call %0 @llvm.ssa.copy.s_s.0(%0 %arg)
; CHECK-NEXT:  %c2 = call %1 @llvm.ssa.copy.s_s.1(%1 %tmp)

define void @test02(%0 %arg, %1 %tmp) {
bb:
  %c1 = call %0 @llvm.ssa.copy.s_s.0(%0 %arg)
  %c2 = call %1 @llvm.ssa.copy.s_s.1(%1 %tmp)
  ret void
}

;--- f02.ll
%1 = type opaque
%2 = type opaque

; CHECK-LABEL: @test03(
; CHECK:      %c1 = call %3 @llvm.ssa.copy.s_s.2(%3 %arg)
; CHECK-NEXT: %c2 = call %2 @llvm.ssa.copy.s_s.3(%2 %tmp)

define void @test03(%1 %tmp, %2 %arg) {
bb:
  %c1 = call %2 @llvm.ssa.copy.s_s.0(%2 %arg)
  %c2 = call %1 @llvm.ssa.copy.s_s.1(%1 %tmp)
  ret void
}

declare %2 @llvm.ssa.copy.s_s.0(%2 returned)

declare %1 @llvm.ssa.copy.s_s.1(%1 returned)
