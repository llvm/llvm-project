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
; CHECK:       %cmp1 = icmp ne %0* %arg, null
; CHECK-NEXT:  %c1 = call %0* @llvm.ssa.copy.p0s_s.0(%0* %arg)
; CHECK-NEXT:  %c2 = call %1* @llvm.ssa.copy.p0s_s.1(%1* %tmp)
; CHECK-NEXT:  %c3a = call %0** @llvm.ssa.copy.p0p0s_s.0(%0** %arg2)
; CHECK-NEXT:  %c3b = call %0** @llvm.ssa.copy.p0p0s_s.0(%0** %arg2)
; CHECK-NEXT:  %c4a = call %1** @llvm.ssa.copy.p0p0s_s.1(%1** %tmp2)
; CHECK-NEXT:  %c4ba = call %1** @llvm.ssa.copy.p0p0s_s.1(%1** %tmp2)
; CHECK-NEXT:  %c5 = call %0*** @llvm.ssa.copy.p0p0p0s_s.0(%0*** %arg3)
; CHECK-NEXT:  %c6 = call %1*** @llvm.ssa.copy.p0p0p0s_s.1(%1*** %tmp3)

define void @test01(%0* %arg, %1* %tmp, %1** %tmp2, %0** %arg2, %1*** %tmp3, %0*** %arg3) {
bb:
  %cmp1 = icmp ne %0* %arg, null
  %c1 = call %0* @llvm.ssa.copy.p0s_s.0(%0* %arg)
  %c2 = call %1* @llvm.ssa.copy.p0s_s.1(%1* %tmp)
  %c3a = call %0** @llvm.ssa.copy.p0p0s_s.1(%0** %arg2)
  %c3b = call %0** @llvm.ssa.copy.p0p0s_s.1(%0** %arg2)
  %c4a = call %1** @llvm.ssa.copy.p0p0s_s.0(%1** %tmp2)
  %c4ba = call %1** @llvm.ssa.copy.p0p0s_s.0(%1** %tmp2)
  %c5 = call %0*** @llvm.ssa.copy.p0p0p0s_s.1(%0*** %arg3)
  %c6 = call %1*** @llvm.ssa.copy.p0p0p0s_s.0(%1*** %tmp3)
  ret void
}

declare %0* @llvm.ssa.copy.p0s_s.0(%0* returned)

declare %1* @llvm.ssa.copy.p0s_s.1(%1* returned)

declare %0** @llvm.ssa.copy.p0p0s_s.1(%0** returned)

declare %1** @llvm.ssa.copy.p0p0s_s.0(%1** returned)

declare %0*** @llvm.ssa.copy.p0p0p0s_s.1(%0*** returned)

declare %1*** @llvm.ssa.copy.p0p0p0s_s.0(%1*** returned)

; now with recycling of previous declarations:
; CHECK-LABEL: @test02(
; CHECK:       %cmp1 = icmp ne %0* %arg, null
; CHECK-NEXT:  %c4a = call %1** @llvm.ssa.copy.p0p0s_s.1(%1** %tmp2)
; CHECK-NEXT:  %c6 = call %1*** @llvm.ssa.copy.p0p0p0s_s.1(%1*** %tmp3)
; CHECK-NEXT:  %c1 = call %0* @llvm.ssa.copy.p0s_s.0(%0* %arg)
; CHECK-NEXT:  %c2 = call %1* @llvm.ssa.copy.p0s_s.1(%1* %tmp)
; CHECK-NEXT:  %c3b = call %0** @llvm.ssa.copy.p0p0s_s.0(%0** %arg2)
; CHECK-NEXT:  %c4ba = call %1** @llvm.ssa.copy.p0p0s_s.1(%1** %tmp2)
; CHECK-NEXT:  %c5 = call %0*** @llvm.ssa.copy.p0p0p0s_s.0(%0*** %arg3)

define void @test02(%0* %arg, %1* %tmp, %1** %tmp2, %0** %arg2, %1*** %tmp3, %0*** %arg3) {
bb:
  %cmp1 = icmp ne %0* %arg, null
  %c4a = call %1** @llvm.ssa.copy.p0p0s_s.0(%1** %tmp2)
  %c6 = call %1*** @llvm.ssa.copy.p0p0p0s_s.0(%1*** %tmp3)
  %c1 = call %0* @llvm.ssa.copy.p0s_s.0(%0* %arg)
  %c2 = call %1* @llvm.ssa.copy.p0s_s.1(%1* %tmp)
  %c3b = call %0** @llvm.ssa.copy.p0p0s_s.1(%0** %arg2)
  %c4ba = call %1** @llvm.ssa.copy.p0p0s_s.0(%1** %tmp2)
  %c5 = call %0*** @llvm.ssa.copy.p0p0p0s_s.1(%0*** %arg3)
  ret void
}

;--- f02.ll
%1 = type opaque
%2 = type opaque

; CHECK-LABEL: @test03(
; CHECK:      %cmp1 = icmp ne %3* %arg, null
; CHECK-NEXT: %c1 = call %3* @llvm.ssa.copy.p0s_s.2(%3* %arg)
; CHECK-NEXT: %c2 = call %2* @llvm.ssa.copy.p0s_s.3(%2* %tmp)
; CHECK-NEXT: %c3 = call %3** @llvm.ssa.copy.p0p0s_s.2(%3** %arg2)
; CHECK-NEXT: %c4 = call %2** @llvm.ssa.copy.p0p0s_s.3(%2** %tmp2)

define void @test03(%1* %tmp, %2* %arg, %1** %tmp2, %2** %arg2) {
bb:
  %cmp1 = icmp ne %2* %arg, null
  %c1 = call %2* @llvm.ssa.copy.p0s_s.0(%2* %arg)
  %c2 = call %1* @llvm.ssa.copy.p0s_s.1(%1* %tmp)
  %c3 = call %2** @llvm.ssa.copy.p0p0s_s.1(%2** %arg2)
  %c4 = call %1** @llvm.ssa.copy.p0p0s_s.0(%1** %tmp2)
  ret void
}

declare %2* @llvm.ssa.copy.p0s_s.0(%2* returned)

declare %1* @llvm.ssa.copy.p0s_s.1(%1* returned)

declare %2** @llvm.ssa.copy.p0p0s_s.1(%2** returned)

declare %1** @llvm.ssa.copy.p0p0s_s.0(%1** returned)
