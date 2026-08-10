; RUN: llvm-as -o - %s | llvm-dis -o - 2>&1 | FileCheck %s

; Make sure we can assemble and disassemble IR containing intrinsics with
; unnamed types.

%1 = type opaque
%0 = type opaque

; CHECK-LABEL: @f0(
; CHECK: %c1 = call ptr @llvm.ssa.copy.p0(ptr %arg)
; CHECK: %c2 = call ptr @llvm.ssa.copy.p0(ptr %tmp)
; CHECK: %c3 = call ptr @llvm.ssa.copy.p0(ptr %arg2)
; CHECK: %c4 = call ptr @llvm.ssa.copy.p0(ptr %tmp2)

define void @f0(%0* %arg, %1* %tmp, %1** %tmp2, %0** %arg2) {
bb:
  %cmp1 = icmp ne %0* %arg, null
  %c1 = call %0* @llvm.ssa.copy.p0s_s.0(%0* %arg)
  %c2 = call %1* @llvm.ssa.copy.p0s_s.1(%1* %tmp)
  %c3 = call %0** @llvm.ssa.copy.p0p0s_s.1(%0** %arg2)
  %c4 = call %1** @llvm.ssa.copy.p0p0s_s.0(%1** %tmp2)
  ret void
}

declare %0* @llvm.ssa.copy.p0s_s.0(%0* returned)

declare %1* @llvm.ssa.copy.p0s_s.1(%1* returned)

declare %0** @llvm.ssa.copy.p0p0s_s.1(%0** returned)

declare %1** @llvm.ssa.copy.p0p0s_s.0(%1** returned)
