; RUN: llc -march=mips < %s | FileCheck %s
; RUN: llc -march=mips64el -mcpu=mips64r2 -mattr=n64 < %s | FileCheck %s

%struct.DWstruct = type { i32, i32 }

define i32 @A0(i32 %u, i32 %v) nounwind  {
entry:
; CHECK: multu 
; CHECK: mflo
; CHECK: mfhi
  %asmtmp = tail call %struct.DWstruct asm "multu $2,$3", "={lo},={hi},d,d"( i32 %u, i32 %v ) nounwind
  %asmresult = extractvalue %struct.DWstruct %asmtmp, 0
  %asmresult1 = extractvalue %struct.DWstruct %asmtmp, 1    ; <i32> [#uses=1]
  %res = add i32 %asmresult, %asmresult1
  ret i32 %res
}

@gi2 = external global i32
@gi1 = external global i32
@gi0 = external global i32
@gf0 = external global float
@gf1 = external global float
@gd0 = external global double
@gd1 = external global double

define void @foo0() nounwind {
entry:
; CHECK: addu
  %0 = load i32* @gi1, align 4
  %1 = load i32* @gi0, align 4
  %2 = tail call i32 asm "addu $0, $1, $2", "=r,r,r"(i32 %0, i32 %1) nounwind
  store i32 %2, i32* @gi2, align 4
  ret void
}

define void @foo2() nounwind {
entry:
; CHECK: neg.s
  %0 = load float* @gf1, align 4
  %1 = tail call float asm "neg.s $0, $1", "=f,f"(float %0) nounwind
  store float %1, float* @gf0, align 4
  ret void
}

define void @foo3() nounwind {
entry:
; CHECK: neg.d
  %0 = load double* @gd1, align 8
  %1 = tail call double asm "neg.d $0, $1", "=f,f"(double %0) nounwind
  store double %1, double* @gd0, align 8
  ret void
}

