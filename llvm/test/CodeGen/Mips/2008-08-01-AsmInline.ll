; RUN: llc -march=mips -mcpu=mips32 < %s | FileCheck %s
; RUN: llc -march=mips64el -mcpu=mips64r2 -target-abi=n64 < %s | FileCheck %s

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
  %0 = load i32, ptr @gi1, align 4
  %1 = load i32, ptr @gi0, align 4
  %2 = tail call i32 asm "addu $0, $1, $2", "=r,r,r"(i32 %0, i32 %1) nounwind
  store i32 %2, ptr @gi2, align 4
  ret void
}

define void @foo2() nounwind {
entry:
; CHECK: neg.s
  %0 = load float, ptr @gf1, align 4
  %1 = tail call float asm "neg.s $0, $1", "=f,f"(float %0) nounwind
  store float %1, ptr @gf0, align 4
  ret void
}

define void @foo3() nounwind {
entry:
; CHECK: neg.d
  %0 = load double, ptr @gd1, align 8
  %1 = tail call double asm "neg.d $0, $1", "=f,f"(double %0) nounwind
  store double %1, ptr @gd0, align 8
  ret void
}

; Check that RA doesn't allocate registers in the clobber list.
; CHECK-LABEL: foo4:
; CHECK: #APP
; CHECK-NOT: ulh $2
; CHECK: #NO_APP
; CHECK: #APP
; CHECK-NOT: $f0
; CHECK: #NO_APP

define void @foo4() {
entry:
  %0 = tail call i32 asm sideeffect "ulh $0,16($$sp)\0A\09", "=r,~{$2}"()
  store i32 %0, ptr @gi2, align 4
  %1 = load float, ptr @gf0, align 4
  %2 = tail call double asm sideeffect "cvt.d.s $0, $1\0A\09", "=f,f,~{$f0}"(float %1)
  store double %2, ptr @gd0, align 8
  ret void
}
