; RUN: opt -mtriple=amdgcn -passes="expand-fp<opt-level=0>" %s -S -o - | FileCheck --check-prefixes CHECK %s
; RUN: opt -mtriple=amdgcn -passes="expand-fp<opt-level=1>" %s -S -o - | FileCheck --check-prefixes CHECK,OPT1 %s

; Check the handling of potentially infinite numerators in the frem
; expansion at different optimization levels and with different
; fast-math flags.

; CHECK-LABEL: define float @frem_x_maybe_inf(float %x, float %y)
; CHECK: 2:
; CHECK: [[FABS:%.*]] = call float @llvm.fabs.f32(float %x)
; CHECK: [[FCMP:%.*]] = fcmp ult float [[FABS]], 0x7FF0000000000000
; CHECK-NEXT: %ret = select i1 [[FCMP]], float %{{.*}}, float 0x7FF8000000000000
; CHECK-NEXT: ret float %ret
; CHECK-LABEL: }
define float @frem_x_maybe_inf(float %x, float %y)  {
  %ret = frem float %x, %y
  ret float %ret
}

; OPT1-LABEL: define float @frem_x_assumed_non_inf(float %x, float %y)
; OPT1: 2:
; OPT1-NOT: call float @llvm.fabs.f32(float %x)
; OPT1-NOT: fcmp ult float [[FABS]], 0x7FF0000000000000
; OPT1: %ret = select i1 true, float %{{.*}}, float 0x7FF8000000000000
; OPT1-NEXT: ret float %ret
; OPT1-LABEL: }
; OPT0-LABEL: define float @frem_x_assumed_non_inf(float %x, float %y)
; OPT0: 2:
; OPT0: [[FABS:%.*]] = call float @llvm.fabs.f32(float %x)
; OPT0: [[FCMP:%.*]] = fcmp ult float [[FABS]], 0x7FF0000000000000
; OPT0-NEXT: %ret = select i1 [[FCMP]], float %{{.*}}, float 0x7FF8000000000000
; OPT0-NEXT: ret float %ret
; OPT0-LABEL: }
define float @frem_x_assumed_non_inf(float %x, float %y)  {
  %absx = call float @llvm.fabs.f32(float %x)
  %noninf = fcmp ult float %absx, 0x7FF0000000000000
  call void @llvm.assume(i1 %noninf)
  %ret = frem float %x, %y
  ret float %ret
}

; CHECK-LABEL: define float @frem_ninf(float %x, float %y)
; CHECK: 2:
; CHECK-NOT: call float @llvm.fabs.f32(float %x)
; CHECK-NOT: fcmp ult float [[FABS]], 0x7FF0000000000000
; CHECK: %ret = select ninf i1 true, float %{{.*}}, float 0x7FF8000000000000
; CHECK-NEXT: ret float %ret
; CHECK-LABEL: }
define float @frem_ninf(float %x, float %y)  {
  %ret = frem ninf float %x, %y
  ret float %ret
}
