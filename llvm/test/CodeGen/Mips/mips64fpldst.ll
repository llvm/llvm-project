; RUN: llc  < %s -march=mips64el -mcpu=mips4 -target-abi n64 -relocation-model=pic | FileCheck %s -check-prefix=CHECK-N64
; RUN: llc  < %s -march=mips64el -mcpu=mips4 -target-abi n32 -relocation-model=pic | FileCheck %s -check-prefix=CHECK-N32
; RUN: llc  < %s -march=mips64el -mcpu=mips64 -target-abi n64 -relocation-model=pic | FileCheck %s -check-prefix=CHECK-N64
; RUN: llc  < %s -march=mips64el -mcpu=mips64 -target-abi n32 -relocation-model=pic | FileCheck %s -check-prefix=CHECK-N32

@f0 = common global float 0.000000e+00, align 4
@d0 = common global double 0.000000e+00, align 8
@f1 = common global float 0.000000e+00, align 4
@d1 = common global double 0.000000e+00, align 8

define float @funcfl1() nounwind readonly {
entry:
; CHECK-N64: funcfl1
; CHECK-N64: ld $[[R0:[0-9]+]], %got_disp(f0)
; CHECK-N64: lwc1 $f{{[0-9]+}}, 0($[[R0]])
; CHECK-N32: funcfl1
; CHECK-N32: lw $[[R0:[0-9]+]], %got_disp(f0)
; CHECK-N32: lwc1 $f{{[0-9]+}}, 0($[[R0]])
  %0 = load float, float* @f0, align 4
  ret float %0
}

define double @funcfl2() nounwind readonly {
entry:
; CHECK-N64: funcfl2
; CHECK-N64: ld $[[R0:[0-9]+]], %got_disp(d0)
; CHECK-N64: ldc1 $f{{[0-9]+}}, 0($[[R0]])
; CHECK-N32: funcfl2
; CHECK-N32: lw $[[R0:[0-9]+]], %got_disp(d0)
; CHECK-N32: ldc1 $f{{[0-9]+}}, 0($[[R0]])
  %0 = load double, double* @d0, align 8
  ret double %0
}

define void @funcfs1() nounwind {
entry:
; CHECK-N64: funcfs1
; CHECK-N64: ld $[[R0:[0-9]+]], %got_disp(f0)
; CHECK-N64: swc1 $f{{[0-9]+}}, 0($[[R0]])
; CHECK-N32: funcfs1
; CHECK-N32: lw $[[R0:[0-9]+]], %got_disp(f0)
; CHECK-N32: swc1 $f{{[0-9]+}}, 0($[[R0]])
  %0 = load float, float* @f1, align 4
  store float %0, float* @f0, align 4
  ret void
}

define void @funcfs2() nounwind {
entry:
; CHECK-N64: funcfs2
; CHECK-N64: ld $[[R0:[0-9]+]], %got_disp(d0)
; CHECK-N64: sdc1 $f{{[0-9]+}}, 0($[[R0]])
; CHECK-N32: funcfs2
; CHECK-N32: lw $[[R0:[0-9]+]], %got_disp(d0)
; CHECK-N32: sdc1 $f{{[0-9]+}}, 0($[[R0]])
  %0 = load double, double* @d1, align 8
  store double %0, double* @d0, align 8
  ret void
}

