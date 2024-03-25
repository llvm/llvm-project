; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN:   FileCheck --check-prefixes=CHECK,32BIT %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN:   FileCheck --check-prefixes=CHECK,64BIT %s

@gcd = external global { double, double }, align 8
@gcf = external global { float, float }, align 4
@gcfp128 = external global { ppc_fp128, ppc_fp128 }, align 16

declare void @anchor(...)

define dso_local { double, double } @dblCmplxRetCallee()  {
entry:
  %retval = alloca { double, double }, align 8
  %retval.realp = getelementptr inbounds { double, double }, ptr %retval, i32 0, i32 0
  store double 1.000000e+00, ptr %retval.realp, align 8
  %retval.imagp = getelementptr inbounds { double, double }, ptr %retval, i32 0, i32 1
  store double 0.000000e+00, ptr %retval.imagp, align 8
  %0 = load { double, double }, ptr %retval, align 8
  ret { double, double } %0
}

; CHECK-LABEL: .dblCmplxRetCallee:

; CHECK-DAG:   lfs 1,
; CHECK-DAG:   lfs 2,
; CHECK:       blr

define dso_local void @dblCmplxRetCaller()  {
entry:
  %call = call { double, double } @dblCmplxRetCallee()
  %0 = extractvalue { double, double } %call, 0
  %1 = extractvalue { double, double } %call, 1
  store double %0, ptr @gcd, align 8
  store double %1, ptr getelementptr inbounds ({ double, double }, ptr @gcd, i32 0, i32 1), align 8
  call void @anchor()
  ret void
}

; CHECK-LABEL: .dblCmplxRetCaller:

; CHECK:        bl .dblCmplxRetCallee
; 32BIT-NEXT:   lwz [[REG:[0-9]+]], L..C{{[0-9]+}}(2)
; 64BIT-NEXT:   ld [[REG:[0-9]+]], L..C{{[0-9]+}}(2)
; CHECK-DAG:    stfd 1, 0([[REG]])
; CHECK-DAG:    stfd 2, 8([[REG]])
; CHECK-NEXT:   bl .anchor

define dso_local { float, float } @fltCmplxRetCallee()  {
entry:
  %retval = alloca { float, float }, align 4
  %retval.realp = getelementptr inbounds { float, float }, ptr %retval, i32 0, i32 0
  %retval.imagp = getelementptr inbounds { float, float }, ptr %retval, i32 0, i32 1
  store float 1.000000e+00, ptr %retval.realp, align 4
  store float 0.000000e+00, ptr %retval.imagp, align 4
  %0 = load { float, float }, ptr %retval, align 4
  ret { float, float } %0
}

; CHECK-LABEL: .fltCmplxRetCallee:

; CHECK-DAG:   lfs 1,
; CHECK-DAG:   lfs 2,
; CHECK:       blr

define dso_local void @fltCmplxRetCaller()  {
entry:
  %call = call { float, float } @fltCmplxRetCallee()
  %0 = extractvalue { float, float } %call, 0
  %1 = extractvalue { float, float } %call, 1
  store float %0, ptr @gcf, align 4
  store float %1, ptr getelementptr inbounds ({ float, float }, ptr @gcf, i32 0, i32 1), align 4
  call void @anchor()
  ret void
}

; CHECK-LABEL: .fltCmplxRetCaller:

; CHECK:        bl .fltCmplxRetCallee
; 32BIT-NEXT:   lwz [[REG:[0-9]+]], L..C{{[0-9]+}}(2)
; 64BIT-NEXT:   ld [[REG:[0-9]+]], L..C{{[0-9]+}}(2)
; CHECK-DAG:    stfs 1, 0([[REG]])
; CHECK-DAG:    stfs 2, 4([[REG]])
; CHECK-NEXT:   bl .anchor

define dso_local { ppc_fp128, ppc_fp128 } @fp128CmplxRetCallee()  {
entry:
  %retval = alloca { ppc_fp128, ppc_fp128 }, align 16
  %retval.realp = getelementptr inbounds { ppc_fp128, ppc_fp128 }, ptr %retval, i32 0, i32 0
  %retval.imagp = getelementptr inbounds { ppc_fp128, ppc_fp128 }, ptr %retval, i32 0, i32 1
  store ppc_fp128 0xM7ffeffffffffffffffffffffffffffff, ptr %retval.realp, align 16
  store ppc_fp128 0xM3ffefffffffffffffffffffffffffffe, ptr %retval.imagp, align 16
  %0 = load { ppc_fp128, ppc_fp128 }, ptr %retval, align 16
  ret { ppc_fp128, ppc_fp128 } %0
}

; CHECK-LABEL: .fp128CmplxRetCallee:

; CHECK-DAG:  lfd 1,
; CHECK-DAG:  lfd 2,
; CHECK-DAG:  lfd 3,
; CHECK-DAG:  lfd 4,
; CHECK:      blr

define dso_local void @fp128CmplxRetCaller()  {
entry:
  %call = call { ppc_fp128, ppc_fp128 } @fp128CmplxRetCallee()
  %0 = extractvalue { ppc_fp128, ppc_fp128 } %call, 0
  %1 = extractvalue { ppc_fp128, ppc_fp128 } %call, 1
  store ppc_fp128 %0, ptr @gcfp128, align 16
  store ppc_fp128 %1, ptr getelementptr inbounds ({ ppc_fp128, ppc_fp128 }, ptr @gcfp128, i32 0, i32 1), align 16
  call void @anchor()
  ret void
}

; CHECK-LABEL: .fp128CmplxRetCaller:

; CHECK:        bl .fp128CmplxRetCallee
; 32BIT-NEXT:   lwz [[REG:[0-9]+]], L..C{{[0-9]+}}(2)
; 64BIT-NEXT:   ld [[REG:[0-9]+]], L..C{{[0-9]+}}(2)
; CHECK-DAG:    stfd 1, 0([[REG]])
; CHECK-DAG:    stfd 2, 8([[REG]])
; CHECK-DAG:    stfd 3, 16([[REG]])
; CHECK-DAG:    stfd 4, 24([[REG]])
; CHECK-NEXT:   bl .anchor
