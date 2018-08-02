; RUN: llc -o - %s | FileCheck %s
; Check that selection dag legalization of fcopysign works in cases with
; different modes for the arguments.
target triple = "aarch64--"

declare fp128 @llvm.copysign.f128(fp128, fp128)

@val_float = global float zeroinitializer, align 4
@val_double = global double zeroinitializer, align 8
@val_fp128 = global fp128 zeroinitializer, align 16

; CHECK-LABEL: copysign0
; CHECK: ldr [[REG:x[0-9]+]], [x8, :lo12:val_double]
; CHECK: and [[ANDREG:x[0-9]+]], [[REG]], #0x8000000000000000
; CHECK: lsr x[[LSRREGNUM:[0-9]+]], [[ANDREG]], #56
; CHECK: bfxil w[[LSRREGNUM]], w{{[0-9]+}}, #0, #7
; CHECK: strb w[[LSRREGNUM]],
; CHECK: ldr q{{[0-9]+}},
define fp128 @copysign0() {
entry:
  %v = load double, double* @val_double, align 8
  %conv = fpext double %v to fp128
  %call = tail call fp128 @llvm.copysign.f128(fp128 0xL00000000000000007FFF000000000000, fp128 %conv) #2
  ret fp128 %call
}

; CHECK-LABEL: copysign1
; CHECK-DAG: ldr [[REG:q[0-9]+]], [x8, :lo12:val_fp128]
; CHECK-DAG: ldr [[REG:w[0-9]+]], [x8, :lo12:val_float]
; CHECK: and [[ANDREG:w[0-9]+]], [[REG]], #0x80000000
; CHECK: lsr w[[LSRREGNUM:[0-9]+]], [[ANDREG]], #24
; CHECK: bfxil w[[LSRREGNUM]], w{{[0-9]+}}, #0, #7
; CHECK: strb w[[LSRREGNUM]],
; CHECK: ldr q{{[0-9]+}},
define fp128@copysign1() {
entry:
  %v0 = load fp128, fp128* @val_fp128, align 16
  %v1 = load float, float* @val_float, align 4
  %conv = fpext float %v1 to fp128
  %call = tail call fp128 @llvm.copysign.f128(fp128 %v0, fp128 %conv)
  ret fp128 %call
}
