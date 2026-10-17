; RUN: llc < %s -mtriple=mips64el -mcpu=mips64r6 -target-abi=n64 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,R6-64,R6-N64
; RUN: llc < %s -mtriple=mips64 -mcpu=mips64r6 -target-abi=n32 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,R6-64,R6-N32
; RUN: llc < %s -mtriple=mipsel -mcpu=mips32r6 -target-abi=o32 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,R6-O32
; RUN: llc < %s -mtriple=mips64el -mcpu=mips64r2 -target-abi=n64 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,LIB,R2
; RUN: llc < %s -mtriple=mips64el -mcpu=mips64r6 -target-abi=n64 -mattr=+soft-float -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,LIB,SOFT
; RUN: llc < %s -mtriple=mipsel -mcpu=mips32r6 -target-abi=o32 -mattr=+micromips -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,MMR6

; Standard R6 selects fused instructions. Older targets, soft-float and
; microMIPS use libcalls. Contract flags alone do not force fusion.
define float @fma_f32(float %a, float %b, float %c) {
; CHECK-LABEL: fma_f32:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mov.s $f0, $f14
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    maddf.s $f0, $f12, $f13
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    mtc1 $6, $f0
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    maddf.s $f0, $f12, $f14
; LIB: jal fmaf
; MMR6: balc fmaf
; CHECK: .end fma_f32
  %r = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}

define float @acc_first_f32(float %c, float %a, float %b) {
; CHECK-LABEL: acc_first_f32:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mov.s $f0, $f12
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    maddf.s $f0, $f13, $f14
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    mov.s $f0, $f12
; R6-O32-NEXT:    mtc1 $6, $f1
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    maddf.s $f0, $f14, $f1
; LIB: jal fmaf
; MMR6: balc fmaf
; CHECK: .end acc_first_f32
  %r = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}

define float @acc_reused_f32(float %a, float %b, float %c, ptr %p) {
; CHECK-LABEL: acc_reused_f32:
; R6-N64:       # %bb.0:
; R6-N64-NEXT:    mov.s $f0, $f14
; R6-N64-NEXT:    swc1 $f14, 0($7)
; R6-N64-NEXT:    jr $ra
; R6-N64-NEXT:    maddf.s $f0, $f12, $f13
; R6-N32:       # %bb.0:
; R6-N32-NEXT:    mov.s $f0, $f14
; R6-N32-NEXT:    sll $1, $7, 0
; R6-N32-NEXT:    swc1 $f14, 0($1)
; R6-N32-NEXT:    jr $ra
; R6-N32-NEXT:    maddf.s $f0, $f12, $f13
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    sw $6, 0($7)
; R6-O32-NEXT:    mtc1 $6, $f0
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    maddf.s $f0, $f12, $f14
; LIB: jal fmaf
; MMR6: balc fmaf
; CHECK: .end acc_reused_f32
  %r = call float @llvm.fma.f32(float %a, float %b, float %c)
  store volatile float %c, ptr %p
  ret float %r
}

define float @acc_alias_f32(float %a, float %b) {
; CHECK-LABEL: acc_alias_f32:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mov.s $f0, $f12
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    maddf.s $f0, $f12, $f13
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    mov.s $f0, $f12
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    maddf.s $f0, $f12, $f14
; LIB: jal fmaf
; MMR6: balc fmaf
; CHECK: .end acc_alias_f32
  %r = call float @llvm.fma.f32(float %a, float %b, float %a)
  ret float %r
}

define float @msub_f32(float %a, float %b, float %c) {
; CHECK-LABEL: msub_f32:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mov.s $f0, $f14
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    msubf.s $f0, $f12, $f13
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    mtc1 $6, $f0
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    msubf.s $f0, $f12, $f14
; LIB: jal fmaf
; MMR6: balc fmaf
; CHECK: .end msub_f32
  %na = fneg float %a
  %r = call float @llvm.fma.f32(float %na, float %b, float %c)
  ret float %r
}

define float @msub_commuted_f32(float %a, float %b, float %c) {
; CHECK-LABEL: msub_commuted_f32:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mov.s $f0, $f14
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    msubf.s $f0, $f13, $f12
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    mtc1 $6, $f0
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    msubf.s $f0, $f14, $f12
; LIB: jal fmaf
; MMR6: balc fmaf
; CHECK: .end msub_commuted_f32
  %nb = fneg float %b
  %r = call float @llvm.fma.f32(float %a, float %nb, float %c)
  ret float %r
}

define float @plain_f32(float %a, float %b, float %c) {
; CHECK-LABEL: plain_f32:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mul.s $f0, $f12, $f13
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    add.s $f0, $f0, $f14
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    mul.s $f0, $f12, $f14
; R6-O32-NEXT:    mtc1 $6, $f1
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    add.s $f0, $f0, $f1
; R2: madd.s
; SOFT: jal __mulsf3
; SOFT: jal __addsf3
; MMR6: mul.s
; MMR6: add.s
; CHECK: .end plain_f32
  %m = fmul float %a, %b
  %r = fadd float %m, %c
  ret float %r
}

define float @contract_f32(float %a, float %b, float %c) {
; CHECK-LABEL: contract_f32:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mul.s $f0, $f12, $f13
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    add.s $f0, $f0, $f14
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    mul.s $f0, $f12, $f14
; R6-O32-NEXT:    mtc1 $6, $f1
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    add.s $f0, $f0, $f1
; R2: madd.s
; SOFT: jal __mulsf3
; SOFT: jal __addsf3
; MMR6: mul.s
; MMR6: add.s
; CHECK: .end contract_f32
  %m = fmul contract float %a, %b
  %r = fadd contract float %m, %c
  ret float %r
}

define float @strict_f32(float %a, float %b, float %c) strictfp {
; CHECK-LABEL: strict_f32:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mov.s $f0, $f14
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    maddf.s $f0, $f12, $f13
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    mtc1 $6, $f0
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    maddf.s $f0, $f12, $f14
; LIB: jal fmaf
; MMR6: balc fmaf
; CHECK: .end strict_f32
  %r = call float @llvm.experimental.constrained.fma.f32(float %a, float %b, float %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret float %r
}

define float @strict_msub_f32(float %a, float %b, float %c) strictfp {
; CHECK-LABEL: strict_msub_f32:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mov.s $f0, $f14
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    msubf.s $f0, $f12, $f13
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    mtc1 $6, $f0
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    msubf.s $f0, $f12, $f14
; LIB: jal fmaf
; MMR6: balc fmaf
; CHECK: .end strict_msub_f32
  %na = fneg float %a
  %r = call float @llvm.experimental.constrained.fma.f32(float %na, float %b, float %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret float %r
}

define double @fma_f64(double %a, double %b, double %c) {
; CHECK-LABEL: fma_f64:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mov.d $f0, $f14
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    maddf.d $f0, $f12, $f13
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    ldc1 $f0, 16($sp)
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    maddf.d $f0, $f12, $f14
; LIB: jal fma{{$}}
; MMR6: balc fma{{$}}
; CHECK: .end fma_f64
  %r = call double @llvm.fma.f64(double %a, double %b, double %c)
  ret double %r
}

define double @acc_first_f64(double %c, double %a, double %b) {
; CHECK-LABEL: acc_first_f64:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mov.d $f0, $f12
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    maddf.d $f0, $f13, $f14
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    mov.d $f0, $f12
; R6-O32-NEXT:    ldc1 $f1, 16($sp)
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    maddf.d $f0, $f14, $f1
; LIB: jal fma{{$}}
; MMR6: balc fma{{$}}
; CHECK: .end acc_first_f64
  %r = call double @llvm.fma.f64(double %a, double %b, double %c)
  ret double %r
}

define double @acc_reused_f64(double %a, double %b, double %c, ptr %p) {
; CHECK-LABEL: acc_reused_f64:
; R6-N64:       # %bb.0:
; R6-N64-NEXT:    mov.d $f0, $f14
; R6-N64-NEXT:    sdc1 $f14, 0($7)
; R6-N64-NEXT:    jr $ra
; R6-N64-NEXT:    maddf.d $f0, $f12, $f13
; R6-N32:       # %bb.0:
; R6-N32-NEXT:    mov.d $f0, $f14
; R6-N32-NEXT:    sll $1, $7, 0
; R6-N32-NEXT:    sdc1 $f14, 0($1)
; R6-N32-NEXT:    jr $ra
; R6-N32-NEXT:    maddf.d $f0, $f12, $f13
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    lw $1, 24($sp)
; R6-O32-NEXT:    ldc1 $f0, 16($sp)
; R6-O32-NEXT:    sdc1 $f0, 0($1)
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    maddf.d $f0, $f12, $f14
; LIB: jal fma{{$}}
; MMR6: balc fma{{$}}
; CHECK: .end acc_reused_f64
  %r = call double @llvm.fma.f64(double %a, double %b, double %c)
  store volatile double %c, ptr %p
  ret double %r
}

define double @acc_alias_f64(double %a, double %b) {
; CHECK-LABEL: acc_alias_f64:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mov.d $f0, $f12
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    maddf.d $f0, $f12, $f13
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    mov.d $f0, $f12
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    maddf.d $f0, $f12, $f14
; LIB: jal fma{{$}}
; MMR6: balc fma{{$}}
; CHECK: .end acc_alias_f64
  %r = call double @llvm.fma.f64(double %a, double %b, double %a)
  ret double %r
}

define double @msub_f64(double %a, double %b, double %c) {
; CHECK-LABEL: msub_f64:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mov.d $f0, $f14
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    msubf.d $f0, $f12, $f13
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    ldc1 $f0, 16($sp)
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    msubf.d $f0, $f12, $f14
; LIB: jal fma{{$}}
; MMR6: balc fma{{$}}
; CHECK: .end msub_f64
  %na = fneg double %a
  %r = call double @llvm.fma.f64(double %na, double %b, double %c)
  ret double %r
}

define double @msub_commuted_f64(double %a, double %b, double %c) {
; CHECK-LABEL: msub_commuted_f64:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mov.d $f0, $f14
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    msubf.d $f0, $f13, $f12
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    ldc1 $f0, 16($sp)
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    msubf.d $f0, $f14, $f12
; LIB: jal fma{{$}}
; MMR6: balc fma{{$}}
; CHECK: .end msub_commuted_f64
  %nb = fneg double %b
  %r = call double @llvm.fma.f64(double %a, double %nb, double %c)
  ret double %r
}

define double @plain_f64(double %a, double %b, double %c) {
; CHECK-LABEL: plain_f64:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mul.d $f0, $f12, $f13
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    add.d $f0, $f0, $f14
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    mul.d $f0, $f12, $f14
; R6-O32-NEXT:    ldc1 $f1, 16($sp)
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    add.d $f0, $f0, $f1
; R2: madd.d
; SOFT: jal __muldf3
; SOFT: jal __adddf3
; MMR6: mul.d
; MMR6: add.d
; CHECK: .end plain_f64
  %m = fmul double %a, %b
  %r = fadd double %m, %c
  ret double %r
}

define double @contract_f64(double %a, double %b, double %c) {
; CHECK-LABEL: contract_f64:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mul.d $f0, $f12, $f13
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    add.d $f0, $f0, $f14
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    mul.d $f0, $f12, $f14
; R6-O32-NEXT:    ldc1 $f1, 16($sp)
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    add.d $f0, $f0, $f1
; R2: madd.d
; SOFT: jal __muldf3
; SOFT: jal __adddf3
; MMR6: mul.d
; MMR6: add.d
; CHECK: .end contract_f64
  %m = fmul contract double %a, %b
  %r = fadd contract double %m, %c
  ret double %r
}

define double @strict_f64(double %a, double %b, double %c) strictfp {
; CHECK-LABEL: strict_f64:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mov.d $f0, $f14
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    maddf.d $f0, $f12, $f13
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    ldc1 $f0, 16($sp)
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    maddf.d $f0, $f12, $f14
; LIB: jal fma{{$}}
; MMR6: balc fma{{$}}
; CHECK: .end strict_f64
  %r = call double @llvm.experimental.constrained.fma.f64(double %a, double %b, double %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret double %r
}

define double @strict_msub_f64(double %a, double %b, double %c) strictfp {
; CHECK-LABEL: strict_msub_f64:
; R6-64:       # %bb.0:
; R6-64-NEXT:    mov.d $f0, $f14
; R6-64-NEXT:    jr $ra
; R6-64-NEXT:    msubf.d $f0, $f12, $f13
; R6-O32:       # %bb.0:
; R6-O32-NEXT:    ldc1 $f0, 16($sp)
; R6-O32-NEXT:    jr $ra
; R6-O32-NEXT:    msubf.d $f0, $f12, $f14
; LIB: jal fma{{$}}
; MMR6: balc fma{{$}}
; CHECK: .end strict_msub_f64
  %na = fneg double %a
  %r = call double @llvm.experimental.constrained.fma.f64(double %na, double %b, double %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret double %r
}
