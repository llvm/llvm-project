; RUN: llc < %s -mtriple=mips64el -mcpu=mips64r6 -target-abi=n64 | FileCheck %s --check-prefixes=CHECK,R6,R6-64
; RUN: llc < %s -mtriple=mipsel -mcpu=mips32r6 -target-abi=o32 | FileCheck %s --check-prefixes=CHECK,R6,R6-O32
; RUN: llc < %s -mtriple=mips64el -mcpu=mips64r2 -target-abi=n64 | FileCheck %s --check-prefixes=CHECK,LIB,R2
; RUN: llc < %s -mtriple=mips64el -mcpu=mips64r6 -target-abi=n64 -mattr=+soft-float | FileCheck %s --check-prefixes=CHECK,LIB,SOFT
; RUN: llc < %s -mtriple=mipsel -mcpu=mips32r6 -target-abi=o32 -mattr=+micromips | FileCheck %s --check-prefixes=CHECK,MMR6
; RUN: llc < %s -mtriple=mipsel -mcpu=mips32r6 -mattr=+single-float -stop-after=finalize-isel | FileCheck %s --check-prefix=SINGLE
; RUN: llc < %s -mtriple=mips64el -mcpu=mips64r6 -target-abi=n64 -stop-after=finalize-isel | FileCheck %s --check-prefix=MIR
; RUN: llc < %s -mtriple=mips64el -mcpu=mips64r6 -target-abi=n64 -disable-strictnode-mutation | FileCheck %s --check-prefixes=CHECK,R6,R6-64

; Standard R6 selects fused instructions. Older targets, soft-float and
; microMIPS use libcalls. Single-float only permits native f32 operations.
; The FMA addend must occupy the destination FPR of MADDF/MSUBF.

define float @fma_f32(float %a, float %b, float %c) {
; CHECK-LABEL: fma_f32:
; MIR-LABEL: name: fma_f32
; MIR: nofpexcept MADDF_S
; SINGLE-LABEL: name: fma_f32
; SINGLE: nofpexcept MADDF_S
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

; Ordinary multiply/add must remain unfused on R6.
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

; Contract flags permit fusion but do not require it.
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

define double @fma_f64(double %a, double %b, double %c) {
; CHECK-LABEL: fma_f64:
; MIR-LABEL: name: fma_f64
; MIR: nofpexcept MADDF_D
; SINGLE-LABEL: name: fma_f64
; SINGLE: JAL &fma,
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

; Both strict results are unused: the operations must survive and retain
; their exception ordering around observe(). MIR checks reject nofpexcept;
; in assembly the first FMA executes in the call's delay slot.
define void @ordered_f32(float %a, float %b, float %c) strictfp {
; CHECK-LABEL: ordered_f32:
; R6: jal observe
; R6-NEXT: maddf.s
; R6: msubf.s
; MIR-LABEL: name: ordered_f32
; MIR: :fgr32 = MADDF_S
; MIR: JAL @observe
; MIR: :fgr32 = MSUBF_S
; SINGLE-LABEL: name: ordered_f32
; SINGLE: :fgr32 = MADDF_S
; SINGLE: JAL @observe
; SINGLE: :fgr32 = MSUBF_S
; CHECK: .end ordered_f32
  %r0 = call float @llvm.experimental.constrained.fma.f32(float %a, float %b, float %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  call void @observe()
  %neg = fneg float %a
  %r2 = call float @llvm.experimental.constrained.fma.f32(float %neg, float %b, float %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret void
}

define void @ordered_f64(double %a, double %b, double %c) strictfp {
; CHECK-LABEL: ordered_f64:
; R6: jal observe
; R6-NEXT: maddf.d
; R6: msubf.d
; MIR-LABEL: name: ordered_f64
; MIR: :fgr64 = MADDF_D
; MIR: JAL @observe
; MIR: :fgr64 = MSUBF_D
; SINGLE-LABEL: name: ordered_f64
; SINGLE: JAL &fma,
; SINGLE: JAL @observe
; SINGLE: JAL &fma,
; CHECK: .end ordered_f64
  %r0 = call double @llvm.experimental.constrained.fma.f64(double %a, double %b, double %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  call void @observe()
  %neg = fneg double %a
  %r2 = call double @llvm.experimental.constrained.fma.f64(double %neg, double %b, double %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret void
}

declare void @observe()
