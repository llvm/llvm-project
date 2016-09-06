; RUN: llc -verify-machineinstrs  -mtriple=powerpc-unknown-linux-gnu -O0 < %s | FileCheck %s

@ld = common global ppc_fp128 0xM00000000000000000000000000000000, align 16
@ld2 = common global ppc_fp128 0xM00000000000000000000000000000000, align 16
@d = common global double 0.000000e+00, align 8
@f = common global float 0.000000e+00, align 4
@i = common global i32 0, align 4
@ui = common global i32 0, align 4
@var = common global i8 0, align 1

define void @foo() #0 {
entry:
  %c = alloca ppc_fp128, align 16
  %0 = load ppc_fp128, ppc_fp128* @ld, align 16
  %1 = load ppc_fp128, ppc_fp128* @ld2, align 16
  %add = fadd ppc_fp128 %0, %1
  store ppc_fp128 %add, ppc_fp128* %c, align 16
  %2 = load ppc_fp128, ppc_fp128* @ld, align 16
  %3 = load ppc_fp128, ppc_fp128* @ld2, align 16
  %sub = fsub ppc_fp128 %2, %3
  store ppc_fp128 %sub, ppc_fp128* %c, align 16
  %4 = load ppc_fp128, ppc_fp128* @ld, align 16
  %5 = load ppc_fp128, ppc_fp128* @ld2, align 16
  %mul = fmul ppc_fp128 %4, %5
  store ppc_fp128 %mul, ppc_fp128* %c, align 16
  %6 = load ppc_fp128, ppc_fp128* @ld, align 16
  %7 = load ppc_fp128, ppc_fp128* @ld2, align 16
  %div = fdiv ppc_fp128 %6, %7
  store ppc_fp128 %div, ppc_fp128* %c, align 16
  ret void

  ; CHECK-LABEL:    __gcc_qadd
  ; CHECK-LABEL:    __gcc_qsub
  ; CHECK-LABEL:    __gcc_qmul
  ; CHECK-LABEL:    __gcc_qdiv
}

define void @foo1() #0 {
entry:
  %0 = load double, double* @d, align 8
  %conv = fpext double %0 to ppc_fp128
  store ppc_fp128 %conv, ppc_fp128* @ld, align 16
  ret void

  ; CHECK-LABEL:    __gcc_dtoq
}

define void @foo2() #0 {
entry:
  %0 = load ppc_fp128, ppc_fp128* @ld, align 16
  %conv = fptrunc ppc_fp128 %0 to double
  store double %conv, double* @d, align 8
  ret void

  ; CHECK-LABEL:    __gcc_qtod
}

define void @foo3() #0 {
entry:
  %0 = load ppc_fp128, ppc_fp128* @ld, align 16
  %conv = fptrunc ppc_fp128 %0 to float
  store float %conv, float* @f, align 4
  ret void

  ; CHECK-LABEL:    __gcc_qtos
}

define void @foo4() #0 {
entry:
  %0 = load i32, i32* @i, align 4
  %conv = sitofp i32 %0 to ppc_fp128
  store ppc_fp128 %conv, ppc_fp128* @ld, align 16
  ret void

  ; CHECK-LABEL:    __gcc_itoq
}

define void @foo5() #0 {
entry:
  %0 = load i32, i32* @ui, align 4
  %conv = uitofp i32 %0 to ppc_fp128
  store ppc_fp128 %conv, ppc_fp128* @ld, align 16
  ret void

  ; CHECK-LABEL:    __gcc_utoq
}

define void @foo6() #0 {
entry:
  %0 = load ppc_fp128, ppc_fp128* @ld, align 16
  %1 = load ppc_fp128, ppc_fp128* @ld2, align 16
  %cmp = fcmp oeq ppc_fp128 %0, %1
  %conv = zext i1 %cmp to i32
  %conv1 = trunc i32 %conv to i8
  store i8 %conv1, i8* @var, align 1
  ret void

  ; CHECK-LABEL:    __gcc_qeq
}

define void @foo7() #0 {
entry:
  %0 = load ppc_fp128, ppc_fp128* @ld, align 16
  %1 = load ppc_fp128, ppc_fp128* @ld2, align 16
  %cmp = fcmp une ppc_fp128 %0, %1
  %conv = zext i1 %cmp to i32
  %conv1 = trunc i32 %conv to i8
  store i8 %conv1, i8* @var, align 1
  ret void

  ; CHECK-LABEL:    __gcc_qne
}

define void @foo8() #0 {
entry:
  %0 = load ppc_fp128, ppc_fp128* @ld, align 16
  %1 = load ppc_fp128, ppc_fp128* @ld2, align 16
  %cmp = fcmp ogt ppc_fp128 %0, %1
  %conv = zext i1 %cmp to i32
  %conv1 = trunc i32 %conv to i8
  store i8 %conv1, i8* @var, align 1
  ret void

  ; CHECK-LABEL:    __gcc_qgt
}

define void @foo9() #0 {
entry:
  %0 = load ppc_fp128, ppc_fp128* @ld, align 16
  %1 = load ppc_fp128, ppc_fp128* @ld2, align 16
  %cmp = fcmp olt ppc_fp128 %0, %1
  %conv = zext i1 %cmp to i32
  %conv1 = trunc i32 %conv to i8
  store i8 %conv1, i8* @var, align 1
  ret void

  ; CHECK-LABEL:    __gcc_qlt
}

define void @foo10() #0 {
entry:
  %0 = load ppc_fp128, ppc_fp128* @ld, align 16
  %1 = load ppc_fp128, ppc_fp128* @ld2, align 16
  %cmp = fcmp ole ppc_fp128 %0, %1
  %conv = zext i1 %cmp to i32
  %conv1 = trunc i32 %conv to i8
  store i8 %conv1, i8* @var, align 1
  ret void

  ; CHECK-LABEL:    __gcc_qle
}

define void @foo11() #0 {
entry:
  %0 = load ppc_fp128, ppc_fp128* @ld, align 16
  %1 = load ppc_fp128, ppc_fp128* @ld, align 16
  %cmp = fcmp une ppc_fp128 %0, %1
  %conv = zext i1 %cmp to i32
  %conv1 = trunc i32 %conv to i8
  store i8 %conv1, i8* @var, align 1
  ret void

  ; CHECK-LABEL:    __gcc_qunord
}

define void @foo12() #0 {
entry:
  %0 = load ppc_fp128, ppc_fp128* @ld, align 16
  %1 = load ppc_fp128, ppc_fp128* @ld2, align 16
  %cmp = fcmp oge ppc_fp128 %0, %1
  %conv = zext i1 %cmp to i32
  %conv1 = trunc i32 %conv to i8
  store i8 %conv1, i8* @var, align 1
  ret void

  ; CHECK-LABEL:    __gcc_qge
}

attributes #0 = { "use-soft-float"="true" }
