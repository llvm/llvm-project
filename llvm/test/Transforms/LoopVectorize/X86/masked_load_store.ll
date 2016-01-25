; RUN: opt < %s  -O3 -mcpu=corei7-avx -S | FileCheck %s -check-prefix=AVX -check-prefix=AVX1
; RUN: opt < %s  -O3 -mcpu=core-avx2 -S | FileCheck %s -check-prefix=AVX -check-prefix=AVX2
; RUN: opt < %s  -O3 -mcpu=knl -S | FileCheck %s -check-prefix=AVX512

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc_linux"

; The source code:
;
;void foo1(int *A, int *B, int *trigger) {
;
;  for (int i=0; i<10000; i++) {
;    if (trigger[i] < 100) {
;          A[i] = B[i] + trigger[i];
;    }
;  }
;}

;AVX-LABEL: @foo1
;AVX: icmp slt <8 x i32> %wide.load, <i32 100, i32 100, i32 100
;AVX: call <8 x i32> @llvm.masked.load.v8i32
;AVX: add nsw <8 x i32>
;AVX: call void @llvm.masked.store.v8i32
;AVX: ret void

;AVX512-LABEL: @foo1
;AVX512: icmp slt <16 x i32> %wide.load, <i32 100, i32 100, i32 100
;AVX512: call <16 x i32> @llvm.masked.load.v16i32
;AVX512: add nsw <16 x i32>
;AVX512: call void @llvm.masked.store.v16i32
;AVX512: ret void

; Function Attrs: nounwind uwtable
define void @foo1(i32* %A, i32* %B, i32* %trigger) {
entry:
  %A.addr = alloca i32*, align 8
  %B.addr = alloca i32*, align 8
  %trigger.addr = alloca i32*, align 8
  %i = alloca i32, align 4
  store i32* %A, i32** %A.addr, align 8
  store i32* %B, i32** %B.addr, align 8
  store i32* %trigger, i32** %trigger.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 10000
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %idxprom = sext i32 %1 to i64
  %2 = load i32*, i32** %trigger.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32* %2, i64 %idxprom
  %3 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp slt i32 %3, 100
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %4 = load i32, i32* %i, align 4
  %idxprom2 = sext i32 %4 to i64
  %5 = load i32*, i32** %B.addr, align 8
  %arrayidx3 = getelementptr inbounds i32, i32* %5, i64 %idxprom2
  %6 = load i32, i32* %arrayidx3, align 4
  %7 = load i32, i32* %i, align 4
  %idxprom4 = sext i32 %7 to i64
  %8 = load i32*, i32** %trigger.addr, align 8
  %arrayidx5 = getelementptr inbounds i32, i32* %8, i64 %idxprom4
  %9 = load i32, i32* %arrayidx5, align 4
  %add = add nsw i32 %6, %9
  %10 = load i32, i32* %i, align 4
  %idxprom6 = sext i32 %10 to i64
  %11 = load i32*, i32** %A.addr, align 8
  %arrayidx7 = getelementptr inbounds i32, i32* %11, i64 %idxprom6
  store i32 %add, i32* %arrayidx7, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %12 = load i32, i32* %i, align 4
  %inc = add nsw i32 %12, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; The source code:
;
;void foo2(float *A, float *B, int *trigger) {
;
;  for (int i=0; i<10000; i++) {
;    if (trigger[i] < 100) {
;          A[i] = B[i] + trigger[i];
;    }
;  }
;}

;AVX-LABEL: @foo2
;AVX: icmp slt <8 x i32> %wide.load, <i32 100, i32 100, i32 100
;AVX: call <8 x float> @llvm.masked.load.v8f32
;AVX: fadd <8 x float>
;AVX: call void @llvm.masked.store.v8f32
;AVX: ret void

;AVX512-LABEL: @foo2
;AVX512: icmp slt <16 x i32> %wide.load, <i32 100, i32 100, i32 100
;AVX512: call <16 x float> @llvm.masked.load.v16f32
;AVX512: fadd <16 x float>
;AVX512: call void @llvm.masked.store.v16f32
;AVX512: ret void

; Function Attrs: nounwind uwtable
define void @foo2(float* %A, float* %B, i32* %trigger) {
entry:
  %A.addr = alloca float*, align 8
  %B.addr = alloca float*, align 8
  %trigger.addr = alloca i32*, align 8
  %i = alloca i32, align 4
  store float* %A, float** %A.addr, align 8
  store float* %B, float** %B.addr, align 8
  store i32* %trigger, i32** %trigger.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 10000
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %idxprom = sext i32 %1 to i64
  %2 = load i32*, i32** %trigger.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32* %2, i64 %idxprom
  %3 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp slt i32 %3, 100
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %4 = load i32, i32* %i, align 4
  %idxprom2 = sext i32 %4 to i64
  %5 = load float*, float** %B.addr, align 8
  %arrayidx3 = getelementptr inbounds float, float* %5, i64 %idxprom2
  %6 = load float, float* %arrayidx3, align 4
  %7 = load i32, i32* %i, align 4
  %idxprom4 = sext i32 %7 to i64
  %8 = load i32*, i32** %trigger.addr, align 8
  %arrayidx5 = getelementptr inbounds i32, i32* %8, i64 %idxprom4
  %9 = load i32, i32* %arrayidx5, align 4
  %conv = sitofp i32 %9 to float
  %add = fadd float %6, %conv
  %10 = load i32, i32* %i, align 4
  %idxprom6 = sext i32 %10 to i64
  %11 = load float*, float** %A.addr, align 8
  %arrayidx7 = getelementptr inbounds float, float* %11, i64 %idxprom6
  store float %add, float* %arrayidx7, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %12 = load i32, i32* %i, align 4
  %inc = add nsw i32 %12, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; The source code:
;
;void foo3(double *A, double *B, int *trigger) {
;
;  for (int i=0; i<10000; i++) {
;    if (trigger[i] < 100) {
;          A[i] = B[i] + trigger[i];
;    }
;  }
;}

;AVX-LABEL: @foo3
;AVX: icmp slt <4 x i32> %wide.load, <i32 100, i32 100,
;AVX: call <4 x double> @llvm.masked.load.v4f64
;AVX: sitofp <4 x i32> %wide.load to <4 x double>
;AVX: fadd <4 x double>
;AVX: call void @llvm.masked.store.v4f64
;AVX: ret void

;AVX512-LABEL: @foo3
;AVX512: icmp slt <8 x i32> %wide.load, <i32 100, i32 100,
;AVX512: call <8 x double> @llvm.masked.load.v8f64
;AVX512: sitofp <8 x i32> %wide.load to <8 x double>
;AVX512: fadd <8 x double>
;AVX512: call void @llvm.masked.store.v8f64
;AVX512: ret void


; Function Attrs: nounwind uwtable
define void @foo3(double* %A, double* %B, i32* %trigger) #0 {
entry:
  %A.addr = alloca double*, align 8
  %B.addr = alloca double*, align 8
  %trigger.addr = alloca i32*, align 8
  %i = alloca i32, align 4
  store double* %A, double** %A.addr, align 8
  store double* %B, double** %B.addr, align 8
  store i32* %trigger, i32** %trigger.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 10000
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %idxprom = sext i32 %1 to i64
  %2 = load i32*, i32** %trigger.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32* %2, i64 %idxprom
  %3 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp slt i32 %3, 100
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %4 = load i32, i32* %i, align 4
  %idxprom2 = sext i32 %4 to i64
  %5 = load double*, double** %B.addr, align 8
  %arrayidx3 = getelementptr inbounds double, double* %5, i64 %idxprom2
  %6 = load double, double* %arrayidx3, align 8
  %7 = load i32, i32* %i, align 4
  %idxprom4 = sext i32 %7 to i64
  %8 = load i32*, i32** %trigger.addr, align 8
  %arrayidx5 = getelementptr inbounds i32, i32* %8, i64 %idxprom4
  %9 = load i32, i32* %arrayidx5, align 4
  %conv = sitofp i32 %9 to double
  %add = fadd double %6, %conv
  %10 = load i32, i32* %i, align 4
  %idxprom6 = sext i32 %10 to i64
  %11 = load double*, double** %A.addr, align 8
  %arrayidx7 = getelementptr inbounds double, double* %11, i64 %idxprom6
  store double %add, double* %arrayidx7, align 8
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %12 = load i32, i32* %i, align 4
  %inc = add nsw i32 %12, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; The source code:
;
;void foo4(double *A, double *B, int *trigger) {
;
;  for (int i=0; i<10000; i++) {
;    if (trigger[i] < 100) {
;          A[i] = B[i*2] + trigger[i]; << non-cosecutive access
;    }
;  }
;}

;AVX-LABEL: @foo4
;AVX-NOT: llvm.masked
;AVX: ret void

;AVX512-LABEL: @foo4
;AVX512-NOT: llvm.masked
;AVX512: ret void

; Function Attrs: nounwind uwtable
define void @foo4(double* %A, double* %B, i32* %trigger)  {
entry:
  %A.addr = alloca double*, align 8
  %B.addr = alloca double*, align 8
  %trigger.addr = alloca i32*, align 8
  %i = alloca i32, align 4
  store double* %A, double** %A.addr, align 8
  store double* %B, double** %B.addr, align 8
  store i32* %trigger, i32** %trigger.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 10000
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %idxprom = sext i32 %1 to i64
  %2 = load i32*, i32** %trigger.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32* %2, i64 %idxprom
  %3 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp slt i32 %3, 100
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %4 = load i32, i32* %i, align 4
  %mul = mul nsw i32 %4, 2
  %idxprom2 = sext i32 %mul to i64
  %5 = load double*, double** %B.addr, align 8
  %arrayidx3 = getelementptr inbounds double, double* %5, i64 %idxprom2
  %6 = load double, double* %arrayidx3, align 8
  %7 = load i32, i32* %i, align 4
  %idxprom4 = sext i32 %7 to i64
  %8 = load i32*, i32** %trigger.addr, align 8
  %arrayidx5 = getelementptr inbounds i32, i32* %8, i64 %idxprom4
  %9 = load i32, i32* %arrayidx5, align 4
  %conv = sitofp i32 %9 to double
  %add = fadd double %6, %conv
  %10 = load i32, i32* %i, align 4
  %idxprom6 = sext i32 %10 to i64
  %11 = load double*, double** %A.addr, align 8
  %arrayidx7 = getelementptr inbounds double, double* %11, i64 %idxprom6
  store double %add, double* %arrayidx7, align 8
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %12 = load i32, i32* %i, align 4
  %inc = add nsw i32 %12, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

@a = common global [1 x i32*] zeroinitializer, align 8
@c = common global i32* null, align 8

; The loop here should not be vectorized due to trapping
; constant expression
;AVX-LABEL: @foo5
;AVX-NOT: llvm.masked
;AVX: store i32 sdiv
;AVX: ret void

;AVX512-LABEL: @foo5
;AVX512-NOT: llvm.masked
;AVX512: store i32 sdiv
;AVX512: ret void

; Function Attrs: nounwind uwtable
define void @foo5(i32* %A, i32* %B, i32* %trigger) {
entry:
  %A.addr = alloca i32*, align 8
  %B.addr = alloca i32*, align 8
  %trigger.addr = alloca i32*, align 8
  %i = alloca i32, align 4
  store i32* %A, i32** %A.addr, align 8
  store i32* %B, i32** %B.addr, align 8
  store i32* %trigger, i32** %trigger.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 10000
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %idxprom = sext i32 %1 to i64
  %2 = load i32*, i32** %trigger.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32* %2, i64 %idxprom
  %3 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp slt i32 %3, 100
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %4 = load i32, i32* %i, align 4
  %idxprom2 = sext i32 %4 to i64
  %5 = load i32*, i32** %B.addr, align 8
  %arrayidx3 = getelementptr inbounds i32, i32* %5, i64 %idxprom2
  %6 = load i32, i32* %arrayidx3, align 4
  %7 = load i32, i32* %i, align 4
  %idxprom4 = sext i32 %7 to i64
  %8 = load i32*, i32** %trigger.addr, align 8
  %arrayidx5 = getelementptr inbounds i32, i32* %8, i64 %idxprom4
  %9 = load i32, i32* %arrayidx5, align 4
  %add = add nsw i32 %6, %9
  %10 = load i32, i32* %i, align 4
  %idxprom6 = sext i32 %10 to i64
  %11 = load i32*, i32** %A.addr, align 8
  %arrayidx7 = getelementptr inbounds i32, i32* %11, i64 %idxprom6
  store i32 sdiv (i32 1, i32 zext (i1 icmp eq (i32** getelementptr inbounds ([1 x i32*], [1 x i32*]* @a, i64 0, i64 1), i32** @c) to i32)), i32* %arrayidx7, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %12 = load i32, i32* %i, align 4
  %inc = add nsw i32 %12, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Reverse loop
;void foo6(double *in, double *out, unsigned size, int *trigger) {
;
;  for (int i=SIZE-1; i>=0; i--) {
;    if (trigger[i] > 0) {
;      out[i] = in[i] + (double) 0.5;
;    }
;  }
;}
;AVX2-LABEL: @foo6
;AVX2: icmp sgt <4 x i32> %reverse, zeroinitializer
;AVX2: shufflevector <4 x i1>{{.*}}<4 x i32> <i32 3, i32 2, i32 1, i32 0>
;AVX2: call <4 x double> @llvm.masked.load.v4f64
;AVX2: fadd <4 x double>
;AVX2: call void @llvm.masked.store.v4f64
;AVX2: ret void

;AVX512-LABEL: @foo6
;AVX512: icmp sgt <8 x i32> %reverse, zeroinitializer
;AVX512: shufflevector <8 x i1>{{.*}}<8 x i32> <i32 7, i32 6, i32 5, i32 4
;AVX512: call <8 x double> @llvm.masked.load.v8f64
;AVX512: fadd <8 x double>
;AVX512: call void @llvm.masked.store.v8f64
;AVX512: ret void


define void @foo6(double* %in, double* %out, i32 %size, i32* %trigger) {
entry:
  %in.addr = alloca double*, align 8
  %out.addr = alloca double*, align 8
  %size.addr = alloca i32, align 4
  %trigger.addr = alloca i32*, align 8
  %i = alloca i32, align 4
  store double* %in, double** %in.addr, align 8
  store double* %out, double** %out.addr, align 8
  store i32 %size, i32* %size.addr, align 4
  store i32* %trigger, i32** %trigger.addr, align 8
  store i32 4095, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp sge i32 %0, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %idxprom = sext i32 %1 to i64
  %2 = load i32*, i32** %trigger.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32* %2, i64 %idxprom
  %3 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %3, 0
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %4 = load i32, i32* %i, align 4
  %idxprom2 = sext i32 %4 to i64
  %5 = load double*, double** %in.addr, align 8
  %arrayidx3 = getelementptr inbounds double, double* %5, i64 %idxprom2
  %6 = load double, double* %arrayidx3, align 8
  %add = fadd double %6, 5.000000e-01
  %7 = load i32, i32* %i, align 4
  %idxprom4 = sext i32 %7 to i64
  %8 = load double*, double** %out.addr, align 8
  %arrayidx5 = getelementptr inbounds double, double* %8, i64 %idxprom4
  store double %add, double* %arrayidx5, align 8
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %9 = load i32, i32* %i, align 4
  %dec = add nsw i32 %9, -1
  store i32 %dec, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; void foo7 (double * __restrict__  out, double ** __restrict__  in,
;           bool * __restrict__ trigger, unsigned size) {
;
;  for (unsigned i=0; i<size; i++)
;    if (trigger[i] && (in[i] != 0))
;      out[i] = (double) 0.5;
; }

;AVX512-LABEL: @foo7
;AVX512: call <8 x double*> @llvm.masked.load.v8p0f64(<8 x double*>*
;AVX512: call void @llvm.masked.store.v8f64
;AVX512: ret void

define void @foo7(double* noalias %out, double** noalias %in, i8* noalias %trigger, i32 %size) #0 {
entry:
  %out.addr = alloca double*, align 8
  %in.addr = alloca double**, align 8
  %trigger.addr = alloca i8*, align 8
  %size.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store double* %out, double** %out.addr, align 8
  store double** %in, double*** %in.addr, align 8
  store i8* %trigger, i8** %trigger.addr, align 8
  store i32 %size, i32* %size.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %1 = load i32, i32* %size.addr, align 4
  %cmp = icmp ult i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i32, i32* %i, align 4
  %idxprom = zext i32 %2 to i64
  %3 = load i8*, i8** %trigger.addr, align 8
  %arrayidx = getelementptr inbounds i8, i8* %3, i64 %idxprom
  %4 = load i8, i8* %arrayidx, align 1
  %tobool = trunc i8 %4 to i1
  br i1 %tobool, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %idxprom1 = zext i32 %5 to i64
  %6 = load double**, double*** %in.addr, align 8
  %arrayidx2 = getelementptr inbounds double*, double** %6, i64 %idxprom1
  %7 = load double*, double** %arrayidx2, align 8
  %cmp3 = icmp ne double* %7, null
  br i1 %cmp3, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %8 = load i32, i32* %i, align 4
  %idxprom4 = zext i32 %8 to i64
  %9 = load double*, double** %out.addr, align 8
  %arrayidx5 = getelementptr inbounds double, double* %9, i64 %idxprom4
  store double 5.000000e-01, double* %arrayidx5, align 8
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %10 = load i32, i32* %i, align 4
  %inc = add i32 %10, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

;typedef int (*fp)();
;void foo8 (double* __restrict__  out, fp* __restrict__ in, bool * __restrict__ trigger, unsigned size) {
;
;  for (unsigned i=0; i<size; i++)
;    if (trigger[i] && (in[i] != 0))
;      out[i] = (double) 0.5;
;}

;AVX512-LABEL: @foo8
;AVX512: call <8 x i32 ()*> @llvm.masked.load.v8p0f_i32f(<8 x i32 ()*>* %
;AVX512: call void @llvm.masked.store.v8f64
;AVX512: ret void

define void @foo8(double* noalias %out, i32 ()** noalias %in, i8* noalias %trigger, i32 %size) #0 {
entry:
  %out.addr = alloca double*, align 8
  %in.addr = alloca i32 ()**, align 8
  %trigger.addr = alloca i8*, align 8
  %size.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store double* %out, double** %out.addr, align 8
  store i32 ()** %in, i32 ()*** %in.addr, align 8
  store i8* %trigger, i8** %trigger.addr, align 8
  store i32 %size, i32* %size.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %1 = load i32, i32* %size.addr, align 4
  %cmp = icmp ult i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i32, i32* %i, align 4
  %idxprom = zext i32 %2 to i64
  %3 = load i8*, i8** %trigger.addr, align 8
  %arrayidx = getelementptr inbounds i8, i8* %3, i64 %idxprom
  %4 = load i8, i8* %arrayidx, align 1
  %tobool = trunc i8 %4 to i1
  br i1 %tobool, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %idxprom1 = zext i32 %5 to i64
  %6 = load i32 ()**, i32 ()*** %in.addr, align 8
  %arrayidx2 = getelementptr inbounds i32 ()*, i32 ()** %6, i64 %idxprom1
  %7 = load i32 ()*, i32 ()** %arrayidx2, align 8
  %cmp3 = icmp ne i32 ()* %7, null
  br i1 %cmp3, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %8 = load i32, i32* %i, align 4
  %idxprom4 = zext i32 %8 to i64
  %9 = load double*, double** %out.addr, align 8
  %arrayidx5 = getelementptr inbounds double, double* %9, i64 %idxprom4
  store double 5.000000e-01, double* %arrayidx5, align 8
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %10 = load i32, i32* %i, align 4
  %inc = add i32 %10, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
