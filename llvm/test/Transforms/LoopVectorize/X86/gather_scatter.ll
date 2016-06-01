; RUN: opt < %s  -O3 -mcpu=knl -S | FileCheck %s -check-prefix=AVX512

;AVX1-NOT: llvm.masked

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc_linux"

; The source code:
;
;void foo1(float * __restrict__ in, float * __restrict__ out, int * __restrict__ trigger, int * __restrict__ index) {
;
;  for (int i=0; i < SIZE; ++i) {
;    if (trigger[i] > 0) {
;      out[i] = in[index[i]] + (float) 0.5;
;    }
;  }
;}

;AVX512-LABEL: @foo1
;AVX512:  llvm.masked.load.v8i32
;AVX512: llvm.masked.gather.v8f32
;AVX512: llvm.masked.store.v8f32
;AVX512: ret void

; Function Attrs: nounwind uwtable
define void @foo1(float* noalias %in, float* noalias %out, i32* noalias %trigger, i32* noalias %index) {
entry:
  %in.addr = alloca float*, align 8
  %out.addr = alloca float*, align 8
  %trigger.addr = alloca i32*, align 8
  %index.addr = alloca i32*, align 8
  %i = alloca i32, align 4
  store float* %in, float** %in.addr, align 8
  store float* %out, float** %out.addr, align 8
  store i32* %trigger, i32** %trigger.addr, align 8
  store i32* %index, i32** %index.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 4096
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
  %5 = load i32*, i32** %index.addr, align 8
  %arrayidx3 = getelementptr inbounds i32, i32* %5, i64 %idxprom2
  %6 = load i32, i32* %arrayidx3, align 4
  %idxprom4 = sext i32 %6 to i64
  %7 = load float*, float** %in.addr, align 8
  %arrayidx5 = getelementptr inbounds float, float* %7, i64 %idxprom4
  %8 = load float, float* %arrayidx5, align 4
  %add = fadd float %8, 5.000000e-01
  %9 = load i32, i32* %i, align 4
  %idxprom6 = sext i32 %9 to i64
  %10 = load float*, float** %out.addr, align 8
  %arrayidx7 = getelementptr inbounds float, float* %10, i64 %idxprom6
  store float %add, float* %arrayidx7, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %11 = load i32, i32* %i, align 4
  %inc = add nsw i32 %11, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; The source code
;void foo2 (In * __restrict__ in, float * __restrict__ out, int * __restrict__ trigger) {
;
;  for (int i=0; i<SIZE; ++i) {
;    if (trigger[i] > 0) {
;      out[i] = in[i].b + (float) 0.5;
;    }
;  }
;}

%struct.In = type { float, float }

;AVX512-LABEL: @foo2
;AVX512: getelementptr %struct.In, %struct.In* %in, <16 x i64> %{{.*}}, i32 1
;AVX512: llvm.masked.gather.v16f32
;AVX512: llvm.masked.store.v16f32
;AVX512: ret void
define void @foo2(%struct.In* noalias %in, float* noalias %out, i32* noalias %trigger, i32* noalias %index) #0 {
entry:
  %in.addr = alloca %struct.In*, align 8
  %out.addr = alloca float*, align 8
  %trigger.addr = alloca i32*, align 8
  %index.addr = alloca i32*, align 8
  %i = alloca i32, align 4
  store %struct.In* %in, %struct.In** %in.addr, align 8
  store float* %out, float** %out.addr, align 8
  store i32* %trigger, i32** %trigger.addr, align 8
  store i32* %index, i32** %index.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 4096
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
  %5 = load %struct.In*, %struct.In** %in.addr, align 8
  %arrayidx3 = getelementptr inbounds %struct.In, %struct.In* %5, i64 %idxprom2
  %b = getelementptr inbounds %struct.In, %struct.In* %arrayidx3, i32 0, i32 1
  %6 = load float, float* %b, align 4
  %add = fadd float %6, 5.000000e-01
  %7 = load i32, i32* %i, align 4
  %idxprom4 = sext i32 %7 to i64
  %8 = load float*, float** %out.addr, align 8
  %arrayidx5 = getelementptr inbounds float, float* %8, i64 %idxprom4
  store float %add, float* %arrayidx5, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %9 = load i32, i32* %i, align 4
  %inc = add nsw i32 %9, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; The source code
;struct Out {
;  float a;
;  float b;
;};
;void foo3 (In * __restrict__ in, Out * __restrict__ out, int * __restrict__ trigger) {
;
;  for (int i=0; i<SIZE; ++i) {
;    if (trigger[i] > 0) {
;      out[i].b = in[i].b + (float) 0.5;
;    }
;  }
;}

;AVX512-LABEL: @foo3
;AVX512: getelementptr %struct.In, %struct.In* %in, <16 x i64> %{{.*}}, i32 1
;AVX512: llvm.masked.gather.v16f32
;AVX512: fadd <16 x float>
;AVX512: getelementptr %struct.Out, %struct.Out* %out, <16 x i64> %{{.*}}, i32 1
;AVX512: llvm.masked.scatter.v16f32
;AVX512: ret void

%struct.Out = type { float, float }

define void @foo3(%struct.In* noalias %in, %struct.Out* noalias %out, i32* noalias %trigger) {
entry:
  %in.addr = alloca %struct.In*, align 8
  %out.addr = alloca %struct.Out*, align 8
  %trigger.addr = alloca i32*, align 8
  %i = alloca i32, align 4
  store %struct.In* %in, %struct.In** %in.addr, align 8
  store %struct.Out* %out, %struct.Out** %out.addr, align 8
  store i32* %trigger, i32** %trigger.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 4096
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
  %5 = load %struct.In*, %struct.In** %in.addr, align 8
  %arrayidx3 = getelementptr inbounds %struct.In, %struct.In* %5, i64 %idxprom2
  %b = getelementptr inbounds %struct.In, %struct.In* %arrayidx3, i32 0, i32 1
  %6 = load float, float* %b, align 4
  %add = fadd float %6, 5.000000e-01
  %7 = load i32, i32* %i, align 4
  %idxprom4 = sext i32 %7 to i64
  %8 = load %struct.Out*, %struct.Out** %out.addr, align 8
  %arrayidx5 = getelementptr inbounds %struct.Out, %struct.Out* %8, i64 %idxprom4
  %b6 = getelementptr inbounds %struct.Out, %struct.Out* %arrayidx5, i32 0, i32 1
  store float %add, float* %b6, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %9 = load i32, i32* %i, align 4
  %inc = add nsw i32 %9, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
declare void @llvm.masked.scatter.v16f32(<16 x float>, <16 x float*>, i32, <16 x i1>)
