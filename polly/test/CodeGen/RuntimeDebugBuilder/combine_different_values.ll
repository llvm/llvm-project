; RUN: opt %loadNPMPolly -passes=polly-codegen -S \
; RUN: -polly-codegen-add-debug-printing \
; RUN: -polly-ignore-aliasing < %s | FileCheck %s

;    #define N 10
;    void foo(float A[restrict], double B[restrict], char C[restrict],
;             int D[restrict], long E[restrict]) {
;      for (long i = 0; i < N; i++)
;        A[i] += B[i] + C[i] + D[i] + E[i];
;    }
;
;    int main() {
;      float A[N];
;      double B[N];
;      char C[N];
;      int D[N];
;      long E[N];
;
;      for (long i = 0; i < N; i++) {
;        __sync_synchronize();
;        A[i] = B[i] = C[i] = D[i] = E[i] = 42;
;      }
;
;      foo(A, B, C, D, E);
;
;      return A[8];
;    }

; CHECK: @0 = private unnamed_addr addrspace(4) constant [11 x i8] c"Load from \00"
; CHECK: @1 = private unnamed_addr addrspace(4) constant [3 x i8] c": \00"
; CHECK: @2 = private unnamed_addr addrspace(4) constant [2 x i8] c"\0A\00"
; CHECK: @3 = private unnamed_addr constant [12 x i8] c"%s%ld%s%f%s\00"
; CHECK: @4 = private unnamed_addr addrspace(4) constant [11 x i8] c"Load from \00"
; CHECK: @5 = private unnamed_addr addrspace(4) constant [3 x i8] c": \00"
; CHECK: @6 = private unnamed_addr addrspace(4) constant [2 x i8] c"\0A\00"
; CHECK: @7 = private unnamed_addr constant [13 x i8] c"%s%ld%s%ld%s\00"
; CHECK: @8 = private unnamed_addr addrspace(4) constant [11 x i8] c"Load from \00"
; CHECK: @9 = private unnamed_addr addrspace(4) constant [3 x i8] c": \00"
; CHECK: @10 = private unnamed_addr addrspace(4) constant [2 x i8] c"\0A\00"
; CHECK: @11 = private unnamed_addr constant [13 x i8] c"%s%ld%s%ld%s\00"
; CHECK: @12 = private unnamed_addr addrspace(4) constant [11 x i8] c"Load from \00"
; CHECK: @13 = private unnamed_addr addrspace(4) constant [3 x i8] c": \00"
; CHECK: @14 = private unnamed_addr addrspace(4) constant [2 x i8] c"\0A\00"
; CHECK: @15 = private unnamed_addr constant [13 x i8] c"%s%ld%s%ld%s\00"
; CHECK: @16 = private unnamed_addr addrspace(4) constant [11 x i8] c"Load from \00"
; CHECK: @17 = private unnamed_addr addrspace(4) constant [3 x i8] c": \00"
; CHECK: @18 = private unnamed_addr addrspace(4) constant [2 x i8] c"\0A\00"
; CHECK: @19 = private unnamed_addr constant [12 x i8] c"%s%ld%s%f%s\00"
; CHECK: @20 = private unnamed_addr addrspace(4) constant [11 x i8] c"Store to  \00"
; CHECK: @21 = private unnamed_addr addrspace(4) constant [3 x i8] c": \00"
; CHECK: @22 = private unnamed_addr addrspace(4) constant [2 x i8] c"\0A\00"
; CHECK: @23 = private unnamed_addr constant [12 x i8] c"%s%ld%s%f%s\00"

; CHECK: %0 = shl nuw nsw i64 %polly.indvar, 3
; CHECK: %scevgep = getelementptr i8, ptr %B, i64 %0
; CHECK: %tmp3_p_scalar_ = load double, ptr %scevgep, align 8, !alias.scope !0, !noalias !3
; CHECK: %1 = ptrtoint ptr %scevgep to i64
; CHECK: %2 = call i32 (...) @printf(ptr @3, ptr addrspace(4) @0, i64 %1, ptr addrspace(4) @1, double %tmp3_p_scalar_, ptr addrspace(4) @2)
; CHECK: %3 = call i32 @fflush(ptr null)
; CHECK: %scevgep1 = getelementptr i8, ptr %C, i64 %polly.indvar
; CHECK: %tmp5_p_scalar_ = load i8, ptr %scevgep1, align 1, !alias.scope !8, !noalias !9
; CHECK: %4 = ptrtoint ptr %scevgep1 to i64
; CHECK: %5 = sext i8 %tmp5_p_scalar_ to i64
; CHECK: %6 = call i32 (...) @printf(ptr @7, ptr addrspace(4) @4, i64 %4, ptr addrspace(4) @5, i64 %5, ptr addrspace(4) @6)
; CHECK: %7 = call i32 @fflush(ptr null)
; CHECK: %p_tmp6 = sitofp i8 %tmp5_p_scalar_ to double
; CHECK: %p_tmp7 = fadd double %tmp3_p_scalar_, %p_tmp6
; CHECK: %8 = shl nuw nsw i64 %polly.indvar, 2
; CHECK: %scevgep2 = getelementptr i8, ptr %D, i64 %8
; CHECK: %tmp9_p_scalar_ = load i32, ptr %scevgep2, align 4, !alias.scope !10, !noalias !11
; CHECK: %9 = ptrtoint ptr %scevgep2 to i64
; CHECK: %10 = sext i32 %tmp9_p_scalar_ to i64
; CHECK: %11 = call i32 (...) @printf(ptr @11, ptr addrspace(4) @8, i64 %9, ptr addrspace(4) @9, i64 %10, ptr addrspace(4) @10)
; CHECK: %12 = call i32 @fflush(ptr null)
; CHECK: %p_tmp10 = sitofp i32 %tmp9_p_scalar_ to double
; CHECK: %p_tmp11 = fadd double %p_tmp7, %p_tmp10
; CHECK: %13 = shl nuw nsw i64 %polly.indvar, 3
; CHECK: %scevgep3 = getelementptr i8, ptr %E, i64 %13
; CHECK: %tmp13_p_scalar_ = load i64, ptr %scevgep3, align 8, !alias.scope !12, !noalias !13
; CHECK: %14 = ptrtoint ptr %scevgep3 to i64
; CHECK: %15 = call i32 (...) @printf(ptr @15, ptr addrspace(4) @12, i64 %14, ptr addrspace(4) @13, i64 %tmp13_p_scalar_, ptr addrspace(4) @14)
; CHECK: %16 = call i32 @fflush(ptr null)
; CHECK: %p_tmp14 = sitofp i64 %tmp13_p_scalar_ to double
; CHECK: %p_tmp15 = fadd double %p_tmp11, %p_tmp14
; CHECK: %17 = shl nuw nsw i64 %polly.indvar, 2
; CHECK: %scevgep4 = getelementptr i8, ptr %A, i64 %17
; CHECK: %tmp17_p_scalar_ = load float, ptr %scevgep4, align 4, !alias.scope !14, !noalias !15
; CHECK: %18 = ptrtoint ptr %scevgep4 to i64
; CHECK: %19 = fpext float %tmp17_p_scalar_ to double
; CHECK: %20 = call i32 (...) @printf(ptr @19, ptr addrspace(4) @16, i64 %18, ptr addrspace(4) @17, double %19, ptr addrspace(4) @18)
; CHECK: %21 = call i32 @fflush(ptr null)
; CHECK: %p_tmp18 = fpext float %tmp17_p_scalar_ to double
; CHECK: %p_tmp19 = fadd double %p_tmp18, %p_tmp15
; CHECK: %p_tmp20 = fptrunc double %p_tmp19 to float
; CHECK: %22 = ptrtoint ptr %scevgep4 to i64
; CHECK: %23 = fpext float %p_tmp20 to double
; CHECK: %24 = call i32 (...) @printf(ptr @23, ptr addrspace(4) @20, i64 %22, ptr addrspace(4) @21, double %23, ptr addrspace(4) @22)
; CHECK: %25 = call i32 @fflush(ptr null)

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(ptr noalias %A, ptr noalias %B, ptr noalias %C, ptr noalias %D, ptr noalias %E) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb21, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp22, %bb21 ]
  %exitcond = icmp ne i64 %i.0, 10
  br i1 %exitcond, label %bb2, label %bb23

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds double, ptr %B, i64 %i.0
  %tmp3 = load double, ptr %tmp, align 8
  %tmp4 = getelementptr inbounds i8, ptr %C, i64 %i.0
  %tmp5 = load i8, ptr %tmp4, align 1
  %tmp6 = sitofp i8 %tmp5 to double
  %tmp7 = fadd double %tmp3, %tmp6
  %tmp8 = getelementptr inbounds i32, ptr %D, i64 %i.0
  %tmp9 = load i32, ptr %tmp8, align 4
  %tmp10 = sitofp i32 %tmp9 to double
  %tmp11 = fadd double %tmp7, %tmp10
  %tmp12 = getelementptr inbounds i64, ptr %E, i64 %i.0
  %tmp13 = load i64, ptr %tmp12, align 8
  %tmp14 = sitofp i64 %tmp13 to double
  %tmp15 = fadd double %tmp11, %tmp14
  %tmp16 = getelementptr inbounds float, ptr %A, i64 %i.0
  %tmp17 = load float, ptr %tmp16, align 4
  %tmp18 = fpext float %tmp17 to double
  %tmp19 = fadd double %tmp18, %tmp15
  %tmp20 = fptrunc double %tmp19 to float
  store float %tmp20, ptr %tmp16, align 4
  br label %bb21

bb21:                                             ; preds = %bb2
  %tmp22 = add nsw i64 %i.0, 1
  br label %bb1

bb23:                                             ; preds = %bb1
  ret void
}

define i32 @main() {
bb:
  %A = alloca [10 x float], align 16
  %B = alloca [10 x double], align 16
  %C = alloca [10 x i8], align 1
  %D = alloca [10 x i32], align 16
  %E = alloca [10 x i64], align 16
  br label %bb1

bb1:                                              ; preds = %bb7, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp8, %bb7 ]
  %exitcond = icmp ne i64 %i.0, 10
  br i1 %exitcond, label %bb2, label %bb9

bb2:                                              ; preds = %bb1
  fence seq_cst
  %tmp = getelementptr inbounds [10 x i64], ptr %E, i64 0, i64 %i.0
  store i64 42, ptr %tmp, align 8
  %tmp3 = getelementptr inbounds [10 x i32], ptr %D, i64 0, i64 %i.0
  store i32 42, ptr %tmp3, align 4
  %tmp4 = getelementptr inbounds [10 x i8], ptr %C, i64 0, i64 %i.0
  store i8 42, ptr %tmp4, align 1
  %tmp5 = getelementptr inbounds [10 x double], ptr %B, i64 0, i64 %i.0
  store double 4.200000e+01, ptr %tmp5, align 8
  %tmp6 = getelementptr inbounds [10 x float], ptr %A, i64 0, i64 %i.0
  store float 4.200000e+01, ptr %tmp6, align 4
  br label %bb7

bb7:                                              ; preds = %bb2
  %tmp8 = add nsw i64 %i.0, 1
  br label %bb1

bb9:                                              ; preds = %bb1
  %tmp10 = getelementptr inbounds [10 x float], ptr %A, i64 0, i64 0
  %tmp11 = getelementptr inbounds [10 x double], ptr %B, i64 0, i64 0
  %tmp12 = getelementptr inbounds [10 x i8], ptr %C, i64 0, i64 0
  %tmp13 = getelementptr inbounds [10 x i32], ptr %D, i64 0, i64 0
  %tmp14 = getelementptr inbounds [10 x i64], ptr %E, i64 0, i64 0
  call void @foo(ptr %tmp10, ptr %tmp11, ptr %tmp12, ptr %tmp13, ptr %tmp14)
  %tmp15 = getelementptr inbounds [10 x float], ptr %A, i64 0, i64 8
  %tmp16 = load float, ptr %tmp15, align 16
  %tmp17 = fptosi float %tmp16 to i32
  ret i32 %tmp17
}
