; RUN: opt -opaque-pointers=0 %loadPolly -polly-codegen -S \
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

; CHECK: %0 = ptrtoint double* %scevgep to i64
; CHECK: %1 = call i32 (...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @3, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* @0, i32 0, i32 0), i64 %0, i8 addrspace(4)* getelementptr inbounds ([3 x i8], [3 x i8] addrspace(4)* @1, i32 0, i32 0), double %tmp3_p_scalar_, i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* @2, i32 0, i32 0))
; CHECK: %2 = call i32 @fflush(i8* null)
; CHECK: %scevgep1 = getelementptr i8, i8* %C, i64 %polly.indvar
; CHECK: %tmp5_p_scalar_ = load i8, i8* %scevgep1
; CHECK: %3 = ptrtoint i8* %scevgep1 to i64
; CHECK: %4 = sext i8 %tmp5_p_scalar_ to i64
; CHECK: %5 = call i32 (...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @7, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* @4, i32 0, i32 0), i64 %3, i8 addrspace(4)* getelementptr inbounds ([3 x i8], [3 x i8] addrspace(4)* @5, i32 0, i32 0), i64 %4, i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* @6, i32 0, i32 0))
; CHECK: %6 = call i32 @fflush(i8* null)
; CHECK: %p_tmp6 = sitofp i8 %tmp5_p_scalar_ to double
; CHECK: %p_tmp7 = fadd double %tmp3_p_scalar_, %p_tmp6
; CHECK: %scevgep2 = getelementptr i32, i32* %D, i64 %polly.indvar
; CHECK: %tmp9_p_scalar_ = load i32, i32* %scevgep2
; CHECK: %7 = ptrtoint i32* %scevgep2 to i64
; CHECK: %8 = sext i32 %tmp9_p_scalar_ to i64
; CHECK: %9 = call i32 (...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @11, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* @8, i32 0, i32 0), i64 %7, i8 addrspace(4)* getelementptr inbounds ([3 x i8], [3 x i8] addrspace(4)* @9, i32 0, i32 0), i64 %8, i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* @10, i32 0, i32 0))
; CHECK: %10 = call i32 @fflush(i8* null)
; CHECK: %p_tmp10 = sitofp i32 %tmp9_p_scalar_ to double
; CHECK: %p_tmp11 = fadd double %p_tmp7, %p_tmp10
; CHECK: %scevgep3 = getelementptr i64, i64* %E, i64 %polly.indvar
; CHECK: %tmp13_p_scalar_ = load i64, i64* %scevgep3
; CHECK: %11 = ptrtoint i64* %scevgep3 to i64
; CHECK: %12 = call i32 (...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @15, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* @12, i32 0, i32 0), i64 %11, i8 addrspace(4)* getelementptr inbounds ([3 x i8], [3 x i8] addrspace(4)* @13, i32 0, i32 0), i64 %tmp13_p_scalar_, i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* @14, i32 0, i32 0))
; CHECK: %13 = call i32 @fflush(i8* null)
; CHECK: %p_tmp14 = sitofp i64 %tmp13_p_scalar_ to double
; CHECK: %p_tmp15 = fadd double %p_tmp11, %p_tmp14
; CHECK: %scevgep4 = getelementptr float, float* %A, i64 %polly.indvar
; CHECK: %tmp17_p_scalar_ = load float, float* %scevgep4
; CHECK: %14 = ptrtoint float* %scevgep4 to i64
; CHECK: %15 = fpext float %tmp17_p_scalar_ to double
; CHECK: %16 = call i32 (...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @19, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* @16, i32 0, i32 0), i64 %14, i8 addrspace(4)* getelementptr inbounds ([3 x i8], [3 x i8] addrspace(4)* @17, i32 0, i32 0), double %15, i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* @18, i32 0, i32 0))
; CHECK: %17 = call i32 @fflush(i8* null)
; CHECK: %p_tmp18 = fpext float %tmp17_p_scalar_ to double
; CHECK: %p_tmp19 = fadd double %p_tmp18, %p_tmp15
; CHECK: %p_tmp20 = fptrunc double %p_tmp19 to float
; CHECK: %18 = ptrtoint float* %scevgep4 to i64
; CHECK: %19 = fpext float %p_tmp20 to double
; CHECK: %20 = call i32 (...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @23, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(4)* @20, i32 0, i32 0), i64 %18, i8 addrspace(4)* getelementptr inbounds ([3 x i8], [3 x i8] addrspace(4)* @21, i32 0, i32 0), double %19, i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* @22, i32 0, i32 0))
; CHECK: %21 = call i32 @fflush(i8* null)

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(float* noalias %A, double* noalias %B, i8* noalias %C, i32* noalias %D, i64* noalias %E) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb21, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp22, %bb21 ]
  %exitcond = icmp ne i64 %i.0, 10
  br i1 %exitcond, label %bb2, label %bb23

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds double, double* %B, i64 %i.0
  %tmp3 = load double, double* %tmp, align 8
  %tmp4 = getelementptr inbounds i8, i8* %C, i64 %i.0
  %tmp5 = load i8, i8* %tmp4, align 1
  %tmp6 = sitofp i8 %tmp5 to double
  %tmp7 = fadd double %tmp3, %tmp6
  %tmp8 = getelementptr inbounds i32, i32* %D, i64 %i.0
  %tmp9 = load i32, i32* %tmp8, align 4
  %tmp10 = sitofp i32 %tmp9 to double
  %tmp11 = fadd double %tmp7, %tmp10
  %tmp12 = getelementptr inbounds i64, i64* %E, i64 %i.0
  %tmp13 = load i64, i64* %tmp12, align 8
  %tmp14 = sitofp i64 %tmp13 to double
  %tmp15 = fadd double %tmp11, %tmp14
  %tmp16 = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp17 = load float, float* %tmp16, align 4
  %tmp18 = fpext float %tmp17 to double
  %tmp19 = fadd double %tmp18, %tmp15
  %tmp20 = fptrunc double %tmp19 to float
  store float %tmp20, float* %tmp16, align 4
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
  %tmp = getelementptr inbounds [10 x i64], [10 x i64]* %E, i64 0, i64 %i.0
  store i64 42, i64* %tmp, align 8
  %tmp3 = getelementptr inbounds [10 x i32], [10 x i32]* %D, i64 0, i64 %i.0
  store i32 42, i32* %tmp3, align 4
  %tmp4 = getelementptr inbounds [10 x i8], [10 x i8]* %C, i64 0, i64 %i.0
  store i8 42, i8* %tmp4, align 1
  %tmp5 = getelementptr inbounds [10 x double], [10 x double]* %B, i64 0, i64 %i.0
  store double 4.200000e+01, double* %tmp5, align 8
  %tmp6 = getelementptr inbounds [10 x float], [10 x float]* %A, i64 0, i64 %i.0
  store float 4.200000e+01, float* %tmp6, align 4
  br label %bb7

bb7:                                              ; preds = %bb2
  %tmp8 = add nsw i64 %i.0, 1
  br label %bb1

bb9:                                              ; preds = %bb1
  %tmp10 = getelementptr inbounds [10 x float], [10 x float]* %A, i64 0, i64 0
  %tmp11 = getelementptr inbounds [10 x double], [10 x double]* %B, i64 0, i64 0
  %tmp12 = getelementptr inbounds [10 x i8], [10 x i8]* %C, i64 0, i64 0
  %tmp13 = getelementptr inbounds [10 x i32], [10 x i32]* %D, i64 0, i64 0
  %tmp14 = getelementptr inbounds [10 x i64], [10 x i64]* %E, i64 0, i64 0
  call void @foo(float* %tmp10, double* %tmp11, i8* %tmp12, i32* %tmp13, i64* %tmp14)
  %tmp15 = getelementptr inbounds [10 x float], [10 x float]* %A, i64 0, i64 8
  %tmp16 = load float, float* %tmp15, align 16
  %tmp17 = fptosi float %tmp16 to i32
  ret i32 %tmp17
}
