; RUN: opt --mtriple=amdgcn-amd-amdhsa -S -passes='attributor' < %s | FileCheck %s

; verify that the following test case does not assert in the attributor due
; to addrspace 5 to generic casts seen when compiling for amdgcn-amd-amdhsa
;
; clang++ -O2 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 red-DblComplex.cpp
;
; #include <complex>
; std::complex<double> reduce(std::complex<double> dres[], int n) {
;     std::complex<double> dinp(0.0, 0.0);
;     #pragma omp target teams distribute parallel for map(to: dres) map(tofrom:dinp) reduction(+:dinp)
;     for (int i = 0; i < n; i++) {
;         dinp += dres[i];
;     }
;     return(dinp);
; }

; CHECK: define void @_omp_reduction_shuffle_and_reduce_func.3

; ModuleID = 'red-DblComplex.cpp'
source_filename = "red-DblComplex.cpp"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

%"struct.std::complex" = type { { double, double } }

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)

declare i64 @__kmpc_shuffle_int64(i64, i16, i16)
declare void @_omp$reduction$reduction_func.2(i8* %0, i8* %1)

; Function Attrs: convergent norecurse nounwind
define void @_omp_reduction_shuffle_and_reduce_func.3(i8* %0, i16 signext %1, i16 signext %2, i16 signext %3) {
entry:
  %.omp.reduction.remote_reduce_list = alloca [1 x i8*], align 8, addrspace(5)
  %.omp.reduction.remote_reduce_list.ascast = addrspacecast [1 x i8*] addrspace(5)* %.omp.reduction.remote_reduce_list to [1 x i8*]*
  %.omp.reduction.element = alloca %"struct.std::complex", align 8, addrspace(5)
  %.omp.reduction.element.ascast = addrspacecast %"struct.std::complex" addrspace(5)* %.omp.reduction.element to %"struct.std::complex"*
  %4 = bitcast i8* %0 to [1 x i8*]*
  %5 = getelementptr inbounds [1 x i8*], [1 x i8*]* %4, i64 0, i64 0
  %6 = load i8*, i8** %5, align 8
  %7 = getelementptr inbounds [1 x i8*], [1 x i8*]* %.omp.reduction.remote_reduce_list.ascast, i64 0, i64 0
  %8 = bitcast i8* %6 to %"struct.std::complex"*
  %9 = getelementptr %"struct.std::complex", %"struct.std::complex"* %8, i64 1
  %10 = bitcast %"struct.std::complex"* %9 to i8*
  %11 = bitcast %"struct.std::complex"* %8 to i64*
  %12 = bitcast %"struct.std::complex"* %.omp.reduction.element.ascast to i64*
  br label %.shuffle.pre_cond

.shuffle.pre_cond:                                ; preds = %.shuffle.then, %entry
  %13 = phi i64* [ %11, %entry ], [ %23, %.shuffle.then ]
  %14 = phi i64* [ %12, %entry ], [ %24, %.shuffle.then ]
  %15 = bitcast i64* %13 to i8*
  %16 = ptrtoint i8* %10 to i64
  %17 = ptrtoint i8* %15 to i64
  %18 = sub i64 %16, %17
  %19 = sdiv exact i64 %18, ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
  %20 = icmp sgt i64 %19, 7
  br i1 %20, label %.shuffle.then, label %.shuffle.exit

.shuffle.then:                                    ; preds = %.shuffle.pre_cond
  %21 = load i64, i64* %13, align 8
  %22 = call i64 @__kmpc_shuffle_int64(i64 %21, i16 %2, i16 64)
  store i64 %22, i64* %14, align 8
  %23 = getelementptr i64, i64* %13, i64 1
  %24 = getelementptr i64, i64* %14, i64 1
  br label %.shuffle.pre_cond

.shuffle.exit:                                    ; preds = %.shuffle.pre_cond
  %25 = bitcast %"struct.std::complex"* %.omp.reduction.element.ascast to i8*
  store i8* %25, i8** %7, align 8, !tbaa !9
  %26 = icmp eq i16 %3, 0
  %27 = icmp eq i16 %3, 1
  %28 = icmp ult i16 %1, %2
  %29 = and i1 %27, %28
  %30 = icmp eq i16 %3, 2
  %31 = and i16 %1, 1
  %32 = icmp eq i16 %31, 0
  %33 = and i1 %30, %32
  %34 = icmp sgt i16 %2, 0
  %35 = and i1 %33, %34
  %36 = or i1 %26, %29
  %37 = or i1 %36, %35
  br i1 %37, label %then, label %ifcont

then:                                             ; preds = %.shuffle.exit
  %38 = bitcast [1 x i8*]* %.omp.reduction.remote_reduce_list.ascast to i8*
  call void @"_omp$reduction$reduction_func.2"(i8* %0, i8* %38) #3
  br label %ifcont

ifcont:                                           ; preds = %.shuffle.exit, %then
  %39 = icmp uge i16 %1, %2
  %40 = and i1 %27, %39
  br i1 %40, label %then4, label %ifcont6

then4:                                            ; preds = %ifcont
  %41 = load i8*, i8** %7, align 8
  %42 = load i8*, i8** %5, align 8
  %43 = bitcast i8* %41 to %"struct.std::complex"*
  %44 = bitcast i8* %42 to %"struct.std::complex"*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %42, i8* align 8 %41, i64 16, i1 false), !tbaa.struct !15
  br label %ifcont6

ifcont6:                                          ; preds = %ifcont, %then4
  ret void
}

!4 = !{!"clang version 14.0.0"}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"any pointer", !7, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"double", !7, i64 0}
