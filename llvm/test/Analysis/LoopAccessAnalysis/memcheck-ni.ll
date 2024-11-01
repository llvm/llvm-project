; RUN: opt -passes=loop-versioning -S < %s | FileCheck %s

; NB: addrspaces 10-13 are non-integral
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"

%jl_value_t = type opaque
%jl_array_t = type { i8 addrspace(13)*, i64, i16, i16, i32 }

define void @"japi1_permutedims!_33509"(%jl_value_t addrspace(10)**) {
; CHECK: [[CMP:%[^ ]*]] = icmp ult double addrspace(13)* [[A:%[^ ]*]], [[B:%[^ ]*]]
; CHECK: [[SELECT:%[^ ]*]] = select i1 %18, double addrspace(13)* [[A]], double addrspace(13)* [[B]]
top:
  %1 = alloca [3 x i64], align 8
  %2 = load %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %0, align 8
  %3 = getelementptr inbounds %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %0, i64 1
  %4 = load %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %3, align 8
  %5 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 0
  store i64 1, i64* %5, align 8
  %6 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 1
  %7 = load i64, i64* inttoptr (i64 24 to i64*), align 8
  %8 = addrspacecast %jl_value_t addrspace(10)* %4 to %jl_value_t addrspace(11)*
  %9 = bitcast %jl_value_t addrspace(11)* %8 to double addrspace(13)* addrspace(11)*
  %10 = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %9, align 8
  %11 = addrspacecast %jl_value_t addrspace(10)* %2 to %jl_value_t addrspace(11)*
  %12 = bitcast %jl_value_t addrspace(11)* %11 to double addrspace(13)* addrspace(11)*
  %13 = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %12, align 8
  %14 = load i64, i64* %6, align 8
  br label %L74

L74:
  %value_phi20 = phi i64 [ 1, %top ], [ %22, %L74 ]
  %value_phi21 = phi i64 [ 1, %top ], [ %23, %L74 ]
  %value_phi22 = phi i64 [ 1, %top ], [ %25, %L74 ]
  %15 = add i64 %value_phi21, -1
  %16 = getelementptr inbounds double, double addrspace(13)* %10, i64 %15
  %17 = bitcast double addrspace(13)* %16 to i64 addrspace(13)*
  %18 = load i64, i64 addrspace(13)* %17, align 8
  %19 = add i64 %value_phi20, -1
  %20 = getelementptr inbounds double, double addrspace(13)* %13, i64 %19
  %21 = bitcast double addrspace(13)* %20 to i64 addrspace(13)*
  store i64 %18, i64 addrspace(13)* %21, align 8
  %22 = add i64 %value_phi20, 1
  %23 = add i64 %14, %value_phi21
  %24 = icmp eq i64 %value_phi22, %7
  %25 = add i64 %value_phi22, 1
  br i1 %24, label %L94, label %L74

L94:
  ret void
}
