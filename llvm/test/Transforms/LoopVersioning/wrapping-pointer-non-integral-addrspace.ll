; RUN: opt -loop-versioning -S < %s | FileCheck %s -check-prefix=LV

; NB: addrspaces 10-13 are non-integral
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"

; This matches the test case from PR38290
; Check that we expand the SCEV predicate check using GEP, rather
; than ptrtoint.

%jl_value_t = type opaque
%jl_array_t = type { i8 addrspace(13)*, i64, i16, i16, i32 }

declare i64 @julia_steprange_last_4949()

define void @"japi1_align!_9477"(%jl_value_t addrspace(10)** %arg) {
; LV-LAVEL: L26.lver.check
; LV: [[OFMul:%[^ ]*]]  = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 4, i64 [[Step:%[^ ]*]])
; LV-NEXT: [[OFMulResult:%[^ ]*]] = extractvalue { i64, i1 } [[OFMul]], 0
; LV-NEXT: [[OFMulOverflow:%[^ ]*]] = extractvalue { i64, i1 } [[OFMul]], 1
; LV: [[OFNegMulResult:%[^ ]*]] = sub i64 0, [[OFMulResult]]
; LV-NEXT: [[NegGEP:%[^ ]*]] = getelementptr i8, i8 addrspace(13)* [[Base:%[^ ]*]], i64 [[OFNegMulResult]]
; LV-NEXT: icmp ugt i8 addrspace(13)* [[NegGEP]], [[Base]]
; LV-NOT: inttoptr
; LV-NOT: ptrtoint
top:
  %tmp = load %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %arg, align 8
  %tmp1 = load i32, i32* inttoptr (i64 12 to i32*), align 4
  %tmp2 = sub i32 0, %tmp1
  %tmp3 = call i64 @julia_steprange_last_4949()
  %tmp4 = addrspacecast %jl_value_t addrspace(10)* %tmp to %jl_value_t addrspace(11)*
  %tmp5 = bitcast %jl_value_t addrspace(11)* %tmp4 to %jl_value_t addrspace(10)* addrspace(11)*
  %tmp6 = load %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)* addrspace(11)* %tmp5, align 8
  %tmp7 = addrspacecast %jl_value_t addrspace(10)* %tmp6 to %jl_value_t addrspace(11)*
  %tmp8 = bitcast %jl_value_t addrspace(11)* %tmp7 to i32 addrspace(13)* addrspace(11)*
  %tmp9 = load i32 addrspace(13)*, i32 addrspace(13)* addrspace(11)* %tmp8, align 8
  %tmp10 = sext i32 %tmp2 to i64
  br label %L26

L26:
  %value_phi3 = phi i64 [ 0, %top ], [ %tmp11, %L26 ]
  %tmp11 = add i64 %value_phi3, -1
  %tmp12 = getelementptr inbounds i32, i32 addrspace(13)* %tmp9, i64 %tmp11
  %tmp13 = load i32, i32 addrspace(13)* %tmp12, align 4
  %tmp14 = add i64 %tmp11, %tmp10
  %tmp15 = getelementptr inbounds i32, i32 addrspace(13)* %tmp9, i64 %tmp14
  store i32 %tmp13, i32 addrspace(13)* %tmp15, align 4
  %tmp16 = icmp eq i64 %value_phi3, %tmp3
  br i1 %tmp16, label %L45, label %L26

L45:
  ret void
}

