; RUN: llc < %s -march=nvptx -mcpu=sm_70 -o - 2>&1  | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

@u128_max = internal addrspace(1) global i128 0, align 16
@u128_zero = internal addrspace(1) global i128 0, align 16
@i128_max = internal addrspace(1) global i128 0, align 16
@i128_min = internal addrspace(1) global i128 0, align 16
@v_u128_max = internal addrspace(1) global i128 0, align 16
@v_u128_zero = internal addrspace(1) global i128 0, align 16
@v_i128_max = internal addrspace(1) global i128 0, align 16
@v_i128_min = internal addrspace(1) global i128 0, align 16
@v64 = internal addrspace(1) global i64* null, align 8
@llvm.used = appending global [10 x i8*] [i8* bitcast (i128* addrspacecast (i128 addrspace(1)* @u128_max to i128*) to i8*), i8* bitcast (i128* addrspacecast (i128 addrspace(1)* @u128_zero to i128*) to i8*), i8* bitcast (i128* addrspacecast (i128 addrspace(1)* @i128_max to i128*) to i8*), i8* bitcast (i128* addrspacecast (i128 addrspace(1)* @i128_min to i128*) to i8*), i8* bitcast (i128* addrspacecast (i128 addrspace(1)* @v_u128_max to i128*) to i8*), i8* bitcast (i128* addrspacecast (i128 addrspace(1)* @v_u128_zero to i128*) to i8*), i8* bitcast (i128* addrspacecast (i128 addrspace(1)* @v_i128_max to i128*) to i8*), i8* bitcast (i128* addrspacecast (i128 addrspace(1)* @v_i128_min to i128*) to i8*), i8* bitcast (i64** addrspacecast (i64* addrspace(1)* @v64 to i64**) to i8*), i8* bitcast (void ()* @_Z6kernelv to i8*)], section "llvm.metadata"

; Function Attrs: alwaysinline
define void @_Z6kernelv() #0 {
  ; CHECK-LABLE: _Z6kernelv
  ; CHECK: mov.u64 [[U64_MAX:%rd[0-9]+]], -1;
  ; CHECK: mov.b128 [[U128_MAX:%rq[0-9]+]], {[[U64_MAX]], [[U64_MAX]]};
  ; CHECK: mov.u64 [[I128_MAX_HI:%rd[0-9]+]], 9223372036854775807;
  ; CHECK: mov.b128 [[I128_MAX:%rq[0-9]+]], {[[U64_MAX]], [[I128_MAX_HI]]};
  ; CHECK: mov.u64 [[I128_MIN_HI:%rd[0-9]+]], -9223372036854775808;
  ; CHECK: mov.u64 [[ZERO:%rd[0-9]+]], 0;
  ; CHECK: mov.b128 [[I128_MIN:%rq[0-9]+]], {[[ZERO]], [[I128_MIN_HI]]};
  ; CHECK: mov.b128 [[U128_ZERO:%rq[0-9]+]], {[[ZERO]], [[ZERO]]};
  
  %tmp = load i64*, i64** addrspacecast (i64* addrspace(1)* @v64 to i64**), align 8
  %add.ptr = getelementptr inbounds i64, i64* %tmp, i32 0
  %tmp1 = load i64*, i64** addrspacecast (i64* addrspace(1)* @v64 to i64**), align 8
  %add.ptr2 = getelementptr inbounds i64, i64* %tmp1, i32 1
  call void asm sideeffect "{\0A\09.reg .b64 hi;\0A\09.reg .b64 lo;\0A\09mov.b128 {lo, hi}, $0;\0A\09st.b64 [$1], lo;\0A\09st.b64 [$2], hi;\0A\09st.b128 [$3], $0;\0A\09}\0A\09", "q,l,l,l"(i128 -1, i64* %add.ptr, i64* %add.ptr2, i128* addrspacecast (i128 addrspace(1)* @v_u128_max to i128*)) #1
  %tmp3 = load i64*, i64** addrspacecast (i64* addrspace(1)* @v64 to i64**), align 8
  %add.ptr4 = getelementptr inbounds i64, i64* %tmp3, i32 2
  %tmp5 = load i64*, i64** addrspacecast (i64* addrspace(1)* @v64 to i64**), align 8
  %add.ptr6 = getelementptr inbounds i64, i64* %tmp5, i32 3
  call void asm sideeffect "{\0A\09.reg .b64 hi;\0A\09.reg .b64 lo;\0A\09mov.b128 {lo, hi}, $0;\0A\09st.b64 [$1], lo;\0A\09st.b64 [$2], hi;\0A\09st.b128 [$3], $0;\0A\09}\0A\09", "q,l,l,l"(i128 170141183460469231731687303715884105727, i64* %add.ptr4, i64* %add.ptr6, i128* addrspacecast (i128 addrspace(1)* @v_i128_max to i128*)) #1
  %tmp7 = load i64*, i64** addrspacecast (i64* addrspace(1)* @v64 to i64**), align 8
  %add.ptr8 = getelementptr inbounds i64, i64* %tmp7, i32 4
  %tmp9 = load i64*, i64** addrspacecast (i64* addrspace(1)* @v64 to i64**), align 8
  %add.ptr10 = getelementptr inbounds i64, i64* %tmp9, i32 5
  call void asm sideeffect "{\0A\09.reg .b64 hi;\0A\09.reg .b64 lo;\0A\09mov.b128 {lo, hi}, $0;\0A\09st.b64 [$1], lo;\0A\09st.b64 [$2], hi;\0A\09st.b128 [$3], $0;\0A\09}\0A\09", "q,l,l,l"(i128 -170141183460469231731687303715884105728, i64* %add.ptr8, i64* %add.ptr10, i128* addrspacecast (i128 addrspace(1)* @v_i128_min to i128*)) #1
  %tmp11 = load i64*, i64** addrspacecast (i64* addrspace(1)* @v64 to i64**), align 8
  %add.ptr12 = getelementptr inbounds i64, i64* %tmp11, i32 6
  %tmp13 = load i64*, i64** addrspacecast (i64* addrspace(1)* @v64 to i64**), align 8
  %add.ptr14 = getelementptr inbounds i64, i64* %tmp13, i32 7
  call void asm sideeffect "{\0A\09.reg .b64 hi;\0A\09.reg .b64 lo;\0A\09mov.b128 {lo, hi}, $0;\0A\09st.b64 [$1], lo;\0A\09st.b64 [$2], hi;\0A\09st.b128 [$3], $0;\0A\09}\0A\09", "q,l,l,l"(i128 0, i64* %add.ptr12, i64* %add.ptr14, i128* addrspacecast (i128 addrspace(1)* @v_u128_zero to i128*)) #1
  ret void
}

attributes #0 = { alwaysinline "nvvm.annotations_transplanted" "nvvm.kernel" }
attributes #1 = { nounwind }

!nvvmir.version = !{!0, !1, !0, !1, !1, !0, !0, !0, !1}

!0 = !{i32 2, i32 0, i32 3, i32 1}
!1 = !{i32 2, i32 0}
