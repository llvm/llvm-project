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
@v64 = internal addrspace(1) global ptr null, align 8

; Function Attrs: alwaysinline convergent mustprogress willreturn
define void @_Z6kernelv() {
  ; CHECK-LABEL: _Z6kernelv
  ; CHECK: mov.u64 [[U64_MAX:%rd[0-9]+]], -1; 
  ; CHECK: mov.b128 [[U128_MAX:%rq[0-9]+]], {[[U64_MAX]], [[U64_MAX]]}; 
  ; CHECK: mov.u64 [[I64_MAX:%rd[0-9]+]], 9223372036854775807;
  ; CHECK: mov.b128 [[I128_MAX:%rq[0-9]+]], {[[U64_MAX]], [[I64_MAX]]}
  ; CHECK: mov.u64 [[I64_MIN:%rd[0-9]+]], -9223372036854775808;
  ; CHECK: mov.u64 [[U64_ZERO:%rd[0-9]+]], 0;
  ; CHECK: mov.b128  [[I128_MIN:%rq[0-9]+]], {[[U64_ZERO]], [[I64_MIN]]}
  ; CHECK: mov.b128  [[U128_ZERO:%rq[0-9]+]], {[[U64_ZERO]], [[U64_ZERO]]}

  %tmp = load ptr, ptr addrspace(1) @v64, align 8
  %add.ptr2 = getelementptr inbounds i64, ptr %tmp, i64 1
  tail call void asm sideeffect "{\0A\09.reg .b64 hi;\0A\09.reg .b64 lo;\0A\09mov.b128 {lo, hi}, $0;\0A\09st.b64 [$1], lo;\0A\09st.b64 [$2], hi;\0A\09st.b128 [$3], $0;\0A\09}\0A\09", "q,l,l,l"(i128 -1, ptr %tmp, ptr nonnull %add.ptr2, ptr nonnull addrspacecast (ptr addrspace(1) @v_u128_max to ptr))
  %tmp3 = load ptr, ptr addrspace(1) @v64, align 8
  %add.ptr4 = getelementptr inbounds i64, ptr %tmp3, i64 2
  %add.ptr6 = getelementptr inbounds i64, ptr %tmp3, i64 3
  tail call void asm sideeffect "{\0A\09.reg .b64 hi;\0A\09.reg .b64 lo;\0A\09mov.b128 {lo, hi}, $0;\0A\09st.b64 [$1], lo;\0A\09st.b64 [$2], hi;\0A\09st.b128 [$3], $0;\0A\09}\0A\09", "q,l,l,l"(i128 170141183460469231731687303715884105727, ptr nonnull %add.ptr4, ptr nonnull %add.ptr6, ptr nonnull addrspacecast (ptr addrspace(1) @v_i128_max to ptr))
  %tmp7 = load ptr, ptr addrspace(1) @v64, align 8
  %add.ptr8 = getelementptr inbounds i64, ptr %tmp7, i64 4
  %add.ptr10 = getelementptr inbounds i64, ptr %tmp7, i64 5
  tail call void asm sideeffect "{\0A\09.reg .b64 hi;\0A\09.reg .b64 lo;\0A\09mov.b128 {lo, hi}, $0;\0A\09st.b64 [$1], lo;\0A\09st.b64 [$2], hi;\0A\09st.b128 [$3], $0;\0A\09}\0A\09", "q,l,l,l"(i128 -170141183460469231731687303715884105728, ptr nonnull %add.ptr8, ptr nonnull %add.ptr10, ptr nonnull addrspacecast (ptr addrspace(1) @v_i128_min to ptr))
  %tmp11 = load ptr, ptr addrspace(1) @v64, align 8
  %add.ptr12 = getelementptr inbounds i64, ptr %tmp11, i64 6
  %add.ptr14 = getelementptr inbounds i64, ptr %tmp11, i64 7
  tail call void asm sideeffect "{\0A\09.reg .b64 hi;\0A\09.reg .b64 lo;\0A\09mov.b128 {lo, hi}, $0;\0A\09st.b64 [$1], lo;\0A\09st.b64 [$2], hi;\0A\09st.b128 [$3], $0;\0A\09}\0A\09", "q,l,l,l"(i128 0, ptr nonnull %add.ptr12, ptr nonnull %add.ptr14, ptr nonnull addrspacecast (ptr addrspace(1) @v_u128_zero to ptr))
  ret void
}


!nvvmir.version = !{!2, !3, !2, !3, !3, !2, !2, !2, !3}

!2 = !{i32 2, i32 0, i32 3, i32 1}
!3 = !{i32 2, i32 0}
