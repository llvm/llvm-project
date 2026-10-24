; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-lower-generic-addresses %s -o - | FileCheck %s --check-prefix=IR --implicit-check-not=addrspacecast
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs -filetype=obj %s -o /dev/null
;
; Constants need the same tagging/truncation as instruction casts. Include
; aggregate users, phi incoming values and the private comparison shape
; produced by address-space inference in large device-library functions.
;
; IR-LABEL: @constant_local_casts(
; IR: inttoptr (i64 -1408 to ptr)
; IR: inttoptr (i64 -9223372036854775744 to ptr)
; IR: or i64 {{.*}}, 4611686018427387904
; IR: or i64 {{.*}}, -9223372036854775808

declare i32 @llvm.agx.thread.index.in.simdgroup()

define agx_kernel void @constant_local_casts(ptr addrspace(1) %out, i32 %mode) {
entry:
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %offset = mul i32 %lane, 10
  %base = getelementptr i32, ptr addrspace(1) %out, i32 %offset
  store i32 ptrtoint (ptr addrspace(5) addrspacecast (ptr inttoptr (i64 -1408 to ptr) to ptr addrspace(5)) to i32), ptr addrspace(1) %base
  %p1 = getelementptr i32, ptr addrspace(1) %base, i32 1
  store i32 ptrtoint (ptr addrspace(3) addrspacecast (ptr inttoptr (i64 -9223372036854775744 to ptr) to ptr addrspace(3)) to i32), ptr addrspace(1) %p1
  %p2 = getelementptr i32, ptr addrspace(1) %base, i32 2
  store i64 ptrtoint (ptr addrspacecast (ptr addrspace(5) inttoptr (i32 -1408 to ptr addrspace(5)) to ptr) to i64), ptr addrspace(1) %p2, align 8
  %p4 = getelementptr i32, ptr addrspace(1) %base, i32 4
  store i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) inttoptr (i32 96 to ptr addrspace(3)) to ptr) to i64), ptr addrspace(1) %p4, align 8
  %p6 = getelementptr i32, ptr addrspace(1) %base, i32 6
  store <2 x i32> <i32 ptrtoint (ptr addrspace(5) addrspacecast (ptr inttoptr (i64 -1408 to ptr) to ptr addrspace(5)) to i32), i32 ptrtoint (ptr addrspace(3) addrspacecast (ptr inttoptr (i64 -9223372036854775744 to ptr) to ptr addrspace(3)) to i32)>, ptr addrspace(1) %p6, align 8
  %flipped = xor i32 %lane, %mode
  %bit = and i32 %flipped, 1
  %even = icmp eq i32 %bit, 0
  br i1 %even, label %left, label %right
left:
  br label %join
right:
  br label %join
join:
  %picked = phi ptr addrspace(5) [ addrspacecast (ptr inttoptr (i64 -1408 to ptr) to ptr addrspace(5)), %left ], [ addrspacecast (ptr inttoptr (i64 64 to ptr) to ptr addrspace(5)), %right ]
  %low = ptrtoint ptr addrspace(5) %picked to i32
  %p8 = getelementptr i32, ptr addrspace(1) %base, i32 8
  store i32 %low, ptr addrspace(1) %p8
  %is_sentinel = icmp eq ptr addrspace(5) %picked, addrspacecast (ptr inttoptr (i64 -1408 to ptr) to ptr addrspace(5))
  %flag = zext i1 %is_sentinel to i32
  %p9 = getelementptr i32, ptr addrspace(1) %base, i32 9
  store i32 %flag, ptr addrspace(1) %p9
  ret void
}
