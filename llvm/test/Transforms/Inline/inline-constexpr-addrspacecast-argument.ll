; RUN: opt -S -passes=inline < %s | FileCheck %s
; RUN: opt -S -passes='cgscc(inline)' < %s | FileCheck %s
; RUN: opt -S -passes='module-inline' < %s | FileCheck %s

target datalayout = "e-p3:32:32-p4:64:64-n32"

@lds = internal addrspace(3) global [64 x i64] zeroinitializer

; CHECK-LABEL: @constexpr_addrspacecast_ptr_size_change(
; CHECK: load i64, ptr addrspace(4) addrspacecast (ptr addrspace(3) @lds to ptr addrspace(4))
; CHECK-NEXT: br
define void @constexpr_addrspacecast_ptr_size_change() #0 {
  %tmp0 = call i32 @foo(ptr addrspace(4) addrspacecast (ptr addrspace(3) @lds to ptr addrspace(4))) #1
  ret void
}

define i32 @foo(ptr addrspace(4) %arg) #1 {
bb:
  %tmp = getelementptr i64, ptr addrspace(4) %arg, i64 undef
  %tmp1 = load i64, ptr addrspace(4) %tmp
  br i1 undef, label %bb2, label %bb3

bb2:
  store i64 0, ptr addrspace(4) %tmp
  br label %bb3

bb3:
  unreachable
}

attributes #0 = { nounwind }
attributes #1 = { alwaysinline nounwind }
