; air.threads_per_grid and air.thread_index_in_threadgroup become kernel
; parameters described by their system-value roles, like the other
; system values.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK: define void @kernel({{.*}}<3 x i32> %gridsize, i32 %tgindex)
; CHECK: !"air.threads_per_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gridsize"}
; CHECK: !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tgindex"}

declare { i32, i32, i32 } @air.threads_per_grid.v3i32()
declare i32 @air.thread_index_in_threadgroup.i32()

define void @kernel(ptr addrspace(1) %out) {
entry:
  %g = call { i32, i32, i32 } @air.threads_per_grid.v3i32()
  %gx = extractvalue { i32, i32, i32 } %g, 0
  %i = call i32 @air.thread_index_in_threadgroup.i32()
  %s = add i32 %gx, %i
  store i32 %s, ptr addrspace(1) %out
  ret void
}
