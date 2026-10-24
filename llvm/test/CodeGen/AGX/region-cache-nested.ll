; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs -filetype=obj %s -o /dev/null
;
; Nested divergent regions inside a loop exercise node-cache invalidation
; between child and parent region passes. The downstream simulator checks
; every lane against an independent integer reference for several trip counts.

declare i32 @llvm.agx.thread.position.in.grid.x()

define agx_kernel void @region_cache_nested(ptr addrspace(1) %out, i32 %n) {
entry:
  %id = call i32 @llvm.agx.thread.position.in.grid.x()
  %index = zext i32 %id to i64
  %ptr = getelementptr i32, ptr addrspace(1) %out, i64 %index
  %rem = urem i32 %id, 3
  %third = icmp eq i32 %rem, 0
  br label %header
header:
  %i = phi i32 [ 0, %entry ], [ %next, %latch ]
  %acc = phi i32 [ 0, %entry ], [ %sum, %latch ]
  %more = icmp ult i32 %i, %n
  br i1 %more, label %body, label %exit
body:
  %lane_iter = add i32 %id, %i
  %parity = and i32 %lane_iter, 1
  %even = icmp eq i32 %parity, 0
  br i1 %even, label %then, label %else
then:
  %step = add i32 %i, 1
  %scaled = mul i32 %step, 3
  %even_sum = add i32 %acc, %scaled
  br label %latch
else:
  br i1 %third, label %seven, label %two
seven:
  %seven_sum = add i32 %acc, 7
  br label %odd_join
two:
  %two_sum = add i32 %acc, 2
  br label %odd_join
odd_join:
  %odd_sum = phi i32 [ %seven_sum, %seven ], [ %two_sum, %two ]
  br label %latch
latch:
  %sum = phi i32 [ %even_sum, %then ], [ %odd_sum, %odd_join ]
  %next = add i32 %i, 1
  br label %header
exit:
  store i32 %acc, ptr addrspace(1) %ptr
  ret void
}
