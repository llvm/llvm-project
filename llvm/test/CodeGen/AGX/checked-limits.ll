; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+agx-checked -stop-after=agx-checked-accesses < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+agx-checked -verify-machineinstrs -filetype=obj < %s -o /dev/null

; At most 16 bounds slots: the 17th pointer base is checked for null and
; alignment only. A kernel without accesses keeps its hidden arguments and the
; early return, but has no report block.

; CHECK-LABEL: define agx_kernel void @many(
; CHECK-SAME: "agx-checked"="lo15" %{{[0-9]+}}, i64 "agx-checked"="hi15" %{{[0-9]+}}) #[[MANY:[0-9]+]]
; The last store: null and alignment, no bounds.
; CHECK: [[A:%[0-9]+]] = ptrtoint ptr addrspace(1) %p16 to i64
; CHECK-NEXT: [[L:%[0-9]+]] = trunc i64 [[A]] to i32
; CHECK-NEXT: and i32 [[L]], 3
; CHECK-NEXT: icmp ne i32
; CHECK-NEXT: icmp ult i64 [[A]], 4096
; CHECK-NEXT: or i1
; CHECK-NEXT: br i1
define agx_kernel void @many(ptr addrspace(1) %p0, ptr addrspace(1) %p1, ptr addrspace(1) %p2, ptr addrspace(1) %p3, ptr addrspace(1) %p4, ptr addrspace(1) %p5, ptr addrspace(1) %p6, ptr addrspace(1) %p7, ptr addrspace(1) %p8, ptr addrspace(1) %p9, ptr addrspace(1) %p10, ptr addrspace(1) %p11, ptr addrspace(1) %p12, ptr addrspace(1) %p13, ptr addrspace(1) %p14, ptr addrspace(1) %p15, ptr addrspace(1) %p16) {
  store i32 0, ptr addrspace(1) %p0, align 4
  store i32 0, ptr addrspace(1) %p1, align 4
  store i32 0, ptr addrspace(1) %p2, align 4
  store i32 0, ptr addrspace(1) %p3, align 4
  store i32 0, ptr addrspace(1) %p4, align 4
  store i32 0, ptr addrspace(1) %p5, align 4
  store i32 0, ptr addrspace(1) %p6, align 4
  store i32 0, ptr addrspace(1) %p7, align 4
  store i32 0, ptr addrspace(1) %p8, align 4
  store i32 0, ptr addrspace(1) %p9, align 4
  store i32 0, ptr addrspace(1) %p10, align 4
  store i32 0, ptr addrspace(1) %p11, align 4
  store i32 0, ptr addrspace(1) %p12, align 4
  store i32 0, ptr addrspace(1) %p13, align 4
  store i32 0, ptr addrspace(1) %p14, align 4
  store i32 0, ptr addrspace(1) %p15, align 4
  store i32 0, ptr addrspace(1) %p16, align 4
  ret void
}

; CHECK-LABEL: define agx_kernel void @empty(
; CHECK-SAME: i32 %n, ptr addrspace(1) "agx-checked"="record" [[R:%[0-9]+]], i32 "agx-checked"="threadgroup-bytes" %{{[0-9]+}})
; CHECK-NEXT: entry:
; CHECK-NEXT: %claimed = load i32, ptr addrspace(1) [[R]]
; CHECK-NOT: checked.report
; CHECK: ret void
define agx_kernel void @empty(i32 %n) {
entry:
  ret void
}
; CHECK: attributes #[[MANY]] = {{.*}}"agx-checked-slots"="0:-1,1:-1,2:-1,3:-1,4:-1,5:-1,6:-1,7:-1,8:-1,9:-1,10:-1,11:-1,12:-1,13:-1,14:-1,15:-1"
