; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-lower-generic-addresses < %s | FileCheck %s --check-prefix=IR
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s -o /dev/null

; Threadgroup pointers cast to generic ones are tagged with bit 63, and a
; generic access that may reach either memory tests the tag.

@buf = internal addrspace(3) global [64 x i32] undef
declare i32 @llvm.agx.thread.position.in.threadgroup.x()

; IR-LABEL: @select_load
; IR: %[[OFF:.*]] = ptrtoint ptr addrspace(3) %{{.*}} to i32
; IR: zext i32 %[[OFF]] to i64
; IR: or i64 %{{.*}}, -9223372036854775808
; IR: icmp slt i64 %{{.*}}, 0
; IR: load i32, ptr addrspace(3)
; IR: load i32, ptr addrspace(1)
; IR: phi i32
; CHECK-LABEL: select_load:
; CHECK: threadgroup_load
; CHECK: device_load
define agx_kernel void @select_load(ptr addrspace(1) %out, ptr addrspace(1) %dev, i32 %sel) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %s = getelementptr [64 x i32], ptr addrspace(3) @buf, i32 0, i32 %t
  store i32 %t, ptr addrspace(3) %s
  %gs = addrspacecast ptr addrspace(3) %s to ptr
  %gd = addrspacecast ptr addrspace(1) %dev to ptr
  %c = icmp eq i32 %sel, 0
  %p = select i1 %c, ptr %gs, ptr %gd
  %v = load i32, ptr %p
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %v, ptr addrspace(1) %po
  ret void
}

; IR-LABEL: @round_trip
; IR: %[[G:.*]] = or i64
; IR: %[[B:.*]] = trunc i64 %{{.*}} to i32
; IR: inttoptr i32 %[[B]] to ptr addrspace(3)
; IR-NOT: icmp slt
; IR: ret void
define agx_kernel void @round_trip(i32 %v) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %s = getelementptr [64 x i32], ptr addrspace(3) @buf, i32 0, i32 %t
  %g = addrspacecast ptr addrspace(3) %s to ptr
  %b = addrspacecast ptr %g to ptr addrspace(3)
  store i32 %v, ptr addrspace(3) %b
  ret void
}

; Functions without casts to or from threadgroup pointers keep their generic
; accesses as device accesses.
; IR-LABEL: @device_only
; IR-NOT: icmp slt
; IR: store i32 %v, ptr %p
define agx_kernel void @device_only(ptr %p, i32 %v) {
  store i32 %v, ptr %p
  ret void
}
