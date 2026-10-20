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
; IR: lshr i64 %{{.*}}, 32
; IR: icmp slt i32 %{{.*}}, 0
; IR: load i32, ptr addrspace(3)
; IR: load i32, ptr addrspace(1)
; IR: phi i32
; CHECK-LABEL: select_load:
; CHECK-DAG: threadgroup_load
; CHECK-DAG: device_load
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

; A cast pair folds away before the lowering.
; IR-LABEL: @round_trip
; IR-NOT: ptrtoint
; IR: store i32 %v, ptr addrspace(3) %s
define agx_kernel void @round_trip(i32 %v) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %s = getelementptr [64 x i32], ptr addrspace(3) @buf, i32 0, i32 %t
  %g = addrspacecast ptr addrspace(3) %s to ptr
  %b = addrspacecast ptr %g to ptr addrspace(3)
  store i32 %v, ptr addrspace(3) %b
  ret void
}

; A generic pointer cast to a threadgroup one keeps its low 32 bits.
; IR-LABEL: @from_generic
; IR: %[[I:.*]] = ptrtoint ptr %g to i64
; IR: %[[B:.*]] = trunc i64 %[[I]] to i32
; IR: %[[S:.*]] = inttoptr i32 %[[B]] to ptr addrspace(3)
; IR: store i32 %v, ptr addrspace(3) %[[S]]
; CHECK-LABEL: from_generic:
; CHECK: threadgroup_store
define agx_kernel void @from_generic(ptr %g, i32 %v) {
  %s = addrspacecast ptr %g to ptr addrspace(3)
  store i32 %v, ptr addrspace(3) %s
  ret void
}

; Functions without casts to or from threadgroup pointers keep their generic
; accesses as device accesses.
; IR-LABEL: @device_only
; IR-NOT: lshr
; IR: store i32 %v, ptr %p
define agx_kernel void @device_only(ptr %p, i32 %v) {
  store i32 %v, ptr %p
  ret void
}

; Private pointers are tagged with bit 62. With both tagged memories, an
; access tests the top two bits: 2 or 3 is threadgroup, 1 private, 0 device.
; Atomics have no private form (PTX leaves them undefined).
; IR-LABEL: @three_spaces
; IR: or i64 %{{.*}}, -9223372036854775808
; IR: or i64 %{{.*}}, 4611686018427387904
; IR: lshr i32 %{{.*}}, 30
; IR: icmp uge i32 %{{.*}}, 2
; IR: load i32, ptr addrspace(3)
; IR: icmp eq i32 %{{.*}}, 1
; IR: load i32, ptr addrspace(5)
; IR: load i32, ptr addrspace(1)
; IR: phi i32
; IR: atomicrmw add ptr addrspace(3)
; IR-NOT: atomicrmw add ptr addrspace(5)
; IR: atomicrmw add ptr addrspace(1)
; CHECK-LABEL: three_spaces:
; CHECK-DAG: threadgroup_load
; CHECK-DAG: device_load
define agx_kernel void @three_spaces(ptr addrspace(1) %out, ptr addrspace(1) %dev, i32 %sel) {
  %priv = alloca [4 x i32], addrspace(5)
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %pp = getelementptr [4 x i32], ptr addrspace(5) %priv, i32 0, i32 %sel
  store volatile i32 %sel, ptr addrspace(5) %pp
  %s = getelementptr [64 x i32], ptr addrspace(3) @buf, i32 0, i32 %t
  store i32 %t, ptr addrspace(3) %s
  %gs = addrspacecast ptr addrspace(3) %s to ptr
  %gp = addrspacecast ptr addrspace(5) %pp to ptr
  %gd = addrspacecast ptr addrspace(1) %dev to ptr
  %c0 = icmp eq i32 %sel, 0
  %c1 = icmp eq i32 %sel, 1
  %p01 = select i1 %c1, ptr %gp, ptr %gd
  %p = select i1 %c0, ptr %gs, ptr %p01
  %v = load i32, ptr %p
  %a = atomicrmw add ptr %p, i32 1 monotonic
  %w = add i32 %v, %a
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %w, ptr addrspace(1) %po
  ret void
}
