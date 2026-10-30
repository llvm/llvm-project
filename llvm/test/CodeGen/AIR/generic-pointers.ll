; Generic pointers (address space 4, agx-raise-to-air's hint) are emulated in
; the Metal pipeline with ZLUDA's AGX encoding: threadgroup addresses tagged
; with bit 63, private addresses with bits 63-62 = 01, device addresses as
; they are. An access through a generic pointer of known origin goes to that
; address space; of unknown origin, to the one its tag names, at run time.
; RUN: opt -mtriple=air -passes=air-legalize-unsupported-ir -S %s | FileCheck %s
; RUN: llc -mtriple=air -filetype=obj %s -o /dev/null

@s = internal addrspace(3) global [4 x i32] undef, align 4

; A cast from threadgroup memory sets bit 63; back, it is cleared; a known
; origin needs no test.
; CHECK-LABEL: define void @known(
; CHECK: [[I:%.*]] = or i64 ptrtoint (ptr addrspace(3) @s to i64), -9223372036854775808
; CHECK: [[T:%.*]] = and i64 {{%.*}}, 9223372036854775807
; CHECK: [[P:%.*]] = inttoptr i64 [[T]] to ptr addrspace(3)
; CHECK: store i32 1, ptr addrspace(3) [[P]]
; CHECK-NOT: generic.
; CHECK: ret void
define void @known(ptr addrspace(1) %o) "air-kernel" {
  %g = addrspacecast ptr addrspace(3) @s to ptr addrspace(4)
  store i32 1, ptr addrspace(4) %g
  %i = ptrtoint ptr addrspace(4) %g to i64
  store i64 %i, ptr addrspace(1) %o
  ret void
}

; A generic address read from memory: tested at run time.
; CHECK-LABEL: define void @unknown(
; CHECK: icmp slt i64 [[B:%.*]], 0
; CHECK: generic.notshared:
; CHECK: lshr i64 [[B]], 62
; CHECK: generic.device:
; CHECK: load i32, ptr addrspace(1)
; CHECK: generic.shared:
; CHECK: load i32, ptr addrspace(3)
; CHECK: generic.private:
; CHECK: load i32, ptr %
; CHECK: phi i32
define void @unknown(ptr addrspace(1) %o) "air-kernel" {
  %a = alloca i32
  %ga = addrspacecast ptr %a to ptr addrspace(4)
  %ia = ptrtoint ptr addrspace(4) %ga to i64
  store i64 %ia, ptr addrspace(1) %o
  %x = load i64, ptr addrspace(1) %o
  %p = inttoptr i64 %x to ptr addrspace(4)
  %v = load i32, ptr addrspace(4) %p
  store i32 %v, ptr addrspace(1) %o
  ret void
}
