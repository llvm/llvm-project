; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena-stack < %s | FileCheck %s

; The combiner turns the select of two loads into a load through a select of
; two addresses. The address of a local in the stack in arena is a number when
; it is in a register: it has to be cast.

define internal void @fill(ptr %p) noinline {
  store volatile i32 1, ptr %p
  ret void
}

define i32 @pick(i64 %c) {
; CHECK-LABEL: pick:
; CHECK: r12 += 16
; CHECK: [[P:r[0-9]+]] = r12
; CHECK: [[Q:r[0-9]+]] = addr_space_cast([[P]], 0, 1)
; CHECK: w0 = *(u32 *)([[Q]] + 0)
  %a = alloca [4 x i32], align 4
  call void @fill(ptr %a)
  %p1 = getelementptr i8, ptr %a, i64 4
  %p2 = getelementptr i8, ptr %a, i64 12
  %v1 = load i32, ptr %p1
  %v2 = load i32, ptr %p2
  %b = icmp eq i64 %c, 0
  %r = select i1 %b, i32 %v1, i32 %v2
  ret i32 %r
}
