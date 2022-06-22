; RUN: llc < %s -mtriple=x86_64-- -verify-machineinstrs | FileCheck %s

; rdar://9692967

define void @t1(ptr %p, i32 %b) nounwind {
entry:
  %p.addr = alloca ptr, align 8
  store ptr %p, ptr %p.addr, align 8
  %tmp = load ptr, ptr %p.addr, align 8
; CHECK-LABEL: t1:
; CHECK: movl    $2147483648, %eax
; CHECK: lock orq %r{{.*}}, (%r{{.*}})
  %0 = atomicrmw or ptr %tmp, i64 2147483648 seq_cst
  ret void
}

define void @t2(ptr %p, i32 %b) nounwind {
entry:
  %p.addr = alloca ptr, align 8
  store ptr %p, ptr %p.addr, align 8
  %tmp = load ptr, ptr %p.addr, align 8
; CHECK-LABEL: t2:
; CHECK: lock orq $2147483644, (%r{{.*}})
  %0 = atomicrmw or ptr %tmp, i64 2147483644 seq_cst
  ret void
}
