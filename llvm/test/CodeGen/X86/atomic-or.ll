; RUN: llc < %s -march=x86-64 | FileCheck %s

; rdar://9692967

define void @t1(i64* %p, i32 %b) nounwind {
entry:
  %p.addr = alloca i64*, align 8
  store i64* %p, i64** %p.addr, align 8
  %tmp = load i64** %p.addr, align 8
; CHECK: t1:
; CHECK: movl    $2147483648, %eax
; CHECK: lock
; CHECK-NEXT: orq %r{{.*}}, (%r{{.*}})
  %0 = atomicrmw or i64* %tmp, i64 2147483648 seq_cst
  ret void
}

define void @t2(i64* %p, i32 %b) nounwind {
entry:
  %p.addr = alloca i64*, align 8
  store i64* %p, i64** %p.addr, align 8
  %tmp = load i64** %p.addr, align 8
; CHECK: t2:
; CHECK: lock
; CHECK-NEXT: orq $2147483644, (%r{{.*}})
  %0 = atomicrmw or i64* %tmp, i64 2147483644 seq_cst
  ret void
}
