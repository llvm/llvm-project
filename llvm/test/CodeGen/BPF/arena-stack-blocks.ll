; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena-stack < %s | FileCheck %s
;
; A local in the stack in arena is accessed through r12 without a cast. The
; address of the field is computed in one block and used by memset, which becomes
; stores when insns are selected, in another one. It must not get there in
; a register: 'rX = r12' is a number.

define internal void @use(ptr %p) noinline {
  store i64 1, ptr %p
  ret void
}

define void @f(i1 %c) {
; CHECK-LABEL: f:
; CHECK:       r12 += 32
; CHECK:       if
; CHECK-DAG:   *(u64 *)(r12 - 16) = 0
; CHECK-DAG:   *(u64 *)(r12 - 8) = 0
; CHECK:       exit
entry:
  %s = alloca [4 x i64], align 8
  %field = getelementptr i64, ptr %s, i64 2
  call void @use(ptr %s)
  call void @use(ptr %field)
  br i1 %c, label %then, label %join
then:
  call void @llvm.memset.p0.i64(ptr align 8 %field, i8 0, i64 16, i1 false)
  call void @use(ptr %s)
  br label %join
join:
  ret void
}
