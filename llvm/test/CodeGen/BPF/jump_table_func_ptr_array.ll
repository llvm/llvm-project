; RUN: llc -march=bpf -mcpu=v4 < %s | FileCheck %s

; The real jump table (@__const.foo.jt, all BlockAddress) is converted into a
; .jumptables entry and its global is removed.
; CHECK-LABEL: foo:
; CHECK:       r2 = BPF.JT.0.0 ll
; CHECK:       gotox r1
; CHECK:       .section .jumptables
; CHECK:       BPF.JT.0.0:

; The function-pointer array is preserved and still emitted.
; CHECK-LABEL: get:
; CHECK:       r2 = .Lkfuncs ll
; CHECK:       .Lkfuncs:
; CHECK:       .size .Lkfuncs, 16

@__const.foo.jt = private unnamed_addr constant [2 x ptr] [ptr blockaddress(@foo, %l1), ptr blockaddress(@foo, %l2)], align 8
@kfuncs = private unnamed_addr constant [2 x ptr] [ptr @ext1, ptr @ext2], align 8

declare void @ext1()
declare void @ext2()

define i32 @foo(i32 %a) {
entry:
  %rem = and i32 %a, 1
  %idx = zext nneg i32 %rem to i64
  %g = getelementptr inbounds nuw [2 x ptr], ptr @__const.foo.jt, i64 0, i64 %idx
  %t = load ptr, ptr %g, align 8
  indirectbr ptr %t, [label %l1, label %l2]
l1:
  br label %l2
l2:
  %ret = phi i32 [ 4, %l1 ], [ 3, %entry ]
  ret i32 %ret
}

define ptr @get(i64 %i) {
  %p = getelementptr inbounds [2 x ptr], ptr @kfuncs, i64 0, i64 %i
  %v = load ptr, ptr %p, align 8
  ret ptr %v
}
