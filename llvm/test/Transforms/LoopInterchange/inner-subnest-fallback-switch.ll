; REQUIRES: asserts
;
; RUN: opt < %s -passes=loop-interchange -debug-only=loop-interchange \
; RUN:     -print-after-all -disable-output 2>&1 | \
; RUN:     FileCheck %s --check-prefix=ENABLED
; RUN: opt < %s -passes=loop-interchange -debug-only=loop-interchange \
; RUN:     -loop-interchange-enable-inner-subnest-fallback=false \
; RUN:     -print-after-all -disable-output 2>&1 | \
; RUN:     FileCheck %s --check-prefix=DISABLED \
; RUN:     --implicit-check-not='Considering inner-subnest fallback'
;
; ENABLED: Considering inner-subnest fallback for loop nest
; ENABLED: IR Dump After LoopInterchangePass
;
; DISABLED-NOT: Considering inner-subnest fallback for loop nest
; DISABLED: IR Dump After LoopInterchangePass

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@A = global [4 x [8 x [8 x double]]] zeroinitializer
@B = global [4 x [8 x [8 x double]]] zeroinitializer

define void @fallback_switch() {
entry:
  br label %root.header

root.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %root.latch ]
  br label %a.i.header

a.i.header:
  %ai = phi i64 [ 0, %root.header ], [ %ai.next, %a.i.latch ]
  br label %a.j.body

a.j.body:
  %aj = phi i64 [ 1, %a.i.header ], [ %aj.next, %a.j.body ]
  %aj.prev = sub i64 %aj, 1
  %a.load.ptr = getelementptr inbounds [4 x [8 x [8 x double]]],
      ptr @A, i64 0, i64 %k, i64 %aj, i64 %ai
  %a.value = load double, ptr %a.load.ptr, align 8
  %a.next = fadd double %a.value, 1.000000e+00
  %a.store.ptr = getelementptr inbounds [4 x [8 x [8 x double]]],
      ptr @A, i64 0, i64 %k, i64 %aj.prev, i64 %ai
  store double %a.next, ptr %a.store.ptr, align 8
  %aj.next = add i64 %aj, 1
  %aj.done = icmp eq i64 %aj.next, 8
  br i1 %aj.done, label %a.i.latch, label %a.j.body

a.i.latch:
  %ai.next = add i64 %ai, 1
  %ai.done = icmp eq i64 %ai.next, 7
  br i1 %ai.done, label %b.i.header, label %a.i.header

b.i.header:
  %bi = phi i64 [ 0, %a.i.latch ], [ %bi.next, %b.i.latch ]
  br label %b.j.body

b.j.body:
  %bj = phi i64 [ 1, %b.i.header ], [ %bj.next, %b.j.body ]
  %bj.prev = sub i64 %bj, 1
  %b.load.ptr = getelementptr inbounds [4 x [8 x [8 x double]]],
      ptr @B, i64 0, i64 %k, i64 %bj, i64 %bi
  %b.value = load double, ptr %b.load.ptr, align 8
  %b.next = fadd double %b.value, 1.000000e+00
  %b.store.ptr = getelementptr inbounds [4 x [8 x [8 x double]]],
      ptr @B, i64 0, i64 %k, i64 %bj.prev, i64 %bi
  store double %b.next, ptr %b.store.ptr, align 8
  %bj.next = add i64 %bj, 1
  %bj.done = icmp eq i64 %bj.next, 8
  br i1 %bj.done, label %b.i.latch, label %b.j.body

b.i.latch:
  %bi.next = add i64 %bi, 1
  %bi.done = icmp eq i64 %bi.next, 7
  br i1 %bi.done, label %root.latch, label %b.i.header

root.latch:
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 4
  br i1 %k.done, label %exit, label %root.header

exit:
  ret void
}
