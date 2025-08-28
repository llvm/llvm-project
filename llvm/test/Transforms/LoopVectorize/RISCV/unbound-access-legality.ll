; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -disable-output 2>&1 -mtriple=riscv64 -mattr=+v -S %s | FileCheck %s

define ptr @unbound_strided_access(ptr %first, ptr %last, i32 %value) {
; CHECK-LABEL: LV: Checking a loop in 'unbound_strided_access'
; CHECK:       LV: Not vectorizing: Loop contains strided unbound access.
entry:
  %cond = icmp eq ptr %first, %last
  br i1 %cond, label %return, label %for.body

for.body:
  %first.addr = phi ptr [ %first, %entry ], [ %first.next, %for.inc ]
  %1 = load i32, ptr %first.addr, align 4
  %cond2 = icmp eq i32 %1, %value
  br i1 %cond2, label %for.end, label %for.inc

for.inc:
  %first.next = getelementptr inbounds i32, ptr %first.addr, i64 2
  %cond3 = icmp eq ptr %first.next, %last
  br i1 %cond3, label %for.end, label %for.body

for.end:
  %retval.ph = phi ptr [ %first.addr, %for.body ], [ %last, %for.inc ]
  br label %return

return:
  %retval = phi ptr [ %first, %entry ], [ %retval.ph, %for.end ]
  ret ptr %retval
}

define ptr @single_unbound_access(ptr %first, ptr %last, i32 %value) {
; CHECK-LABEL: LV: Checking a loop in 'single_unbound_access'
; CHECK:       LV: We can vectorize this loop!
; CHECK-NEXT:  LV: Not vectorizing: Auto-vectorization of loops with speculative load is not supported.
entry:
  %cond = icmp eq ptr %first, %last
  br i1 %cond, label %return, label %for.body

for.body:
  %first.addr = phi ptr [ %first, %entry ], [ %first.next, %for.inc ]
  %1 = load i32, ptr %first.addr, align 4
  %cond2 = icmp eq i32 %1, %value
  br i1 %cond2, label %for.end, label %for.inc

for.inc:
  %first.next = getelementptr inbounds i32, ptr %first.addr, i64 1
  %cond3 = icmp eq ptr %first.next, %last
  br i1 %cond3, label %for.end, label %for.body

for.end:
  %retval.ph = phi ptr [ %first.addr, %for.body ], [ %last, %for.inc ]
  br label %return

return:
  %retval = phi ptr [ %first, %entry ], [ %retval.ph, %for.end ]
  ret ptr %retval
}
