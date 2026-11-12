; This test documents how the IR dumped for loop passes differs with -print-loop-func-scope
; and -print-module-scope
;   - Without -print-loop-func-scope, dumps only the loop, with 3 sections- preheader,
;     loop, and exit blocks
;   - With -print-loop-func-scope, dumps only the function which contains the loop
;   - With -print-module-scope, dumps the entire module containing the loop, and disregards
;     the -print-loop-func-scope flag.

; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes=licm -print-after=licm \
; RUN:	   | FileCheck %s -check-prefix=VANILLA
; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes=licm -print-after=licm -print-loop-func-scope \
; RUN:	   | FileCheck %s -check-prefix=LOOPFUNC
; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes=licm -print-after=licm -print-module-scope \
; RUN:	   | FileCheck %s -check-prefix=MODULE
; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes=licm -print-after=licm -print-module-scope -print-loop-func-scope\
; RUN:	   | FileCheck %s -check-prefix=MODULEWITHLOOP

; VANILLA: IR Dump After LICMPass
; VANILLA-NOT: define void @foo
; VANILLA: Preheader:
; VANILLA: Loop:
; VANILLA: Exit blocks

; LOOPFUNC: IR Dump After LICMPass
; LOOPFUNC: (loop:
; LOOPFUNC: define void @foo
; LOOPFUNC-NOT: Preheader:
; LOOPFUNC-NOT: Loop:
; LOOPFUNC-NOT: Exit blocks

; MODULE: IR Dump After LICMPass
; MODULE: ModuleID =
; MODULE: define void @foo
; MODULE-NOT: Preheader:
; MODULE-NOT: Loop:
; MODULE-NOT: Exit blocks
; MODULE: define void @bar
; MODULE: declare void @baz(i32)

; MODULEWITHLOOP: IR Dump After LICMPass
; MODULEWITHLOOP: ModuleID =
; MODULEWITHLOOP: define void @foo
; MODULEWITHLOOP-NOT: Preheader:
; MODULEWITHLOOP-NOT: Loop:
; MODULEWITHLOOP-NOT: Exit blocks
; MODULEWITHLOOP: define void @bar
; MODULEWITHLOOP: declare void @baz(i32)

define void @foo(i32 %n) {
entry:
  br label %loop_cond

loop_cond:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop_body ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %loop_body, label %loop_end

loop_body:
  call void @baz(i32 %i)
  %i_next = add i32 %i, 1
  br label %loop_cond

loop_end:
  ret void
}

define void @bar() {
  ret void
}

declare void @baz(i32)
