; This test documents how the IR dumped for loop passes differs with -print-loop-func-scope
; and -print-module-scope
;   - Without -print-loop-func-scope, dumps only the loop, with 3 sections- preheader,
;     loop, and exit blocks
;   - With -print-loop-func-scope, dumps only the function which contains the loop
;   - With -print-module-scope, dumps the entire module containing the loop, and disregards
;     the -print-loop-func-scope flag.

; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes=licm -print-after=licm\
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

; MODULEWITHLOOP: IR Dump After LICMPass
; MODULEWITHLOOP: ModuleID =
; MODULEWITHLOOP: define void @foo
; MODULEWITHLOOP-NOT: Preheader:
; MODULEWITHLOOP-NOT: Loop:
; MODULEWITHLOOP-NOT: Exit blocks
; MODULEWITHLOOP: define void @bar



define void @foo(i32 %0, i32 %1, i32* %2) {
  %4 = icmp sgt i32 %0, 0
  br i1 %4, label %5, label %8

5:                                                ; preds = %3
  br label %12

6:                                                ; preds = %12
  %7 = phi i32 [ %15, %12 ]
  br label %8

8:                                                ; preds = %6, %3
  %9 = phi i32 [ 1, %3 ], [ %7, %6 ]
  %10 = add nsw i32 %1, %0
  %11 = add nsw i32 %10, %9
  store i32 %11, i32* %2, align 4
  ret void

12:                                               ; preds = %5, %12
  %13 = phi i32 [ %16, %12 ], [ 0, %5 ]
  %14 = phi i32 [ %15, %12 ], [ 1, %5 ]
  %15 = mul nsw i32 %14, %1
  %16 = add nuw nsw i32 %13, 1
  %17 = icmp eq i32 %16, %0
  br i1 %17, label %6, label %12
}

define void @bar() {
  ret void
}
