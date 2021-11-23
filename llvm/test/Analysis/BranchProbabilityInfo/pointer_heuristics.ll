; RUN: opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

define i32 @cmp1(i32* readnone %0, i32* readnone %1) {
; CHECK: Printing analysis results of BPI for function 'cmp1':
  %3 = icmp eq i32* %0, %1
  br i1 %3, label %4, label %6
; CHECK:   edge  ->  probability is 0x30000000 / 0x80000000 = 37.50%
; CHECK:   edge  ->  probability is 0x50000000 / 0x80000000 = 62.50%

4:                                                ; preds = %2
  %5 = tail call i32 bitcast (i32 (...)* @f to i32 ()*)() #2
  br label %8
; CHECK:   edge  ->  probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

6:                                                ; preds = %2
  %7 = tail call i32 bitcast (i32 (...)* @g to i32 ()*)() #2
  br label %8
; CHECK:   edge  ->  probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

8:                                                ; preds = %6, %4
  %9 = phi i32 [ %5, %4 ], [ %7, %6 ]
  ret i32 %9
}

define i32 @cmp2(i32* readnone %0, i32* readnone %1) {
; CHECK: Printing analysis results of BPI for function 'cmp2':
  %3 = icmp eq i32* %0, %1
  br i1 %3, label %6, label %4
; CHECK:   edge  ->  probability is 0x30000000 / 0x80000000 = 37.50%
; CHECK:   edge  ->  probability is 0x50000000 / 0x80000000 = 62.50%

4:                                                ; preds = %2
  %5 = tail call i32 bitcast (i32 (...)* @f to i32 ()*)() #2
  br label %8
; CHECK:   edge  ->  probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

6:                                                ; preds = %2
  %7 = tail call i32 bitcast (i32 (...)* @g to i32 ()*)() #2
  br label %8
; CHECK:   edge  ->  probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

8:                                                ; preds = %6, %4
  %9 = phi i32 [ %5, %4 ], [ %7, %6 ]
  ret i32 %9
}

; CHECK: Printing analysis results of BPI for function 'cmp3':
define i32 @cmp3(i32* readnone %0) {
  %2 = icmp eq i32* %0, null
  br i1 %2, label %3, label %5
; CHECK:   edge  ->  probability is 0x30000000 / 0x80000000 = 37.50%
; CHECK:   edge  ->  probability is 0x50000000 / 0x80000000 = 62.50%

3:                                                ; preds = %1
  %4 = tail call i32 bitcast (i32 (...)* @f to i32 ()*)() #2
  br label %7
; CHECK:   edge  ->  probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

5:                                                ; preds = %1
  %6 = tail call i32 bitcast (i32 (...)* @g to i32 ()*)() #2
  br label %7
; CHECK:   edge  ->  probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

7:                                                ; preds = %5, %3
  %8 = phi i32 [ %6, %5 ], [ %4, %3 ]
  ret i32 %8
}

declare dso_local i32 @f(...) local_unnamed_addr #1
declare dso_local i32 @g(...) local_unnamed_addr #1
