; REQUIRES: asserts
; RUN: opt -S -passes=dfa-jump-threading %s -debug-only=dfa-jump-threading 2>&1 | FileCheck %s

; CHECK-COUNT-3: Exiting early due to unpredictability heuristic.

@.str.1 = private unnamed_addr constant [3 x i8] c"10\00", align 1
@.str.2 = private unnamed_addr constant [3 x i8] c"30\00", align 1
@.str.3 = private unnamed_addr constant [3 x i8] c"20\00", align 1
@.str.4 = private unnamed_addr constant [3 x i8] c"40\00", align 1

define void @test1(i32 noundef %num, i32 noundef %num2) {
entry:
  br label %while.body

while.body:                                       ; preds = %entry, %sw.epilog
  %num.addr.0 = phi i32 [ %num, %entry ], [ %num.addr.1, %sw.epilog ]
  switch i32 %num.addr.0, label %sw.default [
    i32 10, label %sw.bb
    i32 30, label %sw.bb1
    i32 20, label %sw.bb2
    i32 40, label %sw.bb3
  ]

sw.bb:                                            ; preds = %while.body
  %call.i = tail call i32 @bar(ptr noundef nonnull @.str.1)
  br label %sw.epilog

sw.bb1:                                           ; preds = %while.body
  %call.i4 = tail call i32 @bar(ptr noundef nonnull @.str.2)
  br label %sw.epilog

sw.bb2:                                           ; preds = %while.body
  %call.i5 = tail call i32 @bar(ptr noundef nonnull @.str.3)
  br label %sw.epilog

sw.bb3:                                           ; preds = %while.body
  %call.i6 = tail call i32 @bar(ptr noundef nonnull @.str.4)
  %call = tail call noundef i32 @foo()
  %add = add nsw i32 %call, %num2
  br label %sw.epilog

sw.default:                                       ; preds = %while.body
  ret void

sw.epilog:                                        ; preds = %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb
  %num.addr.1 = phi i32 [ %add, %sw.bb3 ], [ 40, %sw.bb2 ], [ 20, %sw.bb1 ], [ 30, %sw.bb ]
  br label %while.body
}


define void @test2(i32 noundef %num, i32 noundef %num2) {
entry:
  br label %while.body

while.body:                                       ; preds = %entry, %sw.epilog
  %num.addr.0 = phi i32 [ %num, %entry ], [ %num.addr.1, %sw.epilog ]
  switch i32 %num.addr.0, label %sw.default [
    i32 10, label %sw.epilog
    i32 30, label %sw.bb1
    i32 20, label %sw.bb2
    i32 40, label %sw.bb3
  ]

sw.bb1:                                           ; preds = %while.body
  br label %sw.epilog

sw.bb2:                                           ; preds = %while.body
  br label %sw.epilog

sw.bb3:                                           ; preds = %while.body
  br label %sw.epilog

sw.default:                                       ; preds = %while.body
  ret void

sw.epilog:                                        ; preds = %while.body, %sw.bb3, %sw.bb2, %sw.bb1
  %.str.4.sink = phi ptr [ @.str.4, %sw.bb3 ], [ @.str.3, %sw.bb2 ], [ @.str.2, %sw.bb1 ], [ @.str.1, %while.body ]
  %num.addr.1 = phi i32 [ %num2, %sw.bb3 ], [ 40, %sw.bb2 ], [ 20, %sw.bb1 ], [ 30, %while.body ]
  %call.i6 = tail call i32 @bar(ptr noundef nonnull %.str.4.sink)
  br label %while.body
}


define void @test3(i32 noundef %num, i32 noundef %num2) {
entry:
  %add = add nsw i32 %num2, 40
  br label %while.body

while.body:                                       ; preds = %entry, %sw.epilog
  %num.addr.0 = phi i32 [ %num, %entry ], [ %num.addr.1, %sw.epilog ]
  switch i32 %num.addr.0, label %sw.default [
    i32 10, label %sw.bb
    i32 30, label %sw.bb1
    i32 20, label %sw.bb2
    i32 40, label %sw.bb3
  ]

sw.bb:                                            ; preds = %while.body
  %call.i = tail call i32 @bar(ptr noundef nonnull @.str.1)
  br label %sw.epilog

sw.bb1:                                           ; preds = %while.body
  %call.i5 = tail call i32 @bar(ptr noundef nonnull @.str.2)
  br label %sw.epilog

sw.bb2:                                           ; preds = %while.body
  %call.i6 = tail call i32 @bar(ptr noundef nonnull @.str.3)
  br label %sw.epilog

sw.bb3:                                           ; preds = %while.body
  %call.i7 = tail call i32 @bar(ptr noundef nonnull @.str.4)
  br label %sw.epilog

sw.default:                                       ; preds = %while.body
  ret void

sw.epilog:                                        ; preds = %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb
  %num.addr.1 = phi i32 [ %add, %sw.bb3 ], [ 40, %sw.bb2 ], [ 20, %sw.bb1 ], [ 30, %sw.bb ]
  br label %while.body
}


declare noundef i32 @foo()
declare noundef i32 @bar(ptr nocapture noundef readonly)
