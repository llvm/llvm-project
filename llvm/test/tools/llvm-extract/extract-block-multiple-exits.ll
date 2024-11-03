; RUN: llvm-extract -S -bb "func:region_start;exiting0;exiting1" --replace-with-call %s | FileCheck %s


; CHECK-LABEL: define void @func(ptr %arg, i1 %c0, i1 %c1, i1 %c2, i8 %dest) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %B.ce.loc = alloca i32, align 4
; CHECK-NEXT:    %c.loc = alloca i32, align 4
; CHECK-NEXT:    %b.loc = alloca i32, align 4
; CHECK-NEXT:    %a.loc = alloca i32, align 4
; CHECK-NEXT:    br i1 %c0, label %codeRepl, label %exit
; CHECK-EMPTY:
; CHECK-NEXT:  codeRepl:
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 -1, ptr %a.loc)
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 -1, ptr %b.loc)
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 -1, ptr %c.loc)
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 -1, ptr %B.ce.loc)
; CHECK-NEXT:    %targetBlock = call i16 @func.region_start(i1 %c1, i1 %c2, i8 %dest, ptr %a.loc, ptr %b.loc, ptr %c.loc, ptr %B.ce.loc)
; CHECK-NEXT:    %a.reload = load i32, ptr %a.loc, align 4
; CHECK-NEXT:    %b.reload = load i32, ptr %b.loc, align 4
; CHECK-NEXT:    %c.reload = load i32, ptr %c.loc, align 4
; CHECK-NEXT:    %B.ce.reload = load i32, ptr %B.ce.loc, align 4
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 -1, ptr %a.loc)
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 -1, ptr %b.loc)
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 -1, ptr %c.loc)
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 -1, ptr %B.ce.loc)
; CHECK-NEXT:    switch i16 %targetBlock, label %exit0 [
; CHECK-NEXT:      i16 0, label %exiting0.exit_crit_edge
; CHECK-NEXT:      i16 1, label %fallback
; CHECK-NEXT:      i16 2, label %exit1
; CHECK-NEXT:      i16 3, label %exit2
; CHECK-NEXT:    ]
; CHECK-EMPTY:
; CHECK-NEXT:  region_start:
; CHECK-NEXT:    %a = add i32 42, 1
; CHECK-NEXT:    br i1 %c1, label %exiting0, label %exiting1
; CHECK-EMPTY:
; CHECK-NEXT:  exiting0:
; CHECK-NEXT:    %b = add i32 42, 2
; CHECK-NEXT:    br i1 %c2, label %exiting0.exit_crit_edge, label %exit0.split
; CHECK-EMPTY:
; CHECK-NEXT:  exiting0.exit_crit_edge:
; CHECK-NEXT:    %b.merge_with_extracted4 = phi i32 [ %b.reload, %codeRepl ], [ %b, %exiting0 ]
; CHECK-NEXT:    br label %exit
; CHECK-EMPTY:
; CHECK-NEXT:  exiting1:
; CHECK-NEXT:    %c = add i32 42, 3
; CHECK-NEXT:    switch i8 %dest, label %fallback [
; CHECK-NEXT:      i8 0, label %exit0.split
; CHECK-NEXT:      i8 1, label %exit1
; CHECK-NEXT:      i8 2, label %exit2
; CHECK-NEXT:      i8 3, label %exit0.split
; CHECK-NEXT:    ]
; CHECK-EMPTY:
; CHECK-NEXT:  fallback:
; CHECK-NEXT:    unreachable
; CHECK-EMPTY:
; CHECK-NEXT:  exit:
; CHECK-NEXT:    %A = phi i32 [ 42, %entry ], [ %b.merge_with_extracted4, %exiting0.exit_crit_edge ]
; CHECK-NEXT:    store i32 %A, ptr %arg, align 4
; CHECK-NEXT:    br label %return
; CHECK-EMPTY:
; CHECK-NEXT:  exit0.split:
; CHECK-NEXT:    %b.merge_with_extracted3 = phi i32 [ %b, %exiting0 ], [ undef, %exiting1 ], [ undef, %exiting1 ]
; CHECK-NEXT:    %B.ce = phi i32 [ %b, %exiting0 ], [ %a, %exiting1 ], [ %a, %exiting1 ]
; CHECK-NEXT:    br label %exit0
; CHECK-EMPTY:
; CHECK-NEXT:  exit0:
; CHECK-NEXT:    %B.ce.merge_with_extracted = phi i32 [ %B.ce.reload, %codeRepl ], [ %B.ce, %exit0.split ]
; CHECK-NEXT:    %b.merge_with_extracted = phi i32 [ %b.reload, %codeRepl ], [ %b.merge_with_extracted3, %exit0.split ]
; CHECK-NEXT:    %a.merge_with_extracted2 = phi i32 [ %a.reload, %codeRepl ], [ %a, %exit0.split ]
; CHECK-NEXT:    store i32 %a.merge_with_extracted2, ptr %arg, align 4
; CHECK-NEXT:    store i32 %B.ce.merge_with_extracted, ptr %arg, align 4
; CHECK-NEXT:    br label %after
; CHECK-EMPTY:
; CHECK-NEXT:  exit1:
; CHECK-NEXT:    %c.merge_with_extracted5 = phi i32 [ %c.reload, %codeRepl ], [ %c, %exiting1 ]
; CHECK-NEXT:    %a.merge_with_extracted1 = phi i32 [ %a.reload, %codeRepl ], [ %a, %exiting1 ]
; CHECK-NEXT:    br label %after
; CHECK-EMPTY:
; CHECK-NEXT:  exit2:
; CHECK-NEXT:    %c.merge_with_extracted = phi i32 [ %c.reload, %codeRepl ], [ %c, %exiting1 ]
; CHECK-NEXT:    store i32 %c.merge_with_extracted, ptr %arg, align 4
; CHECK-NEXT:    store i32 %c.merge_with_extracted, ptr %arg, align 4
; CHECK-NEXT:    br label %return
; CHECK-EMPTY:
; CHECK-NEXT:  after:
; CHECK-NEXT:    %a.merge_with_extracted = phi i32 [ %a.merge_with_extracted2, %exit0 ], [ %a.merge_with_extracted1, %exit1 ]
; CHECK-NEXT:    %D = phi i32 [ %b.merge_with_extracted, %exit0 ], [ %c.merge_with_extracted5, %exit1 ]
; CHECK-NEXT:    store i32 %a.merge_with_extracted, ptr %arg, align 4
; CHECK-NEXT:    store i32 %D, ptr %arg, align 4
; CHECK-NEXT:    br label %return
; CHECK-EMPTY:
; CHECK-NEXT:  return:
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }


; CHECK-LABEL: define internal i16 @func.region_start(i1 %c1, i1 %c2, i8 %dest, ptr %a.out, ptr %b.out, ptr %c.out, ptr %B.ce.out) {
; CHECK-NEXT:  newFuncRoot:
; CHECK-NEXT:    br label %region_start
; CHECK-EMPTY:
; CHECK-NEXT:  region_start:
; CHECK-NEXT:    %a = add i32 42, 1
; CHECK-NEXT:    store i32 %a, ptr %a.out, align 4
; CHECK-NEXT:    br i1 %c1, label %exiting0, label %exiting1
; CHECK-EMPTY:
; CHECK-NEXT:  exiting0:
; CHECK-NEXT:    %b = add i32 42, 2
; CHECK-NEXT:    store i32 %b, ptr %b.out, align 4
; CHECK-NEXT:    br i1 %c2, label %exiting0.exit_crit_edge.exitStub, label %exit0.split
; CHECK-EMPTY:
; CHECK-NEXT:  exiting1:
; CHECK-NEXT:    %c = add i32 42, 3
; CHECK-NEXT:    store i32 %c, ptr %c.out, align 4
; CHECK-NEXT:    switch i8 %dest, label %fallback.exitStub [
; CHECK-NEXT:      i8 0, label %exit0.split
; CHECK-NEXT:      i8 1, label %exit1.exitStub
; CHECK-NEXT:      i8 2, label %exit2.exitStub
; CHECK-NEXT:      i8 3, label %exit0.split
; CHECK-NEXT:    ]
; CHECK-EMPTY:
; CHECK-NEXT:  exit0.split:
; CHECK-NEXT:    %B.ce = phi i32 [ %b, %exiting0 ], [ %a, %exiting1 ], [ %a, %exiting1 ]
; CHECK-NEXT:    store i32 %B.ce, ptr %B.ce.out, align 4
; CHECK-NEXT:    br label %exit0.exitStub
; CHECK-EMPTY:
; CHECK-NEXT:  exiting0.exit_crit_edge.exitStub:
; CHECK-NEXT:    ret i16 0
; CHECK-EMPTY:
; CHECK-NEXT:  fallback.exitStub:
; CHECK-NEXT:    ret i16 1
; CHECK-EMPTY:
; CHECK-NEXT:  exit1.exitStub:
; CHECK-NEXT:    ret i16 2
; CHECK-EMPTY:
; CHECK-NEXT:  exit2.exitStub:
; CHECK-NEXT:    ret i16 3
; CHECK-EMPTY:
; CHECK-NEXT:  exit0.exitStub:
; CHECK-NEXT:    ret i16 4
; CHECK-NEXT:  }


define void @func(ptr %arg, i1 %c0, i1 %c1, i1 %c2, i8 %dest) {
entry:
  br i1 %c0, label %region_start, label %exit

region_start:
  %a = add i32 42, 1
  br i1 %c1, label %exiting0, label %exiting1

exiting0:
  %b = add i32 42, 2
  br i1 %c2, label %exit, label %exit0

exiting1:
  %c = add i32 42, 3
  switch i8 %dest, label %fallback [
    i8 0, label %exit0
    i8 1, label %exit1
    i8 2, label %exit2
    i8 3, label %exit0
  ]

fallback:
  unreachable

exit:
  %A = phi i32 [ 42, %entry ], [ %b, %exiting0 ]
  store i32 %A, ptr %arg
  br label %return

exit0:
  %B = phi i32 [ %b, %exiting0 ], [ %a, %exiting1 ] , [ %a, %exiting1 ]
  store i32 %a, ptr %arg
  store i32 %B, ptr %arg
  br label %after

exit1:
  br label %after

exit2:
  %C = phi i32 [ %c, %exiting1 ]
  store i32 %c, ptr %arg
  store i32 %C, ptr %arg
  br label %return

after:
  %D = phi i32 [ %b, %exit0 ], [ %c, %exit1 ]
  store i32 %a, ptr %arg
  store i32 %D, ptr %arg
  br label %return

return:
  ret void
}
