;RUN: opt -gvn -S < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n8:16:32-S64"
target triple = "thumbv7--linux-gnueabi"

%struct.a = type { i16, i16, [1 x %union.a] }
%union.a = type { i32 }

@length = external global [0 x i32], align 4

; Function Attrs: nounwind
define fastcc void @foo(%struct.a* nocapture readonly %x) {
;CHECK-LABEL: foo
entry:
  br label %bb0

bb0:                                      ; preds = %land.lhs.true, %entry
;CHECK: bb0:
  %x.tr = phi %struct.a* [ %x, %entry ], [ null, %land.lhs.true ]
  %code1 = getelementptr inbounds %struct.a, %struct.a* %x.tr, i32 0, i32 0
  %0 = load i16, i16* %code1, align 4
; CHECK: load i32, i32*
  %conv = zext i16 %0 to i32
  switch i32 %conv, label %if.end.50 [
    i32 43, label %cleanup
    i32 52, label %if.then.5
  ]

if.then.5:                                        ; preds = %bb0
  br i1 undef, label %land.lhs.true, label %if.then.26

land.lhs.true:                                    ; preds = %if.then.5
  br i1 undef, label %cleanup, label %bb0

if.then.26:                                       ; preds = %if.then.5
  %x.tr.lcssa163 = phi %struct.a* [ %x.tr, %if.then.5 ]
  br i1 undef, label %cond.end, label %cond.false

cond.false:                                       ; preds = %if.then.26
; CHECK: cond.false:
; CHECK-NOT: load
  %mode = getelementptr inbounds %struct.a, %struct.a* %x.tr.lcssa163, i32 0, i32 1
  %bf.load = load i16, i16* %mode, align 2
  %bf.shl = shl i16 %bf.load, 8
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %if.then.26
  br i1 undef, label %if.then.44, label %cleanup

if.then.44:                                       ; preds = %cond.end
  unreachable

if.end.50:                                        ; preds = %bb0
;%CHECK: if.end.50:
  %conv.lcssa = phi i32 [ %conv, %bb0 ]
  %arrayidx52 = getelementptr inbounds [0 x i32], [0 x i32]* @length, i32 0, i32 %conv.lcssa
  %1 = load i32, i32* %arrayidx52, align 4
  br i1 undef, label %for.body.57, label %cleanup

for.body.57:                                      ; preds = %if.end.50
  %i.2157 = add nsw i32 %1, -1
  unreachable

cleanup:                                          ; preds = %if.end.50, %cond.end, %land.lhs.true, %bb0
  ret void
}

@yy_c_buf_p = external unnamed_addr global i8*, align 4
@dfg_text = external global i8*, align 4

define void @dfg_lex() {
;CHECK-LABEL: dfg_lex
entry:
  br label %while.bodythread-pre-split

while.bodythread-pre-split:                       ; preds = %while.end, %while.end, %entry
  br i1 undef, label %if.then.14, label %if.end.15

if.then.14:                                       ; preds = %while.end, %while.bodythread-pre-split
  %v1 = load i32, i32* bitcast (i8** @dfg_text to i32*), align 4
  %sub.ptr.sub = sub i32 undef, %v1
  br label %if.end.15

if.end.15:                                        ; preds = %if.then.14, %while.bodythread-pre-split
  %v2 = load i8*, i8** @yy_c_buf_p, align 4
  br label %while.cond.16

while.cond.16:                                    ; preds = %while.cond.16, %if.end.15
  br i1 undef, label %while.cond.16, label %while.end

while.end:                                        ; preds = %while.cond.16
  %add.ptr = getelementptr inbounds i8, i8* %v2, i32 undef
  store i8* %add.ptr, i8** @dfg_text, align 4
  %sub.ptr.rhs.cast25 = ptrtoint i8* %add.ptr to i32
  %sub.ptr.sub26 = sub i32 0, %sub.ptr.rhs.cast25
  switch i32 undef, label %sw.default [
    i32 65, label %while.bodythread-pre-split
    i32 3, label %return
    i32 57, label %while.bodythread-pre-split
    i32 60, label %if.then.14
  ]

sw.default:                                       ; preds = %while.end
  unreachable

return:                                           ; preds = %while.end
  ret void
}
