; RUN: llc < %s -march=x86-64 | FileCheck %s
; PR7814

@g_16 = global i64 -3738643449681751625, align 8  ; <i64*> [#uses=1]
@g_38 = global i32 0, align 4                     ; <i32*> [#uses=2]
@.str = private constant [4 x i8] c"%d\0A\00"     ; <[4 x i8]*> [#uses=1]

define i32 @main() nounwind {
entry:
  %tmp = load i64* @g_16                          ; <i64> [#uses=1]
  %not.lnot = icmp ne i64 %tmp, 0                 ; <i1> [#uses=1]
  %conv = sext i1 %not.lnot to i64                ; <i64> [#uses=1]
  %and = and i64 %conv, 150                       ; <i64> [#uses=1]
  %conv.i = trunc i64 %and to i8                  ; <i8> [#uses=1]
  %cmp = icmp sgt i8 %conv.i, 0                   ; <i1> [#uses=1]
  br i1 %cmp, label %if.then, label %entry.if.end_crit_edge

; CHECK: andl	$150
; CHECK-NEXT: testb
; CHECK-NEXT: jle

entry.if.end_crit_edge:                           ; preds = %entry
  %tmp4.pre = load i32* @g_38                     ; <i32> [#uses=1]
  br label %if.end

if.then:                                          ; preds = %entry
  store i32 1, i32* @g_38
  br label %if.end

if.end:                                           ; preds = %entry.if.end_crit_edge, %if.then
  %tmp4 = phi i32 [ %tmp4.pre, %entry.if.end_crit_edge ], [ 1, %if.then ] ; <i32> [#uses=1]
  %call5 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i32 %tmp4) nounwind ; <i32> [#uses=0]
  ret i32 0
}

declare i32 @printf(i8* nocapture, ...) nounwind
