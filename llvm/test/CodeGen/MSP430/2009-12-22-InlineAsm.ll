; RUN: llc < %s
; PR 5570
; ModuleID = 'test.c'
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8-n8:16"
target triple = "msp430-unknown-unknown"

@buf = common global [10 x i8] zeroinitializer, align 1 ; <[10 x i8]*> [#uses=2]

define i16 @main() noreturn nounwind {
entry:
  %0 = tail call i8* asm "", "=r,0"(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @buf, i16 0, i16 0)) nounwind ; <i8*> [#uses=1]
  %sub.ptr = getelementptr inbounds i8, i8* %0, i16 1 ; <i8*> [#uses=1]
  %sub.ptr.lhs.cast = ptrtoint i8* %sub.ptr to i16 ; <i16> [#uses=1]
  %sub.ptr.sub = sub i16 %sub.ptr.lhs.cast, ptrtoint ([10 x i8]* @buf to i16) ; <i16> [#uses=1]
  %cmp = icmp eq i16 %sub.ptr.sub, 1              ; <i1> [#uses=1]
  br i1 %cmp, label %bar.exit, label %if.then.i

if.then.i:                                        ; preds = %entry
  tail call void @abort() nounwind
  br label %bar.exit

bar.exit:                                         ; preds = %entry, %if.then.i
  tail call void @exit(i16 0) nounwind
  unreachable
}

declare void @exit(i16) noreturn

declare void @abort()
