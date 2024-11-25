;Test  longjmp load from jmp_buf.
; Frame pointer from Slot 1.
; Jump address from Slot 2.
; Stack Pointer from Slot 4.
; Literal Pool Pointer from Slot 5.

; RUN: llc -O1 < %s | FileCheck %s
; CHECK:        stmg    %r11, %r15, 88(%r15)
; CHECK:       aghi    %r15, -160
; CHECK:        lgrl    %r2, .Lstr@GOT
; CHECK:        brasl   %r14, puts@PLT
; CHECK:        larl    %r1, buf
; CHECK:        lg      %r2, 8(%r1)
; CHECK:        lg      %r11, 0(%r1)
; CHECK:        lg      %r13, 32(%r1)
; CHECK:        lg      %r15, 24(%r1)
; CHECK:        br      %r2

@buf = dso_local global [20 x ptr] zeroinitializer, align 8
@str = private unnamed_addr constant [41 x i8] c"Calling longjmp from inside function foo\00", align 1
@str.6 = private unnamed_addr constant [28 x i8] c"Performing function recover\00", align 1
@str.7 = private unnamed_addr constant [23 x i8] c"setjmp has been called\00", align 1
@str.8 = private unnamed_addr constant [21 x i8] c"Calling function foo\00", align 1
@str.10 = private unnamed_addr constant [24 x i8] c"longjmp has been called\00", align 1

; Function Attrs: noreturn nounwind
define dso_local void @foo() local_unnamed_addr #0 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf)
  unreachable
}

; Function Attrs: noreturn nounwind
declare void @llvm.eh.sjlj.longjmp(ptr) #1

; Function Attrs: nofree nounwind
define dso_local void @recover() local_unnamed_addr #2 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.6)
  ret void
}

; Function Attrs: noreturn nounwind
define dso_local noundef signext i32 @main() local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf)
  %cmp.not = icmp eq i32 %0, 0
  br i1 %cmp.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %puts6 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.10)
  tail call void @recover()
  tail call void @exit(i32 noundef signext 1) #6
  unreachable

if.end:                                           ; preds = %entry
;Test  longjmp store to jmp_buf.
; Frame pointer from Slot 1.
; Jump address from Slot 2.
; Stack Pointer from Slot 4.
; CHECK:        larl    %r1, buf
; CHECK:        larl    %r0, .LBB2_3
; CHECK:        stg     %r0, 8(%r1)
; CHECK:        stg     %r15, 24(%r1)
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  %puts4 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.8)
  %puts.i = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf)
  unreachable
}

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #3

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef signext) local_unnamed_addr #4

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #5

attributes #0 = { noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { noreturn nounwind }
attributes #2 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #3 = { nounwind }
attributes #4 = { nofree noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #5 = { nofree nounwind }
attributes #6 = { cold noreturn nounwind }
