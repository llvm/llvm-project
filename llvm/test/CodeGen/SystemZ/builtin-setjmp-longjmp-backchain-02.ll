; Test -mbackchain setjmp  store jmp_buf
; Return address in slot 2.
; Backchain value in slot 3.
; Stack Pointer in slot 4.
; Clobber %r6-%r15, %f8-%f15.

; RUN: llc < %s | FileCheck %s

@buf = dso_local global [20 x ptr] zeroinitializer, align 8
@str = private unnamed_addr constant [9 x i8] c"call foo\00", align 1
@str.2 = private unnamed_addr constant [7 x i8] c"return\00", align 1

; Function Attrs: noreturn nounwind
define dso_local void @foo() local_unnamed_addr #0 {
entry:
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf)
  unreachable
}

; Function Attrs: noreturn nounwind
declare void @llvm.eh.sjlj.longjmp(ptr) #1

; Function Attrs: nounwind
define dso_local noundef signext range(i32 0, 2) i32 @main(i32 noundef signext %argc, ptr nocapture noundef readnone %argv) local_unnamed_addr #2 {
entry:
; CHECK:        stmg    %r6, %r15, 48(%r15)
; CHECK:        lgr     %r1, %r15
; CHECK:        aghi    %r15, -224
; CHECK:        stg     %r1, 0(%r15)
; CHECK:        std     %f8, 216(%r15)                  
; CHECK:        std     %f9, 208(%r15)                 
; CHECK:        std     %f10, 200(%r15)               
; CHECK:        std     %f11, 192(%r15)              
; CHECK:        std     %f12, 184(%r15)             
; CHECK:        std     %f13, 176(%r15)            
; CHECK:        std     %f14, 168(%r15)           
; CHECK:        std     %f15, 160(%r15) 
; CHECK:        larl    %r1, buf
; CHECK:        larl    %r0, .LBB1_2
; CHECK:        stg     %r0, 8(%r1)
; CHECK:        stg     %r15, 24(%r1)
; CHECK:        lg      %r0, 0(%r15)
; CHECK:        stg     %r0, 16(%r1)

  %0 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf)
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %puts2 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.2)
  ret i32 0

if.end:                                           ; preds = %entry
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf)
  unreachable
}

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #3

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #4

attributes #0 = { noreturn nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { noreturn nounwind }
attributes #2 = { nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #3 = { nounwind }
attributes #4 = { nofree nounwind }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
