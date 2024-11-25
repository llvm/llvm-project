; Test setjmp  store jmp_buf
; Return address in slot 2.
; Stack Pointer in slot 4. 
; Clobber %r6-%r15, %f8-%f15.

; RUN: llc -O2 < %s | FileCheck %s

@buf = dso_local global [20 x ptr] zeroinitializer, align 8

; Function Attrs: nounwind
define dso_local signext range(i32 0, 2) i32 @main(i32 noundef signext %argc, ptr nocapture noundef readnone %argv) local_unnamed_addr #0 {
entry:
; CHECK:        stmg    %r6, %r15, 48(%r15)
; CHECK:        aghi    %r15, -224
; CHECK:        std     %f8, 216(%r15)
; CHECK:        std     %f9, 208(%r15)
; CHECK:        std     %f10, 200(%r15)
; CHECK:        std     %f11, 192(%r15)
; CHECK:        std     %f12, 184(%r15)
; CHECK:        std     %f13, 176(%r15)
; CHECK:        std     %f14, 168(%r15)
; CHECK:        std     %f15, 160(%r15)
; CHECK:        larl    %r1, buf
; CHECK:        larl    %r0, .LBB0_2
; CHECK:        stg     %r0, 8(%r1)
; CHECK:        stg     %r15, 24(%r1)
  %0 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf)
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %if.end, label %return

if.end:                                           ; preds = %entry
  tail call void @foo() #1
  br label %return

return:                                           ; preds = %entry, %if.end
  %retval.0 = phi i32 [ 1, %if.end ], [ 0, %entry ]
  ret i32 %retval.0
}

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #1

declare void @foo() local_unnamed_addr #2

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { nounwind }
attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
