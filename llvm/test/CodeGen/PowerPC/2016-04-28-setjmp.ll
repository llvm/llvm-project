; RUN: llc -filetype=obj <%s | llvm-objdump -d - | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

@ptr = common global ptr null, align 8

; Verify there's no junk between these two instructions from misemitted
; EH_SjLj_Setup.

; CHECK: li 3, 1
; CHECK: cmplwi	 3, 0

define void @h() #0 {
  %1 = load ptr, ptr @ptr, align 8
  %2 = tail call i32 @llvm.eh.sjlj.setjmp(ptr %1)
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %0
  tail call void @g()
  br label %6

5:                                                ; preds = %0
  tail call void @f()
  br label %6

6:                                                ; preds = %5, %4
  ret void
}

attributes #0 = { nounwind "frame-pointer"="all" }

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr)

declare void @g()

declare void @f()
