; RUN: llc -filetype=obj <%s | llvm-objdump -d - | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

@ptr = common global ptr null, align 8

; Verify there's no junk between these two instructions from misemitted
; EH_SjLj_Setup.

; CHECK: li 3, 1
; CHECK: cmplwi	 3, 0

define void @h() nounwind {
  %1 = load ptr, ptr @ptr, align 8
  %2 = tail call ptr @llvm.frameaddress(i32 0)
  store ptr %2, ptr %1, align 8
  %3 = tail call ptr @llvm.stacksave()
  %4 = getelementptr inbounds ptr, ptr %1, i64 2
  store ptr %3, ptr %4, align 8
  %5 = tail call i32 @llvm.eh.sjlj.setjmp(ptr %1)
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %8, label %7

; <label>:8:                                      ; preds = %0
  tail call void @g()
  br label %9

; <label>:9:                                      ; preds = %0
  tail call void @f()
  br label %9

; <label>:10:                                     ; preds = %8, %7
  ret void
}

; Function Attrs: nounwind readnone
declare ptr @llvm.frameaddress(i32)

; Function Attrs: nounwind
declare ptr @llvm.stacksave()

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr)

declare void @g()

declare void @f()
