; RUN: llc -filetype=obj <%s | llvm-objdump -d - | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

@ptr = common global i8* null, align 8

; Verify there's no junk between these two instructions from misemitted
; EH_SjLj_Setup.

; CHECK: li 3, 1
; CHECK: cmplwi	 3, 0

define void @h() nounwind {
  %1 = load i8**, i8*** bitcast (i8** @ptr to i8***), align 8
  %2 = tail call i8* @llvm.frameaddress(i32 0)
  store i8* %2, i8** %1, align 8
  %3 = tail call i8* @llvm.stacksave()
  %4 = getelementptr inbounds i8*, i8** %1, i64 2
  store i8* %3, i8** %4, align 8
  %5 = bitcast i8** %1 to i8*
  %6 = tail call i32 @llvm.eh.sjlj.setjmp(i8* %5)
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %9, label %8

; <label>:8:                                      ; preds = %0
  tail call void @g()
  br label %10

; <label>:9:                                      ; preds = %0
  tail call void @f()
  br label %10

; <label>:10:                                     ; preds = %9, %8
  ret void
}

; Function Attrs: nounwind readnone
declare i8* @llvm.frameaddress(i32)

; Function Attrs: nounwind
declare i8* @llvm.stacksave()

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(i8*)

declare void @g()

declare void @f()
