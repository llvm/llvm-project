; PR44154: LLVM c3b06d0c393e caused the body of @main to be replaced with
; unreachable. Check that we perform the expected calls and optimizations.
;
; RUN: opt -S -O3 -o - %s | FileCheck %s
; CHECK-LABEL: @main
; CHECK:       %0 = icmp slt i32 %argc, 2
; CHECK-NEXT:  br i1 %0, label %done, label %do_work
; CHECK-LABEL: do_work:
; CHECK:       %1 = tail call i32 @compute(
; CHECK-LABEL: done:
; CHECK:       %retval = phi i32 [ 0, %entry ], [ %1, %do_work ]
; CHECK-NEXT:  ret i32 %retval

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main(i32 %argc, i8** %argv) #0 {
entry:
  %0 = getelementptr inbounds i8*, i8** %argv, i32 0
  %ptr = load i8*, i8** %0
  %1 = call i32 @compute(i8* %ptr, i32 %argc)
  %2 = icmp slt i32 %argc, 2
  br i1 %2, label %done, label %do_work

do_work:
  %3 = icmp eq i8* %ptr, null
  br i1 %3, label %null, label %done

null:
  call void @call_if_null(i8* %ptr)
  br label %done

done:
  %retval = phi i32 [0, %entry], [%1, %do_work], [%1, %null]
  ret i32 %retval
}

define i32 @compute(i8* nonnull %ptr, i32 %x) #1 {
  ret i32 %x
}

declare void @call_if_null(i8* %ptr) #0

attributes #0 = { nounwind }
attributes #1 = { noinline nounwind readonly }
