; RUN: opt -instcombine -S < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i686-unknown-linux-gnu"

; Check that instcombine doesn't sink dynamic allocas across llvm.stacksave.

; Helper to generate branch conditions.
declare i1 @cond()

declare i32* @use_and_return(i32*)

declare i8* @llvm.stacksave() #0

declare void @llvm.stackrestore(i8*) #0

define void @foo(i32 %x) {
entry:
  %c1 = call i1 @cond()
  br i1 %c1, label %ret, label %nonentry

nonentry:                                         ; preds = %entry
  %argmem = alloca i32, i32 %x, align 4
  %sp = call i8* @llvm.stacksave()
  %c2 = call i1 @cond()
  br i1 %c2, label %ret, label %sinktarget

sinktarget:                                       ; preds = %nonentry
  ; Arrange for there to be a single use of %argmem by returning it.
  %p = call i32* @use_and_return(i32* nonnull %argmem)
  store i32 13, i32* %p, align 4
  call void @llvm.stackrestore(i8* %sp)
  %0 = call i32* @use_and_return(i32* %p)
  br label %ret

ret:                                              ; preds = %sinktarget, %nonentry, %entry
  ret void
}

; CHECK-LABEL: define void @foo(i32 %x)
; CHECK: nonentry:
; CHECK:   %argmem = alloca i32, i32 %x
; CHECK:   %sp = call i8* @llvm.stacksave()
; CHECK:   %c2 = call i1 @cond()
; CHECK:   br i1 %c2, label %ret, label %sinktarget
; CHECK: sinktarget:
; CHECK:   %p = call i32* @use_and_return(i32* nonnull %argmem)
; CHECK:   store i32 13, i32* %p
; CHECK:   call void @llvm.stackrestore(i8* %sp)
; CHECK:   %0 = call i32* @use_and_return(i32* %p)

attributes #0 = { nounwind }
