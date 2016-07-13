; RUN: opt < %s -disable-basicaa -cfl-steens-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL: Function: test1
; CHECK: MayAlias: i32* %X, i32* %tmp
; CHECK: MayAlias: i32* %tmp, i8** %ap
; CHECK: NoAlias: i8** %ap, i8** %aq
; CHECK: MayAlias: i32* %tmp, i8** %aq

define i32* @test1(i32* %X, ...) {
  ; Initialize variable argument processing
  %ap = alloca i8*
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap2)

  ; Read a single pointer argument
  %tmp = va_arg i8** %ap, i32*

  ; Demonstrate usage of llvm.va_copy and llvm.va_end
  %aq = alloca i8*
  %aq2 = bitcast i8** %aq to i8*
  call void @llvm.va_copy(i8* %aq2, i8* %ap2)
  call void @llvm.va_end(i8* %aq2)

  ; Stop processing of arguments.
  call void @llvm.va_end(i8* %ap2)
  ret i32* %tmp
}

declare void @llvm.va_start(i8*)
declare void @llvm.va_copy(i8*, i8*)
declare void @llvm.va_end(i8*)

