; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL: Function: test1
; CHECK-DAG: MayAlias: i32* %X, i32* %tmp
; CHECK-DAG: MayAlias: i8** %ap, i32* %tmp
; CHECK-DAG: NoAlias: i8** %ap, i8** %aq
; CHECK-DAG: MayAlias: i8** %aq, i32* %tmp

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

  load i32, i32* %X
  load i8*, i8** %ap
  load i8*, i8** %aq
  load i32, i32* %tmp
  ret i32* %tmp
}

declare void @llvm.va_start(i8*)
declare void @llvm.va_copy(i8*, i8*)
declare void @llvm.va_end(i8*)

