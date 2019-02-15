; RUN: not llvm-as -o /dev/null %s 2>&1 | FileCheck %s
; PR1633

declare void @llvm.gcroot(i8**, i8*)

define void @caller_must_use_gc() {
  ; CHECK: Enclosing function does not use GC.
  ; CHECK-NEXT: call void @llvm.gcroot(i8** %alloca, i8* null)
  %alloca = alloca i8*
  call void @llvm.gcroot(i8** %alloca, i8* null)
  ret void
}

define void @must_be_alloca() gc "test" {
; CHECK: llvm.gcroot parameter #1 must be an alloca.
; CHECK-NEXT: call void @llvm.gcroot(i8** null, i8* null)
  call void @llvm.gcroot(i8** null, i8* null)
  ret void
}

define void @non_ptr_alloca_null() gc "test" {
  ; CHECK: llvm.gcroot parameter #1 must either be a pointer alloca, or argument #2 must be a non-null constant.
  ; CHECK-NEXT: call void @llvm.gcroot(i8** %cast.alloca, i8* null)
  %alloca = alloca i32
  %cast.alloca = bitcast i32* %alloca to i8**
  call void @llvm.gcroot(i8** %cast.alloca, i8* null)
  ret void
}

define void @non_constant_arg1(i8* %arg) gc "test" {
  ; CHECK: llvm.gcroot parameter #2 must be a constant.
  ; CHECK-NEXT: call void @llvm.gcroot(i8** %alloca, i8* %arg)
  %alloca = alloca i8*
  call void @llvm.gcroot(i8** %alloca, i8* %arg)
  ret void
}

define void @non_ptr_alloca_non_null() gc "test" {
; CHECK-NOT: llvm.gcroot parameter
  %alloca = alloca i32
  %cast.alloca = bitcast i32* %alloca to i8**
  call void @llvm.gcroot(i8** %cast.alloca, i8* inttoptr (i64 123 to i8*))
  ret void
}

define void @casted_alloca() gc "test" {
; CHECK-NOT: llvm.gcroot parameter
  %alloca = alloca i32*
  %ptr.cast.alloca = bitcast i32** %alloca to i8**
  call void @llvm.gcroot(i8** %ptr.cast.alloca, i8* null)
  ret void
}
