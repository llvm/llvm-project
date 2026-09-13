; RUN: opt < %s -passes='module(coro-early),cgscc(coro-split),module(coro-cleanup)' -S | FileCheck %s
; RUN: opt < %s -passes='default<O2>' -S | FileCheck --check-prefix=CHECK-O2 %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare token @llvm.coro.id.retcon(i32, i32, ptr, ptr, ptr, ptr)
declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)

declare noalias ptr @malloc(i64)
declare void @free(ptr)

define ptr @prototype(ptr dereferenceable(8) %buf, i1 %unwind) {
  ret ptr null
}

define void @prototype_once(ptr dereferenceable(8) %buf, i1 %unwind) {
  ret void
}

; CHECK-LABEL: define ptr @f_retcon(ptr dereferenceable(8) %buf)
; CHECK-NEXT:  entry:
; CHECK:         %val_ptr = getelementptr inbounds i32, ptr %buf, i64 1
; CHECK-NEXT:    store i32 42, ptr %val_ptr, align 4
; CHECK-NEXT:    %val = load i32, ptr %val_ptr, align 4
; CHECK-NEXT:    ret ptr null

; CHECK-O2-LABEL: define ptr @f_retcon(ptr dereferenceable(8) %buf)
; CHECK-O2-NEXT:  entry:
; CHECK-O2-NEXT:    %val_ptr = getelementptr inbounds {{.*}} ptr %buf, i64 4
; CHECK-O2-NEXT:    store i32 42, ptr %val_ptr, align 4
; CHECK-O2-NEXT:    ret ptr null

define ptr @f_retcon(ptr dereferenceable(8) %buf) {
entry:
  ; We can set size to 0 and align to 1 because this coroutine has 0 suspends,
  ; meaning it will always be elided and requires no frame storage.
  %id = call token @llvm.coro.id.retcon(i32 0, i32 1, ptr %buf, ptr @prototype, ptr @malloc, ptr @free)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %buf)
  
  ; Access the frame (buffer) via handle.
  %val_ptr = getelementptr inbounds i32, ptr %hdl, i64 1
  store i32 42, ptr %val_ptr, align 4
  %val = load i32, ptr %val_ptr, align 4
  
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret ptr null
}

; CHECK-LABEL: define ptr @f_retcon_once(ptr dereferenceable(8) %buf)
; CHECK-NEXT:  entry:
; CHECK:         %val_ptr = getelementptr inbounds i32, ptr %buf, i64 1
; CHECK-NEXT:    store i32 42, ptr %val_ptr, align 4
; CHECK-NEXT:    %val = load i32, ptr %val_ptr, align 4
; CHECK-NEXT:    ret ptr null

; CHECK-O2-LABEL: define ptr @f_retcon_once(ptr dereferenceable(8) %buf)
; CHECK-O2-NEXT:  entry:
; CHECK-O2-NEXT:    %val_ptr = getelementptr inbounds {{.*}} ptr %buf, i64 4
; CHECK-O2-NEXT:    store i32 42, ptr %val_ptr, align 4
; CHECK-O2-NEXT:    ret ptr null

define ptr @f_retcon_once(ptr dereferenceable(8) %buf) {
entry:
  ; We can set size to 0 and align to 1 because this coroutine has 0 suspends,
  ; meaning it will always be elided and requires no frame storage.
  %id = call token @llvm.coro.id.retcon.once(i32 0, i32 1, ptr %buf, ptr @prototype_once, ptr @malloc, ptr @free)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %buf)
  
  ; Access the frame (buffer) via handle.
  %val_ptr = getelementptr inbounds i32, ptr %hdl, i64 1
  store i32 42, ptr %val_ptr, align 4
  %val = load i32, ptr %val_ptr, align 4
  
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret ptr null
}
