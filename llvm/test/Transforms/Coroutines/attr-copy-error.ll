; XFAIL: *
; RUN: opt -passes='cgscc(coro-split)' -S %s | FileCheck %s

target triple = "riscv64-unknown-linux-gnu"

;; CoroSplit currently copies the ramp function's complete function attribute
;; list to generated resume/destroy functions. That is wrong for
;; cfi_type_hash: their ABI signature differs from the ramp's signature.
;; If CoroSplit is ever changed to stop doing that, the generated functions 
;; will get their own, different attribute group and the test will XPASS.
;; manually run opt -passes='cgscc(coro-split)' -S llvm/test/Transforms/Coroutines/attr-copy-error.ll \
;; | ./build/bin/FileCheck llvm/test/Transforms/Coroutines/attr-copy-error.ll
;; to confirm the failure is on RAMPATTR being equal for all checks.
;;
;; XFAIL until a general policy exists for target-specific callable-ABI
;; attributes copied by IR transforms.

;; 308871 is arbitrarily chosen, it is not the ramp function's actual cfi_type_hash.
define ptr @foo() presplitcoroutine "cfi_type_hash"="308871" {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @foo, ptr null)
  %need.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.alloc, label %dyn.alloc, label %begin

dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  br label %begin

begin:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %phi)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8 @llvm.coro.suspend(token, i1)
declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)

declare noalias ptr @malloc(i32)
declare void @free(ptr)

; CHECK-LABEL: define ptr @foo()
; CHECK-SAME: #[[RAMPATTR:[0-9]+]] {

; CHECK-LABEL: define internal{{.*}} @foo.resume(
; CHECK-NOT: #[[RAMPATTR]]
; CHECK: entry.resume:

; CHECK-LABEL: define internal{{.*}} @foo.destroy(
; CHECK-NOT: #[[RAMPATTR]]
; CHECK: entry.destroy:

; CHECK-LABEL: define internal{{.*}} @foo.cleanup(
; CHECK-NOT: #[[RAMPATTR]]
; CHECK: entry.cleanup: