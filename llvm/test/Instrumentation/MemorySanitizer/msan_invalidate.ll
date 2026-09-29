; Regression test for msan not invalidating GlobalsAA.
; RUN: opt < %s -S -passes='require<globals-aa>,module(msan),require<globals-aa>,early-cse<memssa>' 2>&1 | FileCheck %s

target triple = "x86_64-unknown-linux"

define ptr @foo(ptr %p) local_unnamed_addr sanitize_memory {
entry:
  ret ptr %p
}

define i32 @test() local_unnamed_addr sanitize_memory {
entry:
  ; CHECK-LABEL: define i32 @test()

  %x = alloca i32
  store i32 7, ptr %x
  
  ; CHECK: store i64 0, ptr @__msan_retval_tls
  ; CHECK-NEXT: call ptr @foo(

  %call = call ptr @foo(ptr %x)

  ; If GlobalsAA is eliminated correctly, early-cse should not remove next load.
  ; CHECK-NEXT: %[[MSRET:.*]] = load i64, ptr @__msan_retval_tls
  ; CHECK-NEXT: %[[MSCMP:.*]] = icmp ne i64 %[[MSRET]], 0
  ; CHECK-NEXT: br i1 %[[MSCMP]],

  %ret = load i32, ptr %call
  ret i32 %ret
}
