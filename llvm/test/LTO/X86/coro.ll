; RUN: opt %s -passes='lto-pre-link<O0>' -S -o %t1.ll
; RUN: FileCheck %s --check-prefixes=CHECK,PRELINK --implicit-check-not="call void @llvm.coro" --input-file=%t1.ll
; RUN: llvm-as %t1.ll -o %t1.bc
; RUN: llvm-lto2 run %t1.bc -o %t2.o -r=%t1.bc,test,plx -r=%t1.bc,extern_func,plx -save-temps
; RUN: llvm-dis %t2.o.0.5.precodegen.bc -o - | FileCheck %s --implicit-check-not="call void @llvm.coro"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-fuchsia"

declare void @extern_func()

; CHECK: define{{.*}} void @test(
; PRELINK: call ptr @llvm.coro.subfn.addr
; PRELINK: call ptr @llvm.coro.subfn.addr
define void @test(ptr %hdl) {
  call void @llvm.coro.resume(ptr %hdl)
  call void @llvm.coro.destroy(ptr %hdl)
  call i1 @llvm.coro.done(ptr %hdl)
  ret void
}

declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)
declare i1 @llvm.coro.done(ptr)

