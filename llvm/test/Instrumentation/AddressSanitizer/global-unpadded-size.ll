; Check that instrumented globals record their pre-padding size, and that the
; attachment survives a bitcode round trip: the sanitizer pass and the consumer
; can be in different processes when LTO is in use.

; RUN: opt < %s -passes=asan -S | FileCheck %s
; RUN: opt < %s -passes=asan | llvm-dis | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@small = global i32 7, align 4
@exact = global [32 x i8] zeroinitializer, align 1
@big = global [256 x i8] zeroinitializer, align 1

; CHECK: @small = global { i32, [28 x i8] } {{.*}}!sanitize.unpadded.size ![[SMALL:[0-9]+]]
; CHECK: @exact = global { [32 x i8], [32 x i8] } {{.*}}!sanitize.unpadded.size ![[EXACT:[0-9]+]]
; CHECK: @big = global { [256 x i8], [64 x i8] } {{.*}}!sanitize.unpadded.size ![[BIG:[0-9]+]]

; CHECK-DAG: ![[SMALL]] = !{i64 4}
; CHECK-DAG: ![[EXACT]] = !{i64 32}
; CHECK-DAG: ![[BIG]] = !{i64 256}
