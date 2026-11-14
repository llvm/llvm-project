; Check that in the ThinLTO backend, indirect call promotion driven by
; instrumentation PGO (!VP metadata) can promote to a function that is a CFI
; jump table member. This requires ICP to run before LowerTypeTests, which
; renames the body of such a function to <name>.cfi.

; REQUIRES: x86-registered-target

; RUN: opt -thinlto-bc -thinlto-split-lto-unit %s -o %t.bc
; RUN: llvm-lto2 run -save-temps %t.bc -o %t.out \
; RUN:   -r=%t.bc,add1,plx \
; RUN:   -r=%t.bc,add2,plx \
; RUN:   -r=%t.bc,table,plx \
; RUN:   -r=%t.bc,call_it,plx
; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s

; Both indirect call targets were promoted. Each comparison is redirected to
; the jump table entry (@add1 / @add2), while the direct calls target the .cfi
; bodies (@add1.cfi is inlined; noinline @add2.cfi remains a direct call).
; CHECK-LABEL: define hidden {{.*}}i32 @call_it(ptr {{.*}}%f, i32 {{.*}}%x)
; CHECK: icmp eq ptr {{.*}}, @add1{{$|,}}
; CHECK: add i32 %x, 1
; CHECK-NOT: call {{.*}}@add1
; CHECK: icmp eq ptr {{.*}}, @add2{{$|,}}
; CHECK: call i32 @add2.cfi(i32 %x)
; CHECK: call i32 %f(i32 %x)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; @add1 and @add2 must be address taken, or they would be dead-stripped and
; never get jump table entries.
@table = hidden constant [2 x ptr] [ptr @add1, ptr @add2]

define hidden i32 @add1(i32 %x) !type !0 {
  %r = add i32 %x, 1
  ret i32 %r
}

define hidden i32 @add2(i32 %x) noinline !type !0 {
  %r = add i32 %x, 2
  ret i32 %r
}

define hidden i32 @call_it(ptr %f, i32 %x) {
  %t = call i1 @llvm.type.test(ptr %f, metadata !"_ZTSFiiE")
  br i1 %t, label %cont, label %trap

trap:
  call void @llvm.ubsantrap(i8 2)
  unreachable

cont:
  %r = call i32 %f(i32 %x), !prof !1
  ret i32 %r
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.ubsantrap(i8)

!0 = !{i64 0, !"_ZTSFiiE"}
!1 = !{!"VP", i32 0, i64 2000, i64 -8030161760475241492, i64 1200, i64 5299563681496682983, i64 800}
