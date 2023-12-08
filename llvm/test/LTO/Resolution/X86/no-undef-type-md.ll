; RUN: opt <%s -o %t0.o -thinlto-bc -thinlto-split-lto-unit
; RUN: llvm-as -o %t1.o %S/Inputs/no-undef-type-md.ll
; RUN: llvm-lto2 run -o %t-obj.o \
; RUN: %t0.o \
; RUN: -r=%t0.o,a, \
; RUN: -r=%t0.o,b,pl \
; RUN: %t1.o \
; RUN: -r=%t1.o,a,pl \
; RUN: | FileCheck --allow-empty --check-prefix=ERROR %s
; RUN: llvm-nm %t-obj.o.0 %t-obj.o.1 -S | FileCheck %s

; ERROR-NOT: expected a Function or null
; ERROR-NOT: ptr undef

; CHECK: -obj.o.0:
; CHECK: -obj.o.1:

; ModuleID = 'test.cpp.o'
source_filename = "test.cpp"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @a() {
entry:
  ret i32 0
}

define i32 @b() {
entry:
  ret i32 0
}

!llvm.module.flags = !{!39}

!39 = !{i32 5, !"CG Profile", !40}
!40 = !{!41}
!41 = !{ptr @b, ptr @a, i64 2594092}
