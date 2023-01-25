; REQUIRES: x86-registered-target

; Test CFI through the thin link and backend.

; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t.o %s

; RUN: llvm-lto2 run -save-temps %t.o \
; RUN:   -o %t3 \
; RUN:   -r=%t.o,test,px \
; RUN:   -r=%t.o,_ZTV1B, \
; RUN:   -r=%t.o,_ZN1B1fEi, \
; RUN:   -r=%t.o,_ZTV1B,px
; RUN: llvm-dis %t3.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.B = type { %struct.A }
%struct.A = type { ptr }

@_ZTV1B = constant { [3 x ptr] } { [3 x ptr] [ptr undef, ptr undef, ptr undef] }, !type !0

; CHECK-IR-LABEL: define void @test
define void @test(ptr %b) {
entry:
  ; Ensure that traps are conditional. Invalid TYPE_ID can cause
  ; unconditional traps.
  ; CHECK-IR: br i1 {{.*}}, label %trap
  %vtable2 = load ptr, ptr %b
  %0 = tail call i1 @llvm.type.test(ptr %vtable2, metadata !"_ZTS1A")
  br i1 %0, label %cont, label %trap

trap:
  tail call void @llvm.trap()
  unreachable

cont:
  ; CHECK-IR-LABEL: ret void
  ret void
}
; CHECK-IR-LABEL: }

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.trap()

declare i32 @_ZN1B1fEi(ptr %this, i32 %a)

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
