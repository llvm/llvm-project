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
%struct.A = type { i32 (...)** }

@_ZTV1B = constant { [3 x i8*] } { [3 x i8*] [i8* undef, i8* undef, i8* undef] }, !type !0

; CHECK-IR-LABEL: define void @test
define void @test(i8* %b) {
entry:
  ; Ensure that traps are conditional. Invalid TYPE_ID can cause
  ; unconditional traps.
  ; CHECK-IR: br i1 {{.*}}, label %trap
  %0 = bitcast i8* %b to i8**
  %vtable2 = load i8*, i8** %0
  %1 = tail call i1 @llvm.type.test(i8* %vtable2, metadata !"_ZTS1A")
  br i1 %1, label %cont, label %trap

trap:
  tail call void @llvm.trap()
  unreachable

cont:
  ; CHECK-IR-LABEL: ret void
  ret void
}
; CHECK-IR-LABEL: }

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.trap()

declare i32 @_ZN1B1fEi(%struct.B* %this, i32 %a)

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
