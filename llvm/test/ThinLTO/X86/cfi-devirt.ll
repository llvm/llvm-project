; REQUIRES: x86-registered-target

; Test CFI devirtualization through the thin link and backend.

; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t.o %s

; Legacy PM
; FIXME: Fix machine verifier issues and remove -verify-machineinstrs=0. PR39436.
; RUN: llvm-lto2 run %t.o -save-temps -pass-remarks=. \
; RUN:   -verify-machineinstrs=0 \
; RUN:   -o %t3 \
; RUN:   -r=%t.o,test,px \
; RUN:   -r=%t.o,_ZN1A1nEi,p \
; RUN:   -r=%t.o,_ZN1B1fEi,p \
; RUN:   -r=%t.o,_ZN1C1fEi,p \
; RUN:   -r=%t.o,empty,p \
; RUN:   -r=%t.o,_ZTV1B, \
; RUN:   -r=%t.o,_ZTV1C, \
; RUN:   -r=%t.o,_ZN1A1nEi, \
; RUN:   -r=%t.o,_ZN1B1fEi, \
; RUN:   -r=%t.o,_ZN1C1fEi, \
; RUN:   -r=%t.o,_ZTV1B,px \
; RUN:   -r=%t.o,_ZTV1C,px 2>&1 | FileCheck %s --check-prefix=REMARK -dump-input=always
; RUN: llvm-dis %t3.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR -dump-input=always

; New PM
; FIXME: Fix machine verifier issues and remove -verify-machineinstrs=0. PR39436.
; RUN: llvm-lto2 run %t.o -save-temps -use-new-pm -pass-remarks=. \
; RUN:   -verify-machineinstrs=0 \
; RUN:   -o %t3 \
; RUN:   -r=%t.o,test,px \
; RUN:   -r=%t.o,_ZN1A1nEi,p \
; RUN:   -r=%t.o,_ZN1B1fEi,p \
; RUN:   -r=%t.o,_ZN1C1fEi,p \
; RUN:   -r=%t.o,empty,p \
; RUN:   -r=%t.o,_ZTV1B, \
; RUN:   -r=%t.o,_ZTV1C, \
; RUN:   -r=%t.o,_ZN1A1nEi, \
; RUN:   -r=%t.o,_ZN1B1fEi, \
; RUN:   -r=%t.o,_ZN1C1fEi, \
; RUN:   -r=%t.o,_ZTV1B,px \
; RUN:   -r=%t.o,_ZTV1C,px 2>&1 | FileCheck %s --check-prefix=REMARK -dump-input=always
; RUN: llvm-dis %t3.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR -dump-input=always

; REMARK: single-impl: devirtualized a call to _ZN1A1nEi

; Next check that we emit an error when trying to LTO link this module
; containing an llvm.type.checked.load (with a split LTO Unit) with one
; that does not have a split LTO Unit.
; RUN: opt -thinlto-bc -o %t2.o %S/Inputs/empty.ll
; RUN: not llvm-lto2 run %t.o %t2.o -save-temps -pass-remarks=. \
; RUN:   -verify-machineinstrs=0 -thinlto-threads=1 \
; RUN:   -o %t3 \
; RUN:   -r=%t.o,test,px \
; RUN:   -r=%t.o,_ZN1A1nEi,p \
; RUN:   -r=%t.o,_ZN1B1fEi,p \
; RUN:   -r=%t.o,_ZN1C1fEi,p \
; RUN:   -r=%t.o,empty,p \
; RUN:   -r=%t.o,_ZTV1B, \
; RUN:   -r=%t.o,_ZTV1C, \
; RUN:   -r=%t.o,_ZN1A1nEi, \
; RUN:   -r=%t.o,_ZN1B1fEi, \
; RUN:   -r=%t.o,_ZN1C1fEi, \
; RUN:   -r=%t.o,_ZTV1B,px \
; RUN:   -r=%t.o,_ZTV1C,px 2>&1 | FileCheck %s --check-prefix=ERROR -dump-input=always
; ERROR: LLVM ERROR: inconsistent LTO Unit splitting with llvm.type.test or llvm.type.checked.load

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { i32 (...)** }
%struct.B = type { %struct.A }
%struct.C = type { %struct.A }

@_ZTV1B = constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.B*, i32)* @_ZN1B1fEi to i8*), i8* bitcast (i32 (%struct.A*, i32)* @_ZN1A1nEi to i8*)] }, !type !0, !type !1
@_ZTV1C = constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.C*, i32)* @_ZN1C1fEi to i8*), i8* bitcast (i32 (%struct.A*, i32)* @_ZN1A1nEi to i8*)] }, !type !0, !type !2

; Put declaration first to test handling of remarks when the first
; function has no basic blocks.
declare void @empty()

; CHECK-IR-LABEL: define i32 @test
define i32 @test(%struct.A* %obj, i32 %a) {
entry:
  %0 = bitcast %struct.A* %obj to i8**
  %vtable5 = load i8*, i8** %0

  %1 = tail call { i8*, i1 } @llvm.type.checked.load(i8* %vtable5, i32 8, metadata !"_ZTS1A")
  %2 = extractvalue { i8*, i1 } %1, 1
  br i1 %2, label %cont, label %trap

trap:
  tail call void @llvm.trap()
  unreachable

cont:
  %3 = extractvalue { i8*, i1 } %1, 0
  %4 = bitcast i8* %3 to i32 (%struct.A*, i32)*

  ; Check that the call was devirtualized.
  ; CHECK-IR: %call = tail call i32 @_ZN1A1nEi
  %call = tail call i32 %4(%struct.A* nonnull %obj, i32 %a)
  %vtable16 = load i8*, i8** %0
  %5 = tail call { i8*, i1 } @llvm.type.checked.load(i8* %vtable16, i32 0, metadata !"_ZTS1A")
  %6 = extractvalue { i8*, i1 } %5, 1
  br i1 %6, label %cont2, label %trap

cont2:
  %7 = extractvalue { i8*, i1 } %5, 0
  %8 = bitcast i8* %7 to i32 (%struct.A*, i32)*

  ; Check that traps are conditional. Invalid TYPE_ID can cause
  ; unconditional traps.
  ; CHECK-IR: br i1 {{.*}}, label %trap

  ; We still have to call it as virtual.
  ; CHECK-IR: %call3 = tail call i32 %8
  %call3 = tail call i32 %8(%struct.A* nonnull %obj, i32 %call)
  ret i32 %call3
}
; CHECK-IR-LABEL: ret i32
; CHECK-IR-LABEL: }

declare { i8*, i1 } @llvm.type.checked.load(i8*, i32, metadata)
declare void @llvm.trap()

declare i32 @_ZN1B1fEi(%struct.B* %this, i32 %a)
declare i32 @_ZN1A1nEi(%struct.A* %this, i32 %a)
declare i32 @_ZN1C1fEi(%struct.C* %this, i32 %a)

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
!2 = !{i64 16, !"_ZTS1C"}
