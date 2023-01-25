; REQUIRES: x86-registered-target

; Test CFI devirtualization through the thin link and backend.

; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t.o %s

; RUN: llvm-lto2 run %t.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
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
; RUN:   -r=%t.o,_ZTV1C,px 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t3.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

; REMARK: single-impl: devirtualized a call to _ZN1A1nEi

; Next check that we emit an error when trying to LTO link this module
; containing an llvm.type.checked.load (with a split LTO Unit) with one
; that does not have a split LTO Unit. Use -thinlto-distributed-indexes
; to ensure it is being caught in the thin link.
; RUN: opt -thinlto-bc -o %t2.o %S/Inputs/empty.ll
; RUN: not llvm-lto2 run %t.o %t2.o -thinlto-distributed-indexes \
; RUN:   -whole-program-visibility \
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
; RUN:   -r=%t.o,_ZTV1C,px 2>&1 | FileCheck %s --check-prefix=ERROR
; ERROR: failed: inconsistent LTO Unit splitting (recompile with -fsplit-lto-unit)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { ptr }
%struct.B = type { %struct.A }
%struct.C = type { %struct.A }

@_ZTV1B = constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr undef, ptr @_ZN1B1fEi, ptr @_ZN1A1nEi] }, !type !0, !type !1
@_ZTV1C = constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr undef, ptr @_ZN1C1fEi, ptr @_ZN1A1nEi] }, !type !0, !type !2

; Put declaration first to test handling of remarks when the first
; function has no basic blocks.
declare void @empty()

; CHECK-IR-LABEL: define i32 @test
define i32 @test(ptr %obj, i32 %a) {
entry:
  %vtable5 = load ptr, ptr %obj

  %0 = tail call { ptr, i1 } @llvm.type.checked.load(ptr %vtable5, i32 8, metadata !"_ZTS1A")
  %1 = extractvalue { ptr, i1 } %0, 1
  br i1 %1, label %cont, label %trap

trap:
  tail call void @llvm.trap()
  unreachable

cont:
  %2 = extractvalue { ptr, i1 } %0, 0

  ; Check that the call was devirtualized.
  ; CHECK-IR: %call = tail call i32 @_ZN1A1nEi
  %call = tail call i32 %2(ptr nonnull %obj, i32 %a)
  %vtable16 = load ptr, ptr %obj
  %3 = tail call { ptr, i1 } @llvm.type.checked.load(ptr %vtable16, i32 0, metadata !"_ZTS1A")
  %4 = extractvalue { ptr, i1 } %3, 1
  br i1 %4, label %cont2, label %trap

cont2:
  %5 = extractvalue { ptr, i1 } %3, 0

  ; Check that traps are conditional. Invalid TYPE_ID can cause
  ; unconditional traps.
  ; CHECK-IR: br i1 {{.*}}, label %trap, label %cont2

  ; We still have to call it as virtual.
  ; CHECK-IR: %call3 = tail call i32 %5
  %call3 = tail call i32 %5(ptr nonnull %obj, i32 %call)
  ret i32 %call3
}
; CHECK-IR-LABEL: ret i32
; CHECK-IR-LABEL: }

declare { ptr, i1 } @llvm.type.checked.load(ptr, i32, metadata)
declare void @llvm.trap()

declare i32 @_ZN1B1fEi(ptr %this, i32 %a)
declare i32 @_ZN1A1nEi(ptr %this, i32 %a)
declare i32 @_ZN1C1fEi(ptr %this, i32 %a)

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
!2 = !{i64 16, !"_ZTS1C"}
