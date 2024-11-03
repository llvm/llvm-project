; REQUIRES: x86-registered-target

;; Ensure we don't incorrectly devirtualization when one vtable contains an
;; alias (i.e. ensure analysis does not improperly ignore this implementation).

;; Test pure ThinLTO

;; Generate unsplit module with summary for ThinLTO index-based WPD.
; RUN: opt -thinlto-bc -o %t1.o %s

;; Check that we have properly recorded the alias in the vtable summary.
; RUN llvm-dis -o - %t1.o | FileCheck %s --check-prefix SUMMARY
; SUMMARY: gv: (name: "_ZTV1D", {{.*}} vTableFuncs: ((virtFunc: ^[[ALIAS:([0-9]+)]], offset: 16))
; SUMMARY: ^[[ALIAS]] = gv: (name: "_ZN1D1mEiAlias"

; RUN: llvm-lto2 run %t1.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -wholeprogramdevirt-print-index-based \
; RUN:   -o %t2 \
; RUN:   -r=%t1.o,test,px \
; RUN:   -r=%t1.o,_ZTV1D,px \
; RUN:   -r=%t1.o,_ZN1D1mEi,px \
; RUN:   -r=%t1.o,_ZN1D1mEiAlias,px \
; RUN:   -r=%t1.o,_ZTV1B,px \
; RUN:   -r=%t1.o,_ZN1B1mEi,px \
; RUN:   2>&1 | FileCheck %s --implicit-check-not {{[Dd]}}evirtualized --allow-empty
; RUN: llvm-dis %t2.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

;; Test hybrid Thin/Regular LTO

;; Generate split module with summary for hybrid Thin/Regular LTO WPD.
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t3.o %s

; RUN: llvm-lto2 run %t3.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -o %t4 \
; RUN:   -r=%t3.o,test,px \
; RUN:   -r=%t3.o,_ZTV1D, \
; RUN:   -r=%t3.o,_ZTV1D,px \
; RUN:   -r=%t3.o,_ZN1D1mEi,px \
; RUN:   -r=%t3.o,_ZN1D1mEiAlias,px \
; RUN:   -r=%t3.o,_ZN1D1mEiAlias, \
; RUN:   -r=%t3.o,_ZTV1B, \
; RUN:   -r=%t3.o,_ZTV1B,px \
; RUN:   -r=%t3.o,_ZN1B1mEi,px \
; RUN:   -r=%t3.o,_ZN1B1mEi, \
; RUN:   2>&1 | FileCheck %s --implicit-check-not {{[Dd]}}evirtualized --allow-empty
; RUN: llvm-dis %t4.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

;; Test Regular LTO

; RUN: opt -o %t5.o %s
; RUN: llvm-lto2 run %t5.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -o %t6 \
; RUN:   -r=%t5.o,test,px \
; RUN:   -r=%t5.o,_ZTV1D,px \
; RUN:   -r=%t5.o,_ZN1D1mEi,px \
; RUN:   -r=%t5.o,_ZN1D1mEiAlias,px \
; RUN:   -r=%t5.o,_ZTV1B,px \
; RUN:   -r=%t5.o,_ZN1B1mEi,px \
; RUN:   2>&1 | FileCheck %s --implicit-check-not {{[Dd]}}evirtualized --allow-empty
; RUN: llvm-dis %t6.0.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.D = type { ptr }
%struct.B = type { %struct.D }

@_ZTV1D = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr undef, ptr @_ZN1D1mEiAlias] }, !type !0
@_ZTV1B = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr undef, ptr @_ZN1B1mEi] }, !type !0, !type !1


define i32 @_ZN1D1mEi(ptr %this, i32 %a) {
   ret i32 0;
}

@_ZN1D1mEiAlias = unnamed_addr alias i32 (ptr, i32), ptr @_ZN1D1mEi

define i32 @_ZN1B1mEi(ptr %this, i32 %a) {
   ret i32 0;
}

; CHECK-IR-LABEL: define i32 @test
define i32 @test(ptr %obj2, i32 %a) {
entry:
  %vtable2 = load ptr, ptr %obj2
  %p2 = call i1 @llvm.type.test(ptr %vtable2, metadata !"_ZTS1D")
  call void @llvm.assume(i1 %p2)

  %fptr33 = load ptr, ptr %vtable2, align 8

  ;; Confirm the call was not devirtualized.
  ;; CHECK-IR: %call4 = tail call i32 %fptr33
  %call4 = tail call i32 %fptr33(ptr nonnull %obj2, i32 %a)
  ret i32 %call4
}
; CHECK-IR-LABEL: ret i32
; CHECK-IR-LABEL: }

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i64 16, !"_ZTS1D"}
!1 = !{i64 16, !"_ZTS1B"}
