; REQUIRES: x86-registered-target

; Test index-based devirtualization when one copy is an alias.

; Generate unsplit module with summary for ThinLTO index-based WPD.
; RUN: opt -thinlto-bc -o %t3.o %s
; RUN: opt -thinlto-bc -o %t4.o %p/Inputs/devirt_alias.ll

; RUN: llvm-lto2 run %t3.o %t4.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -wholeprogramdevirt-print-index-based \
; RUN:   -o %t5 \
; RUN:   -r=%t3.o,test,px \
; RUN:   -r=%t3.o,_ZTV1D, \
; RUN:   -r=%t3.o,_ZN1D1mEi, \
; RUN:   -r=%t4.o,_ZN1D1mEi,p \
; RUN:   -r=%t4.o,_ZTV1D,px \
; RUN:   -r=%t4.o,some_name,px \
; RUN:   2>&1 | FileCheck %s --check-prefix=REMARK --check-prefix=PRINT
; RUN: llvm-dis %t5.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR1

; PRINT-DAG: Devirtualized call to {{.*}} (_ZN1D1mEi)

; REMARK-DAG: single-impl: devirtualized a call to _ZN1D1mEi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.D = type { i32 (...)** }

@_ZTV1D = linkonce_odr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.D*, i32)* @_ZN1D1mEi to i8*)] }, !type !3

; CHECK-IR1-LABEL: define i32 @test
define i32 @test(%struct.D* %obj2, i32 %a) {
entry:
  %0 = bitcast %struct.D* %obj2 to i8***
  %vtable2 = load i8**, i8*** %0
  %1 = bitcast i8** %vtable2 to i8*
  %p2 = call i1 @llvm.type.test(i8* %1, metadata !"_ZTS1D")
  call void @llvm.assume(i1 %p2)

  %2 = bitcast i8** %vtable2 to i32 (%struct.D*, i32)**
  %fptr33 = load i32 (%struct.D*, i32)*, i32 (%struct.D*, i32)** %2, align 8

  ; Check that the call was devirtualized.
  ; CHECK-IR1: %call4 = tail call i32 @_ZN1D1mEi
  %call4 = tail call i32 %fptr33(%struct.D* nonnull %obj2, i32 %a)
  ret i32 %call4
}
; CHECK-IR1-LABEL: ret i32
; CHECK-IR1-LABEL: }

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)
declare i32 @_ZN1D1mEi(%struct.D* %this, i32 %a)

attributes #0 = { noinline optnone }

!3 = !{i64 16, !"_ZTS1D"}
