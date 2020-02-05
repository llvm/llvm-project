; REQUIRES: x86-registered-target

; Test that index-based devirtualization in the presence of external
; vtables with the same name (with comdat) succeeds.

; Generate unsplit module with summary for ThinLTO index-based WPD.
; RUN: opt -thinlto-bc -o %t3.o %s
; RUN: opt -thinlto-bc -o %t4.o %p/Inputs/devirt_external_comdat_same_guid.ll

; RUN: llvm-lto2 run %t3.o %t4.o -save-temps -use-new-pm -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -wholeprogramdevirt-print-index-based \
; RUN:   -o %t5 \
; RUN:   -r=%t3.o,use_B,px \
; RUN:   -r=%t3.o,test,px \
; RUN:   -r=%t3.o,_ZTV1B,px \
; RUN:   -r=%t3.o,_ZN1B1fEi,px \
; RUN:   -r=%t3.o,_ZN1B1nEi,px \
; RUN:   -r=%t4.o,_ZTV1B,x \
; RUN:   -r=%t4.o,_ZN1B1fEi,x \
; RUN:   -r=%t4.o,_ZN1A1nEi,x \
; RUN:   -r=%t4.o,test2,px \
; RUN:   2>&1 | FileCheck %s --check-prefix=REMARK --check-prefix=PRINT
; RUN: llvm-dis %t5.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR1
; RUN: llvm-dis %t5.2.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR2

; PRINT-DAG: Devirtualized call to {{.*}} (_ZN1A1nEi)

; REMARK-DAG: single-impl: devirtualized a call to _ZN1A1nEi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

source_filename = "-"

%struct.A = type { i32 (...)** }
%struct.B = type { %struct.A }

$_ZTV1B = comdat any

@_ZTV1B = constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.B*, i32)* @_ZN1B1fEi to i8*), i8* bitcast (i32 (%struct.B*, i32)* @_ZN1B1nEi to i8*)] }, comdat, !type !0, !type !1

define i32 @_ZN1B1fEi(%struct.B* %this, i32 %a) #0 comdat($_ZTV1B) {
   ret i32 0;
}

define i32 @_ZN1B1nEi(%struct.B* %this, i32 %a) #0 comdat($_ZTV1B) {
   ret i32 0;
}

; Ensures that vtable of B is live so that we will attempt devirt.
define dso_local i32 @use_B(%struct.B* %a) {
entry:
  %0 = bitcast %struct.B* %a to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTV1B, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8
  ret i32 0
}

; CHECK-IR1: define i32 @test(
define i32 @test(%struct.A* %obj, i32 %a) {
entry:
  %0 = bitcast %struct.A* %obj to i8***
  %vtable = load i8**, i8*** %0
  %1 = bitcast i8** %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %1, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr i8*, i8** %vtable, i32 1
  %2 = bitcast i8** %fptrptr to i32 (%struct.A*, i32)**
  %fptr1 = load i32 (%struct.A*, i32)*, i32 (%struct.A*, i32)** %2, align 8

  ; Check that the call was devirtualized.
  ; CHECK-IR1: tail call i32 {{.*}}@_ZN1A1nEi
  %call = tail call i32 %fptr1(%struct.A* nonnull %obj, i32 %a)

  ret i32 %call
}

; CHECK-IR2: define i32 @test2
; Check that the call was devirtualized.
; CHECK-IR2:   tail call i32 @_ZN1A1nEi

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)

attributes #0 = { noinline optnone }

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
