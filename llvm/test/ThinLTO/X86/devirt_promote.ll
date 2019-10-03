; REQUIRES: x86-registered-target

; Test devirtualization requiring promotion of local targets, where the
; promotion is required by one devirtualization and needs to be updated
; for a second devirtualization in the defining module as a post-pass
; update.

; Generate unsplit module with summary for ThinLTO index-based WPD.
; RUN: opt -thinlto-bc -o %t3.o %s
; RUN: opt -thinlto-bc -o %t4.o %p/Inputs/devirt_promote.ll

; RUN: llvm-lto2 run %t3.o %t4.o -save-temps -use-new-pm -pass-remarks=. \
; RUN:   -wholeprogramdevirt-print-index-based \
; RUN:   -o %t5 \
; RUN:   -r=%t3.o,test,px \
; RUN:   -r=%t4.o,_ZN1B1fEi,p \
; RUN:   -r=%t4.o,test2,px \
; RUN:   -r=%t4.o,_ZTV1B,px \
; RUN:   2>&1 | FileCheck %s --check-prefix=REMARK --check-prefix=PRINT
; RUN: llvm-dis %t5.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR1
; RUN: llvm-dis %t5.2.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR2
; RUN: llvm-nm %t5.1 | FileCheck %s --check-prefix=NM-INDEX1
; RUN: llvm-nm %t5.2 | FileCheck %s --check-prefix=NM-INDEX2

; NM-INDEX1: U _ZN1A1nEi.llvm.

; Make sure that not only did _ZN1A1nEi get promoted (due to the
; devirtualization in the other module) but the reference due to the
; devirtualization in its defining module should be to the promoted
; symbol.
; NM-INDEX2-NOT: U _ZN1A1nEi
; NM-INDEX2: T _ZN1A1nEi.llvm.
; NM-INDEX2-NOT: U _ZN1A1nEi

; We should devirt call to _ZN1A1nEi once in importing module and once
; in original (exporting) module.
; REMARK-COUNT-2: single-impl: devirtualized a call to _ZN1A1nEi.llvm.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { i32 (...)** }

; CHECK-IR1-LABEL: define i32 @test
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
  ; CHECK-IR1: %call = tail call i32 bitcast (void ()* @_ZN1A1nEi
  %call = tail call i32 %fptr1(%struct.A* nonnull %obj, i32 %a)

  ret i32 %call
}
; CHECK-IR1-LABEL: ret i32
; CHECK-IR1-LABEL: }

; CHECK-IR2: define i32 @test2
; Check that the call was devirtualized.
; CHECK-IR2:   %call4 = tail call i32 @_ZN1A1nEi

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)

attributes #0 = { noinline optnone }
