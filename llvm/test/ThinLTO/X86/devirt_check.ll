; REQUIRES: x86-registered-target

; Test that devirtualization option -wholeprogramdevirt-check adds code to check
; that the devirtualization decision was correct and trap if not.

; The vtables have vcall_visibility metadata with hidden visibility, to enable
; devirtualization.

; Generate unsplit module with summary for ThinLTO index-based WPD.
; RUN: opt -thinlto-bc -o %t2.o %s
; RUN: llvm-lto2 run %t2.o -save-temps -use-new-pm -pass-remarks=. \
; RUN:	 -wholeprogramdevirt-check \
; RUN:   -o %t3 \
; RUN:   -r=%t2.o,test,px \
; RUN:   -r=%t2.o,_ZN1A1nEi,p \
; RUN:   -r=%t2.o,_ZN1B1fEi,p \
; RUN:   -r=%t2.o,_ZTV1B,px 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t3.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

; REMARK-DAG: single-impl: devirtualized a call to _ZN1A1nEi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { i32 (...)** }
%struct.B = type { %struct.A }

@_ZTV1B = constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.B*, i32)* @_ZN1B1fEi to i8*), i8* bitcast (i32 (%struct.A*, i32)* @_ZN1A1nEi to i8*)] }, !type !0, !type !1, !vcall_visibility !5


; CHECK-IR-LABEL: define i32 @test
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

  ; Check that the call was devirtualized, but preceeded by a check guarding
  ; a trap if the function pointer doesn't match.
  ; CHECK-IR:   %.not = icmp eq i32 (%struct.A*, i32)* %fptr1, @_ZN1A1nEi
  ; CHECK-IR:   br i1 %.not, label %3, label %2
  ; CHECK-IR: 2:
  ; CHECK-IR:   tail call void @llvm.debugtrap()
  ; CHECK-IR:   br label %3
  ; CHECK-IR: 3:
  ; CHECK-IR:   tail call i32 @_ZN1A1nEi
  %call = tail call i32 %fptr1(%struct.A* nonnull %obj, i32 %a)

  ret i32 %call
}
; CHECK-IR-LABEL:   ret i32
; CHECK-IR-LABEL: }

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)

define i32 @_ZN1B1fEi(%struct.B* %this, i32 %a) #0 {
   ret i32 0;
}

define i32 @_ZN1A1nEi(%struct.A* %this, i32 %a) #0 {
   ret i32 0;
}

; Make sure we don't inline or otherwise optimize out the direct calls.
attributes #0 = { noinline optnone }

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
!3 = !{i64 16, !4}
!4 = distinct !{}
!5 = !{i64 1}
