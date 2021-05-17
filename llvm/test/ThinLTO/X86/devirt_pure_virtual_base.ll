;; Test that pure virtual base is handled correctly.

;; Index based WPD
;; Generate unsplit module with summary for ThinLTO index-based WPD.
; RUN: opt --thinlto-bc -o %t1a.o %s
; RUN: llvm-lto2 run %t1a.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -o %t3a \
; RUN:   -r=%t1a.o,_start,plx \
; RUN:   -r=%t1a.o,_ZTV1A,p \
; RUN:   -r=%t1a.o,_ZTV1B,p \
; RUN:   -r=%t1a.o,_ZN1A1fEi,p \
; RUN:   -r=%t1a.o,_ZN1A1nEi,p \
; RUN:   -r=%t1a.o,_ZN1B1fEi,p \
; RUN:   -r=%t1a.o,__cxa_pure_virtual, \
; RUN:   2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t3a.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

;; Hybrid WPD
;; Generate split module with summary for hybrid Thin/Regular LTO WPD.
; RUN: opt --thinlto-bc --thinlto-split-lto-unit -o %t1b.o %s
; RUN: llvm-lto2 run %t1b.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -o %t3b \
; RUN:   -r=%t1b.o,_start,plx \
; RUN:   -r=%t1b.o,_ZTV1A, \
; RUN:   -r=%t1b.o,_ZTV1B, \
; RUN:   -r=%t1b.o,__cxa_pure_virtual, \
; RUN:   -r=%t1b.o,_ZN1A1fEi,p \
; RUN:   -r=%t1b.o,_ZN1A1nEi,p \
; RUN:   -r=%t1b.o,_ZN1B1fEi,p \
; RUN:   -r=%t1b.o,_ZTV1A,p \
; RUN:   -r=%t1b.o,_ZTV1B,p \
; RUN:   -r=%t1b.o,_ZN1A1nEi, \
; RUN:   -r=%t1b.o,_ZN1B1fEi, \
; RUN:   -r=%t1b.o,__cxa_pure_virtual, \
; RUN:   2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t3b.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

;; Regular LTO WPD
; RUN: opt -o %t1c.o %s
; RUN: llvm-lto2 run %t1c.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -o %t3c \
; RUN:   -r=%t1c.o,_start,plx \
; RUN:   -r=%t1c.o,_ZTV1A,p \
; RUN:   -r=%t1c.o,_ZTV1B,p \
; RUN:   -r=%t1c.o,_ZN1A1fEi,p \
; RUN:   -r=%t1c.o,_ZN1A1nEi,p \
; RUN:   -r=%t1c.o,_ZN1B1fEi,p \
; RUN:   -r=%t1c.o,__cxa_pure_virtual, \
; RUN:   2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t3c.0.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

; REMARK-DAG: single-impl: devirtualized a call to _ZN1A1nEi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { i32 (...)** }
%struct.B = type { %struct.A }

@_ZTV1A = linkonce_odr unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (void ()* @__cxa_pure_virtual to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*)] }, !type !0, !vcall_visibility !2
@_ZTV1B = linkonce_odr unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.B*, i32)* @_ZN1B1fEi to i8*), i8* bitcast (i32 (%struct.A*, i32)* @_ZN1A1nEi to i8*)] }, !type !0, !type !1, !vcall_visibility !2

;; Prevent the vtables from being dead code eliminated.
@llvm.used = appending global [2 x i8*] [ i8* bitcast ( { [4 x i8*] }* @_ZTV1A to i8*), i8* bitcast ( { [4 x i8*] }* @_ZTV1B to i8*)]

; CHECK-IR-LABEL: define dso_local i32 @_start
define i32 @_start(%struct.A* %obj, i32 %a) {
entry:
  %0 = bitcast %struct.A* %obj to i8***
  %vtable = load i8**, i8*** %0
  %1 = bitcast i8** %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %1, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr i8*, i8** %vtable, i32 1
  %2 = bitcast i8** %fptrptr to i32 (%struct.A*, i32)**
  %fptr1 = load i32 (%struct.A*, i32)*, i32 (%struct.A*, i32)** %2, align 8

  ;; Check that the call was devirtualized.
  ; CHECK-IR: %call = tail call i32 @_ZN1A1nEi
  ; CHECK-NODEVIRT-IR: %call = tail call i32 %fptr1
  %call = tail call i32 %fptr1(%struct.A* nonnull %obj, i32 %a)

  ret i32 %call
}
; CHECK-IR-LABEL: ret i32
; CHECK-IR-NEXT: }

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)
declare void @__cxa_pure_virtual() unnamed_addr

define linkonce_odr i32 @_ZN1A1fEi(%struct.A* %this, i32 %a) #0 {
   ret i32 0
}

define linkonce_odr i32 @_ZN1A1nEi(%struct.A* %this, i32 %a) #0 {
   ret i32 0
}

define linkonce_odr i32 @_ZN1B1fEi(%struct.B* %this, i32 %a) #0 {
   ret i32 0
}

;; Make sure we don't inline or otherwise optimize out the direct calls.
attributes #0 = { noinline optnone }

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
!2 = !{i64 0}
