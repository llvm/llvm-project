; Test that unreachable functions are ignored by WPD in hybrid LTO, and thin LTO.
; In this test case, the unreachable function is the virtual deleting destructor of an abstract class.

; Generate split module with summary for hybrid Regular LTO WPD
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %s -o %t-main.bc
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %p/Inputs/devirt_after_filtering_unreachable_lib.ll -o %t-foo.bc

; Check that deleting destructor of pure virtual class is unreachable.

; Check that deleting destructor of pure virtual class is unreachable.
; RUN: llvm-modextract -b -n=0 %t-foo.bc -o %t-foo.bc.0
; RUN: llvm-dis -o - %t-foo.bc.0 | FileCheck %s --check-prefix=UNREACHABLEFLAG

; Tests that devirtualization happens.
; RUN: llvm-lto2 run -save-temps %t-main.bc %t-foo.bc -pass-remarks=. -o %t \
; RUN:   -whole-program-visibility \
; RUN:   -r=%t-foo.bc,_Z3fooP4Base,pl \
; RUN:   -r=%t-foo.bc,_ZN7DerivedD0Ev,pl \
; RUN:   -r=%t-foo.bc,_ZN4BaseD0Ev,pl \
; RUN:   -r=%t-foo.bc,_ZTV7Derived,l \
; RUN:   -r=%t-foo.bc,_ZTV4Base,l \
; RUN:   -r=%t-foo.bc,_ZN7DerivedD0Ev, \
; RUN:   -r=%t-foo.bc,_ZN4BaseD0Ev, \
; RUN:   -r=%t-foo.bc,_ZTV7Derived,pl \
; RUN:   -r=%t-foo.bc,_ZTV4Base,pl \
; RUN:   -r=%t-main.bc,main,plx \
; RUN:   -r=%t-main.bc,_Znwm,pl \
; RUN:   -r=%t-main.bc,_Z3fooP4Base, \
; RUN:   -r=%t-main.bc,_ZTV7Derived, 2>&1 | FileCheck %s --check-prefix=REMARK
 
; REMARK-COUNT-1: single-impl: devirtualized a call to _ZN7DerivedD0Ev

; Generate unsplit module with summary for ThinLTO index-based WPD.
; RUN: opt -thinlto-bc -o %t3.o %s

; Check that deleting destructor of pure virtual class is unreachable.
; RUN: opt -thinlto-bc -o %t4.o %p/Inputs/devirt_after_filtering_unreachable_lib.ll
; RUN: llvm-dis -o - %t4.o | FileCheck %s --check-prefix=UNREACHABLEFLAG

; UNREACHABLEFLAG: gv: (name: "_ZN4BaseD0Ev", {{.*}}, funcFlags: ({{.*}} mustBeUnreachable: 1

; Test that devirtualized happen in index based WPD
; RUN: llvm-lto2 run %t4.o %t3.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -wholeprogramdevirt-print-index-based \
; RUN:   -o %t5 \
; RUN:   -r=%t4.o,_Z3fooP4Base,pl \
; RUN:   -r=%t4.o,_ZN7DerivedD0Ev,pl \
; RUN:   -r=%t4.o,_ZN4BaseD0Ev,pl \
; RUN:   -r=%t4.o,_ZTV7Derived,pl \
; RUN:   -r=%t4.o,_ZTV4Base,pl \
; RUN:   -r=%t3.o,main,plx \
; RUN:   -r=%t3.o,_Znwm, \
; RUN:   -r=%t3.o,_Z3fooP4Base, \
; RUN:   -r=%t3.o,_ZTV7Derived,  2>&1 | FileCheck %s --check-prefix=THINREMARK

; THINREMARK: Devirtualized call to {{.*}} (_ZN7DerivedD0Ev)
; THINREMARK: single-impl: devirtualized a call to _ZN7DerivedD0Ev
; THINREMARK: single-impl: devirtualized a call to _ZN7DerivedD0Ev

; ModuleID = 'tmp.cc'
source_filename = "tmp.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Derived = type { %Base }
%Base = type { ptr }

@_ZTV7Derived = external constant { [5 x ptr] }

define hidden i32 @main() {
entry:
  %call = tail call ptr @_Znwm(i64 8)
  store ptr getelementptr inbounds ({ [5 x ptr] }, ptr @_ZTV7Derived, i64 0, i32 0, i64 2), ptr %call
  tail call void @_Z3fooP4Base(ptr nonnull %call)
  ret i32 0
}

declare ptr @_Znwm(i64)

declare void @_Z3fooP4Base(ptr)
