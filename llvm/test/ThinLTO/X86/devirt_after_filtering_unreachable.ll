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
; RUN:   -r=%t-foo.bc,_ZN7Derived1xEv,pl \
; RUN:   -r=%t-foo.bc,puts, \
; RUN:   -r=%t-foo.bc,_Z3fooP4Base,pl \
; RUN:   -r=%t-foo.bc,_ZN7DerivedD2Ev,pl \
; RUN:   -r=%t-foo.bc,_ZN7DerivedD0Ev,pl \
; RUN:   -r=%t-foo.bc,_ZN4BaseD2Ev,pl \
; RUN:   -r=%t-foo.bc,_ZN4BaseD0Ev,pl \
; RUN:   -r=%t-foo.bc,__cxa_pure_virtual, \
; RUN:   -r=%t-foo.bc,_ZdlPv, \
; RUN:   -r=%t-foo.bc,_ZTV7Derived,l \
; RUN:   -r=%t-foo.bc,_ZTVN10__cxxabiv120__si_class_type_infoE, \
; RUN:   -r=%t-foo.bc,_ZTS7Derived,pl \
; RUN:   -r=%t-foo.bc,_ZTVN10__cxxabiv117__class_type_infoE, \
; RUN:   -r=%t-foo.bc,_ZTS4Base,pl \
; RUN:   -r=%t-foo.bc,_ZTI4Base,pl \
; RUN:   -r=%t-foo.bc,_ZTI7Derived,pl \
; RUN:   -r=%t-foo.bc,_ZTV4Base,l \
; RUN:   -r=%t-foo.bc,__cxa_pure_virtual, \
; RUN:   -r=%t-foo.bc,_ZN7Derived1xEv, \
; RUN:   -r=%t-foo.bc,_ZN7DerivedD2Ev, \
; RUN:   -r=%t-foo.bc,_ZN7DerivedD0Ev, \
; RUN:   -r=%t-foo.bc,_ZN4BaseD2Ev, \
; RUN:   -r=%t-foo.bc,_ZN4BaseD0Ev, \
; RUN:   -r=%t-foo.bc,_ZTV7Derived,pl \
; RUN:   -r=%t-foo.bc,_ZTI4Base, \
; RUN:   -r=%t-foo.bc,_ZTI7Derived, \
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
; RUN:   -r=%t4.o,_ZN7Derived1xEv,pl \
; RUN:   -r=%t4.o,puts, \
; RUN:   -r=%t4.o,_Z3fooP4Base,pl \
; RUN:   -r=%t4.o,_ZN7DerivedD2Ev,pl \
; RUN:   -r=%t4.o,_ZN7DerivedD0Ev,pl \
; RUN:   -r=%t4.o,_ZN4BaseD2Ev,pl \
; RUN:   -r=%t4.o,_ZN4BaseD0Ev,pl \
; RUN:   -r=%t4.o,__cxa_pure_virtual, \
; RUN:   -r=%t4.o,_ZdlPv, \
; RUN:   -r=%t4.o,_ZTV7Derived,pl \
; RUN:   -r=%t4.o,_ZTVN10__cxxabiv120__si_class_type_infoE, \
; RUN:   -r=%t4.o,_ZTS7Derived,pl \
; RUN:   -r=%t4.o,_ZTVN10__cxxabiv117__class_type_infoE, \
; RUN:   -r=%t4.o,_ZTS4Base,pl \
; RUN:   -r=%t4.o,_ZTI4Base,pl \
; RUN:   -r=%t4.o,_ZTI7Derived,pl \
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

%class.Derived = type { %class.Base }
%class.Base = type { i32 (...)** }

@_ZTV7Derived = external dso_local unnamed_addr constant { [5 x i8*] }, align 8

define hidden i32 @main() local_unnamed_addr {
entry:
  %call = tail call noalias nonnull dereferenceable(8) i8* @_Znwm(i64 8)
  %0 = bitcast i8* %call to %class.Derived*
  %1 = getelementptr inbounds %class.Derived, %class.Derived* %0, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTV7Derived, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %1, align 8, !tbaa !4
  %2 = getelementptr %class.Derived, %class.Derived* %0, i64 0, i32 0
  tail call void @_Z3fooP4Base(%class.Base* nonnull %2)
  ret i32 0
}

declare dso_local nonnull i8* @_Znwm(i64) local_unnamed_addr

declare dso_local void @_Z3fooP4Base(%class.Base*) local_unnamed_addr

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"Virtual Function Elim", i32 0}
!2 = !{i32 7, !"uwtable", i32 1}
!4 = !{!5, !5, i64 0}
!5 = !{!"vtable pointer", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}