; Test that unreachable functions are ignored by WPD in hybrid LTO.
; In this test case, the unreachable function is the virtual deleting destructor of an abstract class.

; Generate split module with summary for hybrid Regular LTO WPD
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %s -o %t-main.bc
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %p/Inputs/devirt_hybrid_after_filtering_unreachable_lib.ll -o %t-foo.bc
; Tests that devirtualization happens.
; RUN: llvm-lto2 run -save-temps %t-main.bc %t-foo.bc -pass-remarks=. -o %t \
; RUN:   -whole-program-visibility \
; RUN:   -r=%t-foo.bc,_ZN7Derived1xEv,pl \
; RUN:   -r=%t-foo.bc,printf, \
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
; RUN:   -r=%t-main.bc,_ZN7DerivedC2Ev,pl \
; RUN:   -r=%t-main.bc,_Z3fooP4Base, \
; RUN:   -r=%t-main.bc,_ZN4BaseC2Ev,pl \
; RUN:   -r=%t-main.bc,_ZN4BaseD2Ev, \
; RUN:   -r=%t-main.bc,_ZN4BaseD0Ev, \
; RUN:   -r=%t-main.bc,__cxa_pure_virtual, \
; RUN:   -r=%t-main.bc,printf, \
; RUN:   -r=%t-main.bc,_ZTV7Derived, \
; RUN:   -r=%t-main.bc,_ZTV4Base, \
; RUN:   -r=%t-main.bc,_ZTVN10__cxxabiv117__class_type_infoE, \
; RUN:   -r=%t-main.bc,_ZTS4Base, \
; RUN:   -r=%t-main.bc,_ZTI4Base, \
; RUN:   -r=%t-main.bc,__cxa_pure_virtual, \
; RUN:   -r=%t-main.bc,_ZN4BaseD2Ev, \
; RUN:   -r=%t-main.bc,_ZN4BaseD0Ev, \
; RUN:   -r=%t-main.bc,_ZTV4Base, \
; RUN:   -r=%t-main.bc,_ZTI4Base, 2>&1 | FileCheck %s --check-prefix=REMARK
 
; REMARK-COUNT-1: single-impl: devirtualized a call to _ZN7DerivedD0Ev

; ModuleID = 'tmp.cc'
source_filename = "tmp.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.Derived = type { %class.Base }
%class.Base = type { i32 (...)** }

$_ZN7DerivedC2Ev = comdat any

$_ZN4BaseC2Ev = comdat any

$_ZN4BaseD2Ev = comdat any

$_ZN4BaseD0Ev = comdat any

$_ZTV4Base = comdat any

$_ZTS4Base = comdat any

$_ZTI4Base = comdat any

@_ZTV7Derived = external dso_local unnamed_addr constant { [5 x i8*] }, align 8
@_ZTV4Base = linkonce_odr hidden unnamed_addr constant { [5 x i8*] } { [5 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI4Base to i8*), i8* bitcast (void (%class.Base*)* @_ZN4BaseD2Ev to i8*), i8* bitcast (void (%class.Base*)* @_ZN4BaseD0Ev to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*)] }, comdat, align 8, !type !0, !type !1, !vcall_visibility !2
@_ZTVN10__cxxabiv117__class_type_infoE = external dso_local global i8*
@_ZTS4Base = linkonce_odr hidden constant [6 x i8] c"4Base\00", comdat, align 1
@_ZTI4Base = linkonce_odr hidden constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @_ZTS4Base, i32 0, i32 0) }, comdat, align 8
@.str = private unnamed_addr constant [18 x i8] c"In Base::~Base()\0A\00", align 1

define hidden i32 @main() {
entry:
  %d = alloca %class.Derived*, align 8
  %call = call noalias nonnull i8* @_Znwm(i64 8)
  %0 = bitcast i8* %call to %class.Derived*
  call void @_ZN7DerivedC2Ev(%class.Derived* nonnull align 8 dereferenceable(8) %0)
  store %class.Derived* %0, %class.Derived** %d, align 8
  %1 = load %class.Derived*, %class.Derived** %d, align 8
  %2 = bitcast %class.Derived* %1 to %class.Base*
  call void @_Z3fooP4Base(%class.Base* %2)
  ret i32 0
}

declare dso_local nonnull i8* @_Znwm(i64)

define linkonce_odr hidden void @_ZN7DerivedC2Ev(%class.Derived* nonnull align 8 dereferenceable(8) %this) unnamed_addr comdat align 2 {
entry:
  %this.addr = alloca %class.Derived*, align 8
  store %class.Derived* %this, %class.Derived** %this.addr, align 8
  %this1 = load %class.Derived*, %class.Derived** %this.addr, align 8
  %0 = bitcast %class.Derived* %this1 to %class.Base*
  call void @_ZN4BaseC2Ev(%class.Base* nonnull align 8 dereferenceable(8) %0)
  %1 = bitcast %class.Derived* %this1 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTV7Derived, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)*** %1, align 8
  ret void
}

declare dso_local void @_Z3fooP4Base(%class.Base*)

define linkonce_odr hidden void @_ZN4BaseC2Ev(%class.Base* nonnull align 8 dereferenceable(8) %this) unnamed_addr comdat align 2 {
entry:
  %this.addr = alloca %class.Base*, align 8
  store %class.Base* %this, %class.Base** %this.addr, align 8
  %this1 = load %class.Base*, %class.Base** %this.addr, align 8
  %0 = bitcast %class.Base* %this1 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTV4Base, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)*** %0, align 8
  ret void
}

define linkonce_odr hidden void @_ZN4BaseD2Ev(%class.Base* nonnull align 8 dereferenceable(8) %this) unnamed_addr comdat align 2 {
entry:
  %this.addr = alloca %class.Base*, align 8
  store %class.Base* %this, %class.Base** %this.addr, align 8
  %this1 = load %class.Base*, %class.Base** %this.addr, align 8
  %0 = bitcast %class.Base* %this1 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTV4Base, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)*** %0, align 8
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0))
  ret void
}

define linkonce_odr hidden void @_ZN4BaseD0Ev(%class.Base* nonnull align 8 dereferenceable(8) %this) unnamed_addr comdat align 2 {
entry:
  %this.addr = alloca %class.Base*, align 8
  store %class.Base* %this, %class.Base** %this.addr, align 8
  %this1 = load %class.Base*, %class.Base** %this.addr, align 8
  call void @llvm.trap()
  unreachable
}

declare dso_local void @__cxa_pure_virtual() unnamed_addr

declare dso_local i32 @printf(i8*, ...)

declare void @llvm.trap()

!llvm.module.flags = !{!3, !4, !5, !6}

!0 = !{i64 16, !"_ZTS4Base"}
!1 = !{i64 32, !"_ZTSM4BaseFvvE.virtual"}
!2 = !{i64 1}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 1, !"Virtual Function Elim", i32 0}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{i32 7, !"frame-pointer", i32 2}
