; RUN: llc < %s -mtriple=x86_64-netbsd -use-ctors | FileCheck %s

; Check that our compiler never emits global constructors
; inside the .init_array section when building for a non supported target.
; Because of this, the test depends on UseInitArray behavior under NetBSD
; as found in Generic_ELF::addClangTargetOptions().

; This is to workaround a Visual Studio bug which causes field
; UseInitArray to be left uninitialized instead of being 
; zero-initialized (as specified in [dcl.init]p7).
; This workaround consists in providing a user default constructor
; that explicitly initializes field UseInitArray.

%class.C = type { i8 }
%class.D = type { i8 }

@c1 = global %class.C zeroinitializer, align 1
@d1 = global %class.D zeroinitializer, align 1
@llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 101, void ()* @_GLOBAL__I_000101, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I_a, i8* null }]

define linkonce_odr void @_ZN1CC1Ev(%class.C* nocapture %this) {
entry:
  ret void
}

define linkonce_odr void @_ZN1DC1Ev(%class.D* nocapture %this) {
entry:
  ret void
}

define linkonce_odr void @_ZN1DC2Ev(%class.D* nocapture %this) {
entry:
  ret void
}

define linkonce_odr void @_ZN1CC2Ev(%class.C* nocapture %this) {
entry:
  ret void
}

define internal void @_GLOBAL__I_000101() nounwind readnone {
entry:
  ret void
}

define internal void @_GLOBAL__I_a() nounwind readnone {
entry:
  ret void
}

; CHECK-NOT: .init_array
