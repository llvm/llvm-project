; RUN: opt < %s -globaldce -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; struct A {
;   A();
;   virtual int foo();
; };
; 
; struct B : A {
;   B();
;   virtual int foo();
; };
; 
; A::A() {}
; B::B() {}
; int A::foo() { return 42; }
; int B::foo() { return 1337; }
; 
; extern "C" int test(B *p) { return p->foo(); }

; The virtual call in test can only be dispatched to B::foo (or a more-derived
; class, if there was one), so A::foo can be removed.

%struct.A = type { i32 (...)** }
%struct.B = type { %struct.A }

; CHECK: @_ZTV1A = internal unnamed_addr constant { [3 x i8*] } zeroinitializer
@_ZTV1A = internal unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* null, i8* bitcast (i32 (%struct.A*)* @_ZN1A3fooEv to i8*)] }, align 8, !type !0, !type !1, !vcall_visibility !2

; CHECK: @_ZTV1B = internal unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* null, i8* bitcast (i32 (%struct.B*)* @_ZN1B3fooEv to i8*)] }
@_ZTV1B = internal unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* null, i8* bitcast (i32 (%struct.B*)* @_ZN1B3fooEv to i8*)] }, align 8, !type !0, !type !1, !type !3, !type !4, !vcall_visibility !2

; CHECK-NOT: define internal i32 @_ZN1A3fooEv(
define internal i32 @_ZN1A3fooEv(%struct.A* nocapture readnone %this) {
entry:
  ret i32 42
}

; CHECK: define internal i32 @_ZN1B3fooEv(
define internal i32 @_ZN1B3fooEv(%struct.B* nocapture readnone %this) {
entry:
  ret i32 1337
}

define hidden void @_ZN1AC2Ev(%struct.A* nocapture %this) {
entry:
  %0 = getelementptr inbounds %struct.A, %struct.A* %this, i64 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV1A, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8
  ret void
}

define hidden void @_ZN1BC2Ev(%struct.B* nocapture %this) {
entry:
  %0 = getelementptr inbounds %struct.B, %struct.B* %this, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV1B, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8
  ret void
}

define hidden i32 @test(%struct.B* %p) {
entry:
  %0 = bitcast %struct.B* %p to i8**
  %vtable1 = load i8*, i8** %0, align 8
  %1 = tail call { i8*, i1 } @llvm.type.checked.load(i8* %vtable1, i32 0, metadata !"_ZTS1B"), !nosanitize !10
  %2 = extractvalue { i8*, i1 } %1, 0, !nosanitize !10
  %3 = bitcast i8* %2 to i32 (%struct.B*)*, !nosanitize !10
  %call = tail call i32 %3(%struct.B* %p)
  ret i32 %call
}

declare { i8*, i1 } @llvm.type.checked.load(i8*, i32, metadata) #2

!llvm.module.flags = !{!5}

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTSM1AFivE.virtual"}
!2 = !{i64 2}
!3 = !{i64 16, !"_ZTS1B"}
!4 = !{i64 16, !"_ZTSM1BFivE.virtual"}
!5 = !{i32 1, !"Virtual Function Elim", i32 1}
!10 = !{}
