; RUN: opt < %s -passes=globaldce -S | FileCheck %s

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
; extern "C" int test(A *p) { return p->foo(); }

; The virtual call in test could be dispatched to either A::foo or B::foo, so
; both must be retained.

%struct.A = type { ptr }
%struct.B = type { %struct.A }

; CHECK: @_ZTV1A = internal unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1A3fooEv] }
@_ZTV1A = internal unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1A3fooEv] }, align 8, !type !0, !type !1, !vcall_visibility !2

; CHECK: @_ZTV1B = internal unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1B3fooEv] }
@_ZTV1B = internal unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1B3fooEv] }, align 8, !type !0, !type !1, !type !3, !type !4, !vcall_visibility !2

; CHECK: define internal i32 @_ZN1A3fooEv(
define internal i32 @_ZN1A3fooEv(ptr nocapture readnone %this) {
entry:
  ret i32 42
}

; CHECK: define internal i32 @_ZN1B3fooEv(
define internal i32 @_ZN1B3fooEv(ptr nocapture readnone %this) {
entry:
  ret i32 1337
}

define hidden void @_ZN1AC2Ev(ptr nocapture %this) {
entry:
  store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV1A, i64 0, i32 0, i64 2), ptr %this, align 8
  ret void
}

define hidden void @_ZN1BC2Ev(ptr nocapture %this) {
entry:
  store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV1B, i64 0, i32 0, i64 2), ptr %this, align 8
  ret void
}

define hidden i32 @test(ptr %p) {
entry:
  %vtable1 = load ptr, ptr %p, align 8
  %0 = tail call { ptr, i1 } @llvm.type.checked.load(ptr %vtable1, i32 0, metadata !"_ZTS1A"), !nosanitize !10
  %1 = extractvalue { ptr, i1 } %0, 0, !nosanitize !10
  %call = tail call i32 %1(ptr %p)
  ret i32 %call
}

declare { ptr, i1 } @llvm.type.checked.load(ptr, i32, metadata) #2

!llvm.module.flags = !{!5}

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTSM1AFivE.virtual"}
!2 = !{i64 2}
!3 = !{i64 16, !"_ZTS1B"}
!4 = !{i64 16, !"_ZTSM1BFivE.virtual"}
!5 = !{i32 1, !"Virtual Function Elim", i32 1}
!10 = !{}
