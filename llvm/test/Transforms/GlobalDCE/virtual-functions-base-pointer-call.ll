; RUN: opt < %s -passes=globaldce -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; struct A {
;   A();
;   virtual int foo(int);
;   virtual int bar(float);
; };
; 
; struct B : A {
;   B();
;   virtual int foo(int);
;   virtual int bar(float);
; };
; 
; A::A() {}
; B::B() {}
; int A::foo(int)   { return 1; }
; int A::bar(float) { return 2; }
; int B::foo(int)   { return 3; }
; int B::bar(float) { return 4; }
; 
; extern "C" int test(A *p, int (A::*q)(int)) { return (p->*q)(42); }

; Member function pointers are tracked by the combination of their object type
; and function type, which must both be compatible. Here, the call is through a
; pointer of type "int (A::*q)(int)", so the call could be dispatched to A::foo
; or B::foo. It can't be dispatched to A::bar or B::bar as the function pointer
; does not match, so those can be removed.

%struct.A = type { i32 (...)** }
%struct.B = type { %struct.A }

; CHECK: @_ZTV1A = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* null, i8* bitcast (i32 (%struct.A*, i32)* @_ZN1A3fooEi to i8*), i8* null] }
@_ZTV1A = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* null, i8* bitcast (i32 (%struct.A*, i32)* @_ZN1A3fooEi to i8*), i8* bitcast (i32 (%struct.A*, float)* @_ZN1A3barEf to i8*)] }, align 8, !type !0, !type !1, !type !2, !vcall_visibility !3
; CHECK: @_ZTV1B = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* null, i8* bitcast (i32 (%struct.B*, i32)* @_ZN1B3fooEi to i8*), i8* null] }
@_ZTV1B = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* null, i8* bitcast (i32 (%struct.B*, i32)* @_ZN1B3fooEi to i8*), i8* bitcast (i32 (%struct.B*, float)* @_ZN1B3barEf to i8*)] }, align 8, !type !0, !type !1, !type !2, !type !4, !type !5, !type !6, !vcall_visibility !3


; CHECK: define internal i32 @_ZN1A3fooEi(
define internal i32 @_ZN1A3fooEi(%struct.A* nocapture readnone %this, i32) unnamed_addr #1 align 2 {
entry:
  ret i32 1
}

; CHECK-NOT: define internal i32 @_ZN1A3barEf(
define internal i32 @_ZN1A3barEf(%struct.A* nocapture readnone %this, float) unnamed_addr #1 align 2 {
entry:
  ret i32 2
}

; CHECK: define internal i32 @_ZN1B3fooEi(
define internal i32 @_ZN1B3fooEi(%struct.B* nocapture readnone %this, i32) unnamed_addr #1 align 2 {
entry:
  ret i32 3
}

; CHECK-NOT: define internal i32 @_ZN1B3barEf(
define internal i32 @_ZN1B3barEf(%struct.B* nocapture readnone %this, float) unnamed_addr #1 align 2 {
entry:
  ret i32 4
}


define hidden void @_ZN1AC2Ev(%struct.A* nocapture %this) {
entry:
  %0 = getelementptr inbounds %struct.A, %struct.A* %this, i64 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTV1A, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8
  ret void
}

define hidden void @_ZN1BC2Ev(%struct.B* nocapture %this) {
entry:
  %0 = getelementptr inbounds %struct.B, %struct.B* %this, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTV1B, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8
  ret void
}

define hidden i32 @test(%struct.A* %p, i64 %q.coerce0, i64 %q.coerce1) {
entry:
  %0 = bitcast %struct.A* %p to i8*
  %1 = getelementptr inbounds i8, i8* %0, i64 %q.coerce1
  %this.adjusted = bitcast i8* %1 to %struct.A*
  %2 = and i64 %q.coerce0, 1
  %memptr.isvirtual = icmp eq i64 %2, 0
  br i1 %memptr.isvirtual, label %memptr.nonvirtual, label %memptr.virtual

memptr.virtual:                                   ; preds = %entry
  %3 = bitcast i8* %1 to i8**
  %vtable = load i8*, i8** %3, align 8
  %4 = add i64 %q.coerce0, -1
  %5 = getelementptr i8, i8* %vtable, i64 %4, !nosanitize !12
  %6 = tail call { i8*, i1 } @llvm.type.checked.load(i8* %5, i32 0, metadata !"_ZTSM1AFiiE.virtual"), !nosanitize !12
  %7 = extractvalue { i8*, i1 } %6, 0, !nosanitize !12
  %memptr.virtualfn = bitcast i8* %7 to i32 (%struct.A*, i32)*, !nosanitize !12
  br label %memptr.end

memptr.nonvirtual:                                ; preds = %entry
  %memptr.nonvirtualfn = inttoptr i64 %q.coerce0 to i32 (%struct.A*, i32)*
  br label %memptr.end

memptr.end:                                       ; preds = %memptr.nonvirtual, %memptr.virtual
  %8 = phi i32 (%struct.A*, i32)* [ %memptr.virtualfn, %memptr.virtual ], [ %memptr.nonvirtualfn, %memptr.nonvirtual ]
  %call = tail call i32 %8(%struct.A* %this.adjusted, i32 42)
  ret i32 %call
}

declare { i8*, i1 } @llvm.type.checked.load(i8*, i32, metadata)

!llvm.module.flags = !{!7}

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTSM1AFiiE.virtual"}
!2 = !{i64 24, !"_ZTSM1AFifE.virtual"}
!3 = !{i64 2}
!4 = !{i64 16, !"_ZTS1B"}
!5 = !{i64 16, !"_ZTSM1BFiiE.virtual"}
!6 = !{i64 24, !"_ZTSM1BFifE.virtual"}
!7 = !{i32 1, !"Virtual Function Elim", i32 1}
!12 = !{}
