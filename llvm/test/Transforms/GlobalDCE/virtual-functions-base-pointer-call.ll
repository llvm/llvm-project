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

%struct.A = type { ptr }
%struct.B = type { %struct.A }

; CHECK: @_ZTV1A = internal unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZN1A3fooEi, ptr null] }
@_ZTV1A = internal unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZN1A3fooEi, ptr @_ZN1A3barEf] }, align 8, !type !0, !type !1, !type !2, !vcall_visibility !3
; CHECK: @_ZTV1B = internal unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZN1B3fooEi, ptr null] }
@_ZTV1B = internal unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZN1B3fooEi, ptr @_ZN1B3barEf] }, align 8, !type !0, !type !1, !type !2, !type !4, !type !5, !type !6, !vcall_visibility !3


; CHECK: define internal i32 @_ZN1A3fooEi(
define internal i32 @_ZN1A3fooEi(ptr nocapture readnone %this, i32) unnamed_addr #1 align 2 {
entry:
  ret i32 1
}

; CHECK-NOT: define internal i32 @_ZN1A3barEf(
define internal i32 @_ZN1A3barEf(ptr nocapture readnone %this, float) unnamed_addr #1 align 2 {
entry:
  ret i32 2
}

; CHECK: define internal i32 @_ZN1B3fooEi(
define internal i32 @_ZN1B3fooEi(ptr nocapture readnone %this, i32) unnamed_addr #1 align 2 {
entry:
  ret i32 3
}

; CHECK-NOT: define internal i32 @_ZN1B3barEf(
define internal i32 @_ZN1B3barEf(ptr nocapture readnone %this, float) unnamed_addr #1 align 2 {
entry:
  ret i32 4
}


define hidden void @_ZN1AC2Ev(ptr nocapture %this) {
entry:
  store ptr getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV1A, i64 0, inrange i32 0, i64 2), ptr %this, align 8
  ret void
}

define hidden void @_ZN1BC2Ev(ptr nocapture %this) {
entry:
  store ptr getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV1B, i64 0, inrange i32 0, i64 2), ptr %this, align 8
  ret void
}

define hidden i32 @test(ptr %p, i64 %q.coerce0, i64 %q.coerce1) {
entry:
  %0 = getelementptr inbounds i8, ptr %p, i64 %q.coerce1
  %1 = and i64 %q.coerce0, 1
  %memptr.isvirtual = icmp eq i64 %1, 0
  br i1 %memptr.isvirtual, label %memptr.nonvirtual, label %memptr.virtual

memptr.virtual:                                   ; preds = %entry
  %vtable = load ptr, ptr %0, align 8
  %2 = add i64 %q.coerce0, -1
  %3 = getelementptr i8, ptr %vtable, i64 %2, !nosanitize !12
  %4 = tail call { ptr, i1 } @llvm.type.checked.load(ptr %3, i32 0, metadata !"_ZTSM1AFiiE.virtual"), !nosanitize !12
  %5 = extractvalue { ptr, i1 } %4, 0, !nosanitize !12
  br label %memptr.end

memptr.nonvirtual:                                ; preds = %entry
  %memptr.nonvirtualfn = inttoptr i64 %q.coerce0 to ptr
  br label %memptr.end

memptr.end:                                       ; preds = %memptr.nonvirtual, %memptr.virtual
  %6 = phi ptr [ %5, %memptr.virtual ], [ %memptr.nonvirtualfn, %memptr.nonvirtual ]
  %call = tail call i32 %6(ptr %0, i32 42)
  ret i32 %call
}

declare { ptr, i1 } @llvm.type.checked.load(ptr, i32, metadata)

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
