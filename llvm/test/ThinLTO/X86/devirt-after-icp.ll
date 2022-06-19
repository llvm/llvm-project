; REQUIRES: x86-registered-target

; Test devirtualization through the thin link and backend, ensuring that
; it is only applied when the type test corresponding to a devirtualization
; dominates an indirect call using the same vtable pointer. Indirect
; call promotion and inlining may introduce a guarded indirect call
; that can be promoted, which uses the same vtable address as the fallback
; indirect call that cannot be devirtualized.

; The code below illustrates the structure when we started with code like:
;
; class A {
;  public:
;   virtual int foo() { return 1; }
;   virtual int bar() { return 1; }
; };
; class B : public A {
;  public:
;   virtual int foo();
;   virtual int bar();
; };
;
; int foo(A *a) {
;   return a->foo(); // ICP profile says most calls are to B::foo()
; }
;
; int B::foo() {
;   return bar();
; }
;
; After the compile step, which will perform ICP and a round of inlining, we
; have something like:
; int foo(A *a) {
;   if (&a->foo() == B::foo())
;     return ((B*)a)->bar(); // Inlined from promoted direct call to B::foo()
;   else
;     return a->foo();
;
; The inlined code seqence will have a type test against "_ZTS1B",
; which will allow us to devirtualize indirect call ((B*)a)->bar() to B::bar();
; Both that type test and the one for the fallback a->foo() indirect call
; will use the same vtable pointer. Without a dominance check, we could
; incorrectly devirtualize a->foo() to B::foo();

; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t.o %s

; RUN: llvm-lto2 run %t.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -o %t3 \
; RUN:   -r=%t.o,_Z3bazP1A,px \
; RUN:   -r=%t.o,_ZN1A3fooEv, \
; RUN:   -r=%t.o,_ZN1A3barEv, \
; RUN:   -r=%t.o,_ZN1B3fooEv, \
; RUN:   -r=%t.o,_ZN1B3barEv, \
; RUN:   -r=%t.o,_ZTV1A, \
; RUN:   -r=%t.o,_ZTV1B, \
; RUN:   -r=%t.o,_ZN1A3fooEv, \
; RUN:   -r=%t.o,_ZN1A3barEv, \
; RUN:   -r=%t.o,_ZN1B3fooEv, \
; RUN:   -r=%t.o,_ZN1B3barEv, \
; RUN:   -r=%t.o,_ZTV1A,px \
; RUN:   -r=%t.o,_ZTV1B,px 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t3.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

; We should only devirtualize the inlined call to bar().
; REMARK-NOT: single-impl: devirtualized a call to _ZN1B3fooEv
; REMARK: single-impl: devirtualized a call to _ZN1B3barEv
; REMARK-NOT: single-impl: devirtualized a call to _ZN1B3fooEv

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%class.A = type { i32 (...)** }
%class.B = type { %class.A }

@_ZTV1A = linkonce_odr hidden unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%class.A*)* @_ZN1A3fooEv to i8*), i8* bitcast (i32 (%class.A*)* @_ZN1A3barEv to i8*)] }, align 8, !type !0
@_ZTV1B = hidden unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%class.B*)* @_ZN1B3fooEv to i8*), i8* bitcast (i32 (%class.B*)* @_ZN1B3barEv to i8*)] }, align 8, !type !0, !type !1

define hidden i32 @_Z3bazP1A(%class.A* %a) local_unnamed_addr {
entry:
  %0 = bitcast %class.A* %a to i32 (%class.A*)***
  %vtable = load i32 (%class.A*)**, i32 (%class.A*)*** %0, align 8
  %1 = bitcast i32 (%class.A*)** %vtable to i8*
  %2 = tail call i1 @llvm.type.test(i8* %1, metadata !"_ZTS1A")
  tail call void @llvm.assume(i1 %2)
  %3 = load i32 (%class.A*)*, i32 (%class.A*)** %vtable, align 8
  ; This is the compare instruction inserted by ICP
  %4 = icmp eq i32 (%class.A*)* %3, bitcast (i32 (%class.B*)* @_ZN1B3fooEv to i32 (%class.A*)*)
  br i1 %4, label %if.true.direct_targ, label %if.false.orig_indirect

; This block contains the promoted and inlined call to B::foo();
; CHECK-IR: if.true.direct_targ:                              ; preds = %entry
if.true.direct_targ:                              ; preds = %entry
  %5 = bitcast %class.A* %a to %class.B*
  %6 = bitcast i32 (%class.A*)** %vtable to i8*
  %7 = tail call i1 @llvm.type.test(i8* %6, metadata !"_ZTS1B")
  tail call void @llvm.assume(i1 %7)
  %vfn.i1 = getelementptr inbounds i32 (%class.A*)*, i32 (%class.A*)** %vtable, i64 1
  %vfn.i = bitcast i32 (%class.A*)** %vfn.i1 to i32 (%class.B*)**
  %8 = load i32 (%class.B*)*, i32 (%class.B*)** %vfn.i, align 8
; Call to bar() can be devirtualized to call to B::bar(), since it was
; inlined from B::foo() after ICP introduced the guarded promotion.
; CHECK-IR: %call.i = tail call i32 @_ZN1B3barEv(ptr nonnull %a)
  %call.i = tail call i32 %8(%class.B* %5)
  br label %if.end.icp

; This block contains the fallback indirect call a->foo()
; CHECK-IR: if.false.orig_indirect:
if.false.orig_indirect:                           ; preds = %entry
; Fallback indirect call to foo() cannot be devirtualized.
; CHECK-IR: %call = tail call i32 %
  %call = tail call i32 %3(%class.A* nonnull %a)
  br label %if.end.icp

if.end.icp:                                       ; preds = %if.false.orig_indirect, %if.true.direct_targ
  %9 = phi i32 [ %call, %if.false.orig_indirect ], [ %call.i, %if.true.direct_targ ]
  ret i32 %9
}

declare i1 @llvm.type.test(i8*, metadata)

declare void @llvm.assume(i1)

declare dso_local i32 @_ZN1B3fooEv(%class.B* %this) unnamed_addr
declare dso_local i32 @_ZN1B3barEv(%class.B*) unnamed_addr
declare dso_local i32 @_ZN1A3barEv(%class.A* %this) unnamed_addr
declare dso_local i32 @_ZN1A3fooEv(%class.A* %this) unnamed_addr

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
