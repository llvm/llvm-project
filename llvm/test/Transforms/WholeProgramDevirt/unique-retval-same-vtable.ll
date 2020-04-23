; Test for PR45393: Two virtual functions that return unique i1 values
; in the same vtable. Both calls are optimized to a comparison of
; this's vptr against the address of the vtable. When nesting these
; checks, LLVM would previously assume the nested check always fails,
; but that assumption does not hold if both checks refer to the same vtable.
; This tests checks that this case is handled correctly.
;
; RUN: opt -S -wholeprogramdevirt -wholeprogramdevirt-summary-action=import \
; RUN:   -wholeprogramdevirt-read-summary=%p/Inputs/unique-retval-same-vtable.yaml \
; RUN:   -O2 -o - %s | FileCheck %s
;
; Check that C::f() contains both possible return values.
; CHECK-LABEL: define {{.*}} @_ZNK1C1fEv
; CHECK-NOT: }
; CHECK: 20074028
; CHECK-NOT: }
; CHECK: 1008434

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.C = type { i32 (...)** }

define hidden i32 @_ZNK1C1fEv(%class.C* %this) {
entry:
  %0 = bitcast %class.C* %this to i1 (%class.C*)***
  %vtable = load i1 (%class.C*)**, i1 (%class.C*)*** %0
  %1 = bitcast i1 (%class.C*)** %vtable to i8*
  %2 = tail call i1 @llvm.type.test(i8* %1, metadata !"_ZTS1C")
  tail call void @llvm.assume(i1 %2)
  %vfn = getelementptr inbounds i1 (%class.C*)*, i1 (%class.C*)** %vtable, i64 2
  %3 = load i1 (%class.C*)*, i1 (%class.C*)** %vfn
  %call = tail call zeroext i1 %3(%class.C* %this)
  br i1 %call, label %if.then, label %return

if.then:
  %vtable2 = load i1 (%class.C*)**, i1 (%class.C*)*** %0
  %4 = bitcast i1 (%class.C*)** %vtable2 to i8*
  %5 = tail call i1 @llvm.type.test(i8* %4, metadata !"_ZTS1C")
  tail call void @llvm.assume(i1 %5)
  %vfn3 = getelementptr inbounds i1 (%class.C*)*, i1 (%class.C*)** %vtable2, i64 3
  %6 = load i1 (%class.C*)*, i1 (%class.C*)** %vfn3
  ; The method being called here and the method being called before
  ; the branch above both return true in the same vtable and only that
  ; vtable. Therefore, if this call is reached, we must select
  ; 20074028. Earlier versions of LLVM mistakenly concluded that
  ; this code *never* selects 200744028.
  %call4 = tail call zeroext i1 %6(%class.C* nonnull %this)
  %. = select i1 %call4, i32 20074028, i32 3007762
  br label %return

return:
  %retval.0 = phi i32 [ %., %if.then ], [ 1008434, %entry ]
  ret i32 %retval.0
}

declare i1 @llvm.type.test(i8*, metadata)

declare void @llvm.assume(i1)
