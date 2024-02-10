
; RUN: opt < %s -passes=pgo-icall-prom -S | FileCheck %s --check-prefix=ICALL-FUNC

; Invoke instcombine after pgo-icall-prom so the address calculation instructions for virtual calls get sink into the basic block for indirect fallback.
; RUN: opt < %s -passes='pgo-icall-prom,instcombine' -icp-enable-vtable-cmp -S | FileCheck %s --check-prefix=ICALL-VTABLE

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTV4Base = dso_local constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZN4Base5func1Ei, ptr @_ZN4Base5func2Ev] }, !type !0
@_ZTV8Derived1 = dso_local constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZN8Derived15func1Ei, ptr @_ZN4Base5func2Ev] }, !type !0, !type !1
@_ZTV8Derived2 = dso_local constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZN8Derived25func1Ei, ptr @_ZN4Base5func2Ev] }, !type !0, !type !2
@_ZTV8Derived3 = dso_local constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZN8Derived35func1Ei, ptr @_ZN4Base5func2Ev] }, !type !0, !type !3

; Test the IR transformation from function-based indirect-call promotion and vtable-based indirect-call promotion.

; The tested function has one function candidate which comes from one vtable.
define i32 @test_one_function_one_vtable(ptr %d) {
; ICALL-FUNC-LABEL: define i32 @test_one_function_one_vtable(
; ICALL-FUNC-SAME: ptr [[D:%.*]]) {
; ICALL-FUNC-NEXT:  entry:
; ICALL-FUNC-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[D]], align 8, !prof [[PROF4:![0-9]+]]
; ICALL-FUNC-NEXT:    [[TMP0:%.*]] = tail call i1 @llvm.type.test(ptr [[VTABLE]], metadata !"_ZTS4Base")
; ICALL-FUNC-NEXT:    tail call void @llvm.assume(i1 [[TMP0]])
; ICALL-FUNC-NEXT:    [[VFN:%.*]] = getelementptr inbounds ptr, ptr [[VTABLE]], i64 1
; ICALL-FUNC-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[VFN]], align 8
; ICALL-FUNC-NEXT:    [[TMP2:%.*]] = icmp eq ptr [[TMP1]], @_ZN4Base5func2Ev
; ICALL-FUNC-NEXT:    br i1 [[TMP2]], label [[IF_TRUE_DIRECT_TARG:%.*]], label [[IF_FALSE_ORIG_INDIRECT:%.*]], !prof [[PROF5:![0-9]+]]
; ICALL-FUNC:       if.true.direct_targ:
; ICALL-FUNC-NEXT:    [[TMP3:%.*]] = tail call i32 @_ZN4Base5func2Ev(ptr [[D]])
; ICALL-FUNC-NEXT:    br label [[IF_END_ICP:%.*]]
; ICALL-FUNC:       if.false.orig_indirect:
; ICALL-FUNC-NEXT:    [[CALL:%.*]] = tail call i32 [[TMP1]](ptr [[D]])
; ICALL-FUNC-NEXT:    br label [[IF_END_ICP]]
; ICALL-FUNC:       if.end.icp:
; ICALL-FUNC-NEXT:    [[TMP4:%.*]] = phi i32 [ [[CALL]], [[IF_FALSE_ORIG_INDIRECT]] ], [ [[TMP3]], [[IF_TRUE_DIRECT_TARG]] ]
; ICALL-FUNC-NEXT:    ret i32 [[TMP4]]
;
; ICALL-VTABLE-LABEL: define i32 @test_one_function_one_vtable(
; ICALL-VTABLE-SAME: ptr [[D:%.*]]) {
; ICALL-VTABLE-NEXT:  entry:
; ICALL-VTABLE-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[D]], align 8
; ICALL-VTABLE-NEXT:    [[TMP0:%.*]] = tail call i1 @llvm.type.test(ptr [[VTABLE]], metadata !"_ZTS4Base")
; ICALL-VTABLE-NEXT:    tail call void @llvm.assume(i1 [[TMP0]])
; ICALL-VTABLE-NEXT:    [[TMP1:%.*]] = icmp eq ptr [[VTABLE]], getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV8Derived2, i64 0, i32 0, i64 2)
; ICALL-VTABLE-NEXT:    br i1 [[TMP1]], label [[IF_TRUE_DIRECT_TARG:%.*]], label [[IF_FALSE_ORIG_INDIRECT:%.*]], !prof [[PROF4:![0-9]+]]
; ICALL-VTABLE:       if.true.direct_targ:
; ICALL-VTABLE-NEXT:    [[TMP2:%.*]] = tail call i32 @_ZN4Base5func2Ev(ptr nonnull [[D]])
; ICALL-VTABLE-NEXT:    br label [[IF_END_ICP:%.*]]
; ICALL-VTABLE:       if.false.orig_indirect:
; ICALL-VTABLE-NEXT:    [[VFN:%.*]] = getelementptr inbounds i8, ptr [[VTABLE]], i64 8
; ICALL-VTABLE-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[VFN]], align 8
; ICALL-VTABLE-NEXT:    [[CALL:%.*]] = tail call i32 [[TMP3]](ptr nonnull [[D]])
; ICALL-VTABLE-NEXT:    br label [[IF_END_ICP]]
; ICALL-VTABLE:       if.end.icp:
; ICALL-VTABLE-NEXT:    [[TMP4:%.*]] = phi i32 [ [[CALL]], [[IF_FALSE_ORIG_INDIRECT]] ], [ [[TMP2]], [[IF_TRUE_DIRECT_TARG]] ]
; ICALL-VTABLE-NEXT:    ret i32 [[TMP4]]
;
entry:
  %vtable = load ptr, ptr %d, !prof !4
  %0 = tail call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS4Base")
  tail call void @llvm.assume(i1 %0)
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 1
  %1 = load ptr, ptr %vfn
  %call = tail call i32 %1(ptr %d), !prof !5
  ret i32 %call
}

; The tested function has one function candidate which comes from two vtables.
define i32 @test_one_function_two_vtables(ptr %d) {
; ICALL-FUNC-LABEL: define i32 @test_one_function_two_vtables(
; ICALL-FUNC-SAME: ptr [[D:%.*]]) {
; ICALL-FUNC-NEXT:  entry:
; ICALL-FUNC-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[D]], align 8, !prof [[PROF6:![0-9]+]]
; ICALL-FUNC-NEXT:    [[TMP0:%.*]] = tail call i1 @llvm.type.test(ptr [[VTABLE]], metadata !"_ZTS4Base")
; ICALL-FUNC-NEXT:    tail call void @llvm.assume(i1 [[TMP0]])
; ICALL-FUNC-NEXT:    [[VFN:%.*]] = getelementptr inbounds ptr, ptr [[VTABLE]], i64 1
; ICALL-FUNC-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[VFN]], align 8
; ICALL-FUNC-NEXT:    [[TMP2:%.*]] = icmp eq ptr [[TMP1]], @_ZN4Base5func2Ev
; ICALL-FUNC-NEXT:    br i1 [[TMP2]], label [[IF_TRUE_DIRECT_TARG:%.*]], label [[IF_FALSE_ORIG_INDIRECT:%.*]], !prof [[PROF5]]
; ICALL-FUNC:       if.true.direct_targ:
; ICALL-FUNC-NEXT:    [[TMP3:%.*]] = tail call i32 @_ZN4Base5func2Ev(ptr [[D]])
; ICALL-FUNC-NEXT:    br label [[IF_END_ICP:%.*]]
; ICALL-FUNC:       if.false.orig_indirect:
; ICALL-FUNC-NEXT:    [[CALL:%.*]] = tail call i32 [[TMP1]](ptr [[D]])
; ICALL-FUNC-NEXT:    br label [[IF_END_ICP]]
; ICALL-FUNC:       if.end.icp:
; ICALL-FUNC-NEXT:    [[TMP4:%.*]] = phi i32 [ [[CALL]], [[IF_FALSE_ORIG_INDIRECT]] ], [ [[TMP3]], [[IF_TRUE_DIRECT_TARG]] ]
; ICALL-FUNC-NEXT:    ret i32 [[TMP4]]
;
; ICALL-VTABLE-LABEL: define i32 @test_one_function_two_vtables(
; ICALL-VTABLE-SAME: ptr [[D:%.*]]) {
; ICALL-VTABLE-NEXT:  entry:
; ICALL-VTABLE-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[D]], align 8
; ICALL-VTABLE-NEXT:    [[TMP0:%.*]] = tail call i1 @llvm.type.test(ptr [[VTABLE]], metadata !"_ZTS4Base")
; ICALL-VTABLE-NEXT:    tail call void @llvm.assume(i1 [[TMP0]])
; ICALL-VTABLE-NEXT:    [[TMP1:%.*]] = icmp eq ptr [[VTABLE]], getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV8Derived1, i64 0, i32 0, i64 2)
; ICALL-VTABLE-NEXT:    [[TMP2:%.*]] = icmp eq ptr [[VTABLE]], getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV8Derived2, i64 0, i32 0, i64 2)
; ICALL-VTABLE-NEXT:    [[ICMP_OR:%.*]] = or i1 [[TMP1]], [[TMP2]]
; ICALL-VTABLE-NEXT:    br i1 [[ICMP_OR]], label [[IF_TRUE_DIRECT_TARG:%.*]], label [[IF_FALSE_ORIG_INDIRECT:%.*]], !prof [[PROF4]]
; ICALL-VTABLE:       if.true.direct_targ:
; ICALL-VTABLE-NEXT:    [[TMP3:%.*]] = tail call i32 @_ZN4Base5func2Ev(ptr nonnull [[D]])
; ICALL-VTABLE-NEXT:    br label [[IF_END_ICP:%.*]]
; ICALL-VTABLE:       if.false.orig_indirect:
; ICALL-VTABLE-NEXT:    [[VFN:%.*]] = getelementptr inbounds i8, ptr [[VTABLE]], i64 8
; ICALL-VTABLE-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[VFN]], align 8
; ICALL-VTABLE-NEXT:    [[CALL:%.*]] = tail call i32 [[TMP4]](ptr nonnull [[D]])
; ICALL-VTABLE-NEXT:    br label [[IF_END_ICP]]
; ICALL-VTABLE:       if.end.icp:
; ICALL-VTABLE-NEXT:    [[TMP5:%.*]] = phi i32 [ [[CALL]], [[IF_FALSE_ORIG_INDIRECT]] ], [ [[TMP3]], [[IF_TRUE_DIRECT_TARG]] ]
; ICALL-VTABLE-NEXT:    ret i32 [[TMP5]]
;
entry:
  %vtable = load ptr, ptr %d, !prof !6
  %0 = tail call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS4Base")
  tail call void @llvm.assume(i1 %0)
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 1
  %1 = load ptr, ptr %vfn
  %call = tail call i32 %1(ptr %d), !prof !5
  ret i32 %call
}

; The tested function has one function candidate which comes from three vtables.
define i32 @test_one_function_three_vtables(ptr %d) {
; ICALL-FUNC-LABEL: define i32 @test_one_function_three_vtables(
; ICALL-FUNC-SAME: ptr [[D:%.*]]) {
; ICALL-FUNC-NEXT:  entry:
; ICALL-FUNC-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[D]], align 8, !prof [[PROF7:![0-9]+]]
; ICALL-FUNC-NEXT:    [[TMP0:%.*]] = tail call i1 @llvm.type.test(ptr [[VTABLE]], metadata !"_ZTS4Base")
; ICALL-FUNC-NEXT:    tail call void @llvm.assume(i1 [[TMP0]])
; ICALL-FUNC-NEXT:    [[VFN:%.*]] = getelementptr inbounds ptr, ptr [[VTABLE]], i64 1
; ICALL-FUNC-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[VFN]], align 8
; ICALL-FUNC-NEXT:    [[TMP2:%.*]] = icmp eq ptr [[TMP1]], @_ZN4Base5func2Ev
; ICALL-FUNC-NEXT:    br i1 [[TMP2]], label [[IF_TRUE_DIRECT_TARG:%.*]], label [[IF_FALSE_ORIG_INDIRECT:%.*]], !prof [[PROF5]]
; ICALL-FUNC:       if.true.direct_targ:
; ICALL-FUNC-NEXT:    [[TMP3:%.*]] = tail call i32 @_ZN4Base5func2Ev(ptr [[D]])
; ICALL-FUNC-NEXT:    br label [[IF_END_ICP:%.*]]
; ICALL-FUNC:       if.false.orig_indirect:
; ICALL-FUNC-NEXT:    [[CALL:%.*]] = tail call i32 [[TMP1]](ptr [[D]])
; ICALL-FUNC-NEXT:    br label [[IF_END_ICP]]
; ICALL-FUNC:       if.end.icp:
; ICALL-FUNC-NEXT:    [[TMP4:%.*]] = phi i32 [ [[CALL]], [[IF_FALSE_ORIG_INDIRECT]] ], [ [[TMP3]], [[IF_TRUE_DIRECT_TARG]] ]
; ICALL-FUNC-NEXT:    ret i32 [[TMP4]]
;
; ICALL-VTABLE-LABEL: define i32 @test_one_function_three_vtables(
; ICALL-VTABLE-SAME: ptr [[D:%.*]]) {
; ICALL-VTABLE-NEXT:  entry:
; ICALL-VTABLE-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[D]], align 8
; ICALL-VTABLE-NEXT:    [[TMP0:%.*]] = tail call i1 @llvm.type.test(ptr [[VTABLE]], metadata !"_ZTS4Base")
; ICALL-VTABLE-NEXT:    tail call void @llvm.assume(i1 [[TMP0]])
; ICALL-VTABLE-NEXT:    [[TMP1:%.*]] = icmp eq ptr [[VTABLE]], getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV8Derived1, i64 0, i32 0, i64 2)
; ICALL-VTABLE-NEXT:    [[TMP2:%.*]] = icmp eq ptr [[VTABLE]], getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV8Derived2, i64 0, i32 0, i64 2)
; ICALL-VTABLE-NEXT:    [[TMP3:%.*]] = icmp eq ptr [[VTABLE]], getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV4Base, i64 0, i32 0, i64 2)
; ICALL-VTABLE-NEXT:    [[ICMP_OR:%.*]] = or i1 [[TMP1]], [[TMP2]]
; ICALL-VTABLE-NEXT:    [[ICMP_OR1:%.*]] = or i1 [[ICMP_OR]], [[TMP3]]
; ICALL-VTABLE-NEXT:    br i1 [[ICMP_OR1]], label [[IF_TRUE_DIRECT_TARG:%.*]], label [[IF_FALSE_ORIG_INDIRECT:%.*]], !prof [[PROF4]]
; ICALL-VTABLE:       if.true.direct_targ:
; ICALL-VTABLE-NEXT:    [[TMP4:%.*]] = tail call i32 @_ZN4Base5func2Ev(ptr nonnull [[D]])
; ICALL-VTABLE-NEXT:    br label [[IF_END_ICP:%.*]]
; ICALL-VTABLE:       if.false.orig_indirect:
; ICALL-VTABLE-NEXT:    [[VFN:%.*]] = getelementptr inbounds i8, ptr [[VTABLE]], i64 8
; ICALL-VTABLE-NEXT:    [[TMP5:%.*]] = load ptr, ptr [[VFN]], align 8
; ICALL-VTABLE-NEXT:    [[CALL:%.*]] = tail call i32 [[TMP5]](ptr nonnull [[D]])
; ICALL-VTABLE-NEXT:    br label [[IF_END_ICP]]
; ICALL-VTABLE:       if.end.icp:
; ICALL-VTABLE-NEXT:    [[TMP6:%.*]] = phi i32 [ [[CALL]], [[IF_FALSE_ORIG_INDIRECT]] ], [ [[TMP4]], [[IF_TRUE_DIRECT_TARG]] ]
; ICALL-VTABLE-NEXT:    ret i32 [[TMP6]]
;
entry:
  %vtable = load ptr, ptr %d, !prof !7
  %0 = tail call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS4Base")
  tail call void @llvm.assume(i1 %0)
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 1
  %1 = load ptr, ptr %vfn
  %call = tail call i32 %1(ptr %d), !prof !5
  ret i32 %call
}


declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1 noundef)
declare i32 @_ZN4Base5func1Ei(ptr, i32)
declare i32 @_ZN8Derived15func1Ei(ptr, i32)
declare i32 @_ZN8Derived25func1Ei(ptr, i32)
declare i32 @_ZN8Derived35func1Ei(ptr, i32)

define i32 @_ZN4Base5func2Ev(ptr %this) {
entry:
  ret i32 0
}

!0 = !{i64 16, !"_ZTS4Base"}
!1 = !{i64 16, !"_ZTS8Derived1"}
!2 = !{i64 16, !"_ZTS8Derived2"}
!3 = !{i64 16, !"_ZTS8Derived3"}
!4 = !{!"VP", i32 2, i64 1600, i64 5035968517245772950, i64 1600}
!5 = !{!"VP", i32 0, i64 1600, i64 -3104805163612457913, i64 1600}
!6 = !{!"VP", i32 2, i64 1600, i64 -9064381665493407289, i64 1000, i64 5035968517245772950, i64 600}
!7 = !{!"VP", i32 2, i64 1600, i64 -9064381665493407289, i64 600, i64 5035968517245772950, i64 550, i64 1960855528937986108, i64 450}

; ICALL-FUNC: [[PROF4]] = !{!"VP", i32 2, i64 1600, i64 5035968517245772950, i64 1600}
; ICALL-FUNC: [[PROF5]] = !{!"branch_weights", i32 1600, i32 0}
; ICALL-FUNC: [[PROF6]] = !{!"VP", i32 2, i64 1600, i64 -9064381665493407289, i64 1000, i64 5035968517245772950, i64 600}
; ICALL-FUNC: [[PROF7]] = !{!"VP", i32 2, i64 1600, i64 -9064381665493407289, i64 600, i64 5035968517245772950, i64 550, i64 1960855528937986108, i64 450}

; ICALL-VTABLE: [[PROF4]] = !{!"branch_weights", i32 1600, i32 0}
