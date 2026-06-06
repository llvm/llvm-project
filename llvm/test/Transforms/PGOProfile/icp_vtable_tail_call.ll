; RUN: opt < %s -passes='pgo-icall-prom' -pass-remarks=pgo-icall-prom -enable-vtable-profile-use -S 2>&1 | FileCheck %s --check-prefixes=VTABLE,REMARK

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; REMARK: remark: <unknown>:0:0: Promote indirect call to _ZN7Derived5func1Eii with count 900 out of 1600, sink 1 instruction(s) and compare 1 vtable(s): {_ZTV7Derived}
; REMARK: remark: <unknown>:0:0: Promote indirect call to _ZN4Base5func1Eii with count 700 out of 700, sink 1 instruction(s) and compare 1 vtable(s): {_ZTV4Base}

@_ZTV7Derived = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN7Derived5func1Eii] }, !type !0, !type !1, !type !2, !type !3
@_ZTV4Base = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN4Base5func1Eii] }, !type !0, !type !1

define i32 @test_tail_call(ptr %ptr, i32 %a, i32 %b) {
; VTABLE-LABEL: define i32 @test_tail_call(
; VTABLE-SAME: ptr [[PTR:%.*]], i32 [[A:%.*]], i32 [[B:%.*]]) {
; VTABLE-NEXT:  entry:
; VTABLE-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[PTR]], align 8
; VTABLE-NEXT:    [[TMP0:%.*]] = tail call i1 @llvm.type.test(ptr [[VTABLE]], metadata !"_ZTS4Base")
; VTABLE-NEXT:    tail call void @llvm.assume(i1 [[TMP0]])
; VTABLE-NEXT:    [[TMP2:%.*]] = icmp eq ptr [[VTABLE]], getelementptr inbounds (i8, ptr @_ZTV7Derived, i32 16)
; VTABLE-NEXT:    br i1 [[TMP2]], label [[IF_TRUE_DIRECT_TARG:%.*]], label [[TMP4:%.*]], !prof [[PROF4:![0-9]+]]
; VTABLE:       if.true.direct_targ:
; VTABLE-NEXT:    [[TMP3:%.*]] = musttail call i32 @_ZN7Derived5func1Eii(ptr [[PTR]], i32 [[A]], i32 [[B]])
; VTABLE-NEXT:    ret i32 [[TMP3]]
; VTABLE:       3:
; VTABLE-NEXT:    [[TMP4:%.*]] = icmp eq ptr [[VTABLE]], getelementptr inbounds (i8, ptr @_ZTV4Base, i32 16)
; VTABLE-NEXT:    br i1 [[TMP4]], label [[IF_TRUE_DIRECT_TARG1:%.*]], label [[TMP7:%.*]], !prof [[PROF5:![0-9]+]]
; VTABLE:       if.true.direct_targ1:
; VTABLE-NEXT:    [[TMP6:%.*]] = musttail call i32 @_ZN4Base5func1Eii(ptr [[PTR]], i32 [[A]], i32 [[B]])
; VTABLE-NEXT:    ret i32 [[TMP6]]
; VTABLE:       6:
; VTABLE-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[VTABLE]], align 8
; VTABLE-NEXT:    [[CALL:%.*]] = musttail call i32 [[TMP1]](ptr [[PTR]], i32 [[A]], i32 [[B]])
; VTABLE-NEXT:    ret i32 [[CALL]]
;
entry:
  %vtable = load ptr, ptr %ptr, !prof !4
  %0 = tail call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS4Base")
  tail call void @llvm.assume(i1 %0)
  %1 = load ptr, ptr %vtable
  %call = musttail call i32 %1(ptr %ptr, i32 %a, i32 %b), !prof !5
  ret i32 %call
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

define i32 @_ZN7Derived5func1Eii(ptr %this, i32 %a, i32 %b) {
entry:
  %sub = sub nsw i32 %a, %b
  ret i32 %sub
}

define i32 @_ZN4Base5func1Eii(ptr %this, i32 %a, i32 %b) {
entry:
  %add = add nsw i32 %b, %a
  ret i32 %add
}


!0 = !{i64 16, !"_ZTS4Base"}
!1 = !{i64 16, !"_ZTSM4BaseFiiiE.virtual"}
!2 = !{i64 16, !"_ZTS7Derived"}
!3 = !{i64 16, !"_ZTSM7DerivedFiiiE.virtual"}
!4 = !{!"VP", i32 2, i64 1600, i64 13870436605473471591, i64 900, i64 1960855528937986108, i64 700}
!5 = !{!"VP", i32 0, i64 1600, i64 7889036118036845314, i64 900, i64 10495086226207060333, i64 700}

; VTABLE: [[PROF4]] = !{!"branch_weights", i32 900, i32 700}
; VTABLE: [[PROF5]] = !{!"branch_weights", i32 700, i32 0}
