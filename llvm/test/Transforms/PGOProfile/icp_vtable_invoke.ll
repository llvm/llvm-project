; RUN: opt < %s -passes='pgo-icall-prom' -enable-vtable-profile-use -S | FileCheck %s --check-prefix=VTABLE

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTV4Base = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN4Base10get_ticketEv] }, !type !0, !type !1
@_ZTV7Derived = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN7Derived10get_ticketEv] }, !type !0, !type !1, !type !2, !type !3

@.str = private constant [15 x i8] c"out of tickets\00"

define i32 @test(ptr %b) personality ptr @__gxx_personality_v0 {
; VTABLE-LABEL: define i32 @test(
; VTABLE-SAME: ptr [[B:%.*]]) personality ptr @__gxx_personality_v0 {
; VTABLE-NEXT:  [[ENTRY:.*:]]
; VTABLE-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[B]], align 8
; VTABLE-NEXT:    [[TMP0:%.*]] = tail call i1 @llvm.type.test(ptr [[VTABLE]], metadata !"_ZTS4Base")
; VTABLE-NEXT:    tail call void @llvm.assume(i1 [[TMP0]])
; VTABLE-NEXT:    [[TMP3:%.*]] = icmp eq ptr [[VTABLE]], getelementptr inbounds (i8, ptr @_ZTV7Derived, i32 16)
; VTABLE-NEXT:    br i1 [[TMP3]], label %[[IF_TRUE_DIRECT_TARG:.*]], label %[[IF_FALSE_ORIG_INDIRECT:.*]], !prof [[PROF4:![0-9]+]]
; VTABLE:       [[IF_TRUE_DIRECT_TARG]]:
; VTABLE-NEXT:    [[TMP2:%.*]] = invoke i32 @_ZN7Derived10get_ticketEv(ptr [[B]])
; VTABLE-NEXT:            to label %[[IF_END_ICP:.*]] unwind label %[[LPAD:.*]]
; VTABLE:       [[IF_FALSE_ORIG_INDIRECT]]:
; VTABLE-NEXT:    [[TMP4:%.*]] = icmp eq ptr [[VTABLE]], getelementptr inbounds (i8, ptr @_ZTV4Base, i32 16)
; VTABLE-NEXT:    br i1 [[TMP4]], label %[[IF_TRUE_DIRECT_TARG1:.*]], label %[[IF_FALSE_ORIG_INDIRECT2:.*]], !prof [[PROF5:![0-9]+]]
; VTABLE:       [[IF_TRUE_DIRECT_TARG1]]:
; VTABLE-NEXT:    [[TMP5:%.*]] = invoke i32 @_ZN4Base10get_ticketEv(ptr [[B]])
; VTABLE-NEXT:            to label %[[IF_END_ICP3:.*]] unwind label %[[LPAD]]
; VTABLE:       [[IF_FALSE_ORIG_INDIRECT2]]:
; VTABLE-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[VTABLE]], align 8
; VTABLE-NEXT:    [[CALL:%.*]] = invoke i32 [[TMP1]](ptr [[B]])
; VTABLE-NEXT:            to label %[[IF_END_ICP3]] unwind label %[[LPAD]]
; VTABLE:       [[IF_END_ICP3]]:
; VTABLE-NEXT:    [[TMP6:%.*]] = phi i32 [ [[CALL]], %[[IF_FALSE_ORIG_INDIRECT2]] ], [ [[TMP5]], %[[IF_TRUE_DIRECT_TARG1]] ]
; VTABLE-NEXT:    br label %[[IF_END_ICP]]
; VTABLE:       [[IF_END_ICP]]:
; VTABLE-NEXT:    [[TMP7:%.*]] = phi i32 [ [[TMP6]], %[[IF_END_ICP3]] ], [ [[TMP2]], %[[IF_TRUE_DIRECT_TARG]] ]
; VTABLE-NEXT:    br label %[[NEXT:.*]]
; VTABLE:       [[NEXT]]:
; VTABLE-NEXT:    ret i32 [[TMP7]]
; VTABLE:       [[LPAD]]:
; VTABLE-NEXT:    [[EXN:%.*]] = landingpad { ptr, i32 }
; VTABLE-NEXT:            cleanup
; VTABLE-NEXT:    unreachable
;
entry:
  %vtable = load ptr, ptr %b, !prof !4
  %0 = tail call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS4Base")
  tail call void @llvm.assume(i1 %0)
  %1 = load ptr, ptr %vtable
  %call = invoke i32 %1(ptr %b) to label %next unwind label %lpad, !prof !5

next:
  ret i32 %call

lpad:
  %exn = landingpad {ptr, i32}
  cleanup
  unreachable
}

declare void @make_error(ptr, ptr, i32)
declare i32 @get_ticket_id()
declare ptr @__cxa_allocate_exception(i64)

define i32 @_ZN4Base10get_ticketEv(ptr %this) personality ptr @__gxx_personality_v0 {
entry:
  %call = tail call i32 @get_ticket_id()
  %cmp.not = icmp eq i32 %call, -1
  br i1 %cmp.not, label %if.end, label %if.then

if.then:
  ret i32 %call

if.end:
  %exception = tail call ptr @__cxa_allocate_exception(i64 1)
  invoke void @make_error(ptr %exception, ptr @.str, i32 1)
  to label %invoke.cont unwind label %lpad

invoke.cont:
  unreachable

lpad:
  %0 = landingpad { ptr, i32 }
  cleanup
  resume { ptr, i32 } %0
}

define i32 @_ZN7Derived10get_ticketEv(ptr %this) personality ptr @__gxx_personality_v0 {
entry:
  %call = tail call i32 @get_ticket_id()
  %cmp.not = icmp eq i32 %call, -1
  br i1 %cmp.not, label %if.end, label %if.then

if.then:
  ret i32 %call

if.end:
  %exception = tail call ptr @__cxa_allocate_exception(i64 1)
  invoke void @make_error(ptr %exception, ptr @.str, i32 2)
  to label %invoke.cont unwind label %lpad

invoke.cont:
  unreachable

lpad:
  %0 = landingpad { ptr, i32 }
  cleanup
  resume { ptr, i32 } %0
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)
declare i32 @__gxx_personality_v0(...)

!0 = !{i64 16, !"_ZTS4Base"}
!1 = !{i64 16, !"_ZTSM4BaseFivE.virtual"}
!2 = !{i64 16, !"_ZTS7Derived"}
!3 = !{i64 16, !"_ZTSM7DerivedFivE.virtual"}
!4 = !{!"VP", i32 2, i64 1600, i64 13870436605473471591, i64 900, i64 1960855528937986108, i64 700}
!5 = !{!"VP", i32 0, i64 1600, i64 14811317294552474744, i64 900, i64 9261744921105590125, i64 700}

; VTABLE: [[PROF4]] = !{!"branch_weights", i32 900, i32 700}
; VTABLE: [[PROF5]] = !{!"branch_weights", i32 700, i32 0}
;.
