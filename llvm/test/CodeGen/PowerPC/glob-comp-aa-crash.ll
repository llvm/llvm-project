; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux -mcpu=a2 < %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64le-unknown-linux"

%"class.std::__1::__assoc_sub_state" = type { %"class.std::__1::__shared_count", %"class.std::__exception_ptr::exception_ptr", %"class.std::__1::mutex", %"class.std::__1::condition_variable", i32 }
%"class.std::__1::__shared_count" = type { ptr, i64 }
%"class.std::__exception_ptr::exception_ptr" = type { ptr }
%"class.std::__1::mutex" = type { %union.pthread_mutex_t }
%union.pthread_mutex_t = type { %"struct.<anonymous union>::__pthread_mutex_s" }
%"struct.<anonymous union>::__pthread_mutex_s" = type { i32, i32, i32, i32, i32, i32, %struct.__pthread_internal_list }
%struct.__pthread_internal_list = type { ptr, ptr }
%"class.std::__1::condition_variable" = type { %union.pthread_cond_t }
%union.pthread_cond_t = type { %struct.anon }
%struct.anon = type { i32, i32, i64, i64, i64, ptr, i32, i32 }
%"class.std::__1::unique_lock" = type { ptr, i8 }

declare i32 @__gxx_personality_v0(...)

; Function Attrs: optsize
define void @_ZNSt3__117__assoc_sub_state4copyEv(ptr %this) #0 align 2 personality ptr @__gxx_personality_v0 {
entry:
  %__lk = alloca %"class.std::__1::unique_lock", align 8
  %ref.tmp = alloca %"class.std::__exception_ptr::exception_ptr", align 8
  %tmp = alloca { i64, i64 }, align 8
  %agg.tmp = alloca %"class.std::__exception_ptr::exception_ptr", align 8
  %__mut_ = getelementptr inbounds %"class.std::__1::__assoc_sub_state", ptr %this, i64 0, i32 2
  store ptr %__mut_, ptr %__lk, align 8, !tbaa !5
  %__owns_.i.i = getelementptr inbounds %"class.std::__1::unique_lock", ptr %__lk, i64 0, i32 1
  store i8 1, ptr %__owns_.i.i, align 8, !tbaa !6
  call void @_ZNSt3__15mutex4lockEv(ptr %__mut_) #4
  invoke void @_ZNSt3__117__assoc_sub_state10__sub_waitERNS_11unique_lockINS_5mutexEEE(ptr %this, ptr %__lk) #4
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %__exception_ = getelementptr inbounds %"class.std::__1::__assoc_sub_state", ptr %this, i64 0, i32 1
  call void @llvm.memset.p0.i64(ptr align 8 %tmp, i8 0, i64 16, i1 false)
  call void @_ZNSt15__exception_ptr13exception_ptrC1EMS0_FvvE(ptr %ref.tmp, ptr byval({ i64, i64 }) %tmp) #5
  %call = call zeroext i1 @_ZNSt15__exception_ptrneERKNS_13exception_ptrES2_(ptr %__exception_, ptr %ref.tmp) #5
  call void @_ZNSt15__exception_ptr13exception_ptrD1Ev(ptr %ref.tmp) #5
  br i1 %call, label %if.then, label %if.end

if.then:                                          ; preds = %invoke.cont
  call void @_ZNSt15__exception_ptr13exception_ptrC1ERKS0_(ptr %agg.tmp, ptr %__exception_) #5
  invoke void @_ZSt17rethrow_exceptionNSt15__exception_ptr13exception_ptrE(ptr %agg.tmp) #6
          to label %invoke.cont4 unwind label %lpad3

invoke.cont4:                                     ; preds = %if.then
  unreachable

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          cleanup
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = extractvalue { ptr, i32 } %0, 1
  br label %ehcleanup

lpad3:                                            ; preds = %if.then
  %3 = landingpad { ptr, i32 }
          cleanup
  %4 = extractvalue { ptr, i32 } %3, 0
  %5 = extractvalue { ptr, i32 } %3, 1
  call void @_ZNSt15__exception_ptr13exception_ptrD1Ev(ptr %agg.tmp) #5
  br label %ehcleanup

if.end:                                           ; preds = %invoke.cont
  %6 = load i8, ptr %__owns_.i.i, align 8, !tbaa !6, !range !4
  %tobool.i.i = icmp eq i8 %6, 0
  br i1 %tobool.i.i, label %_ZNSt3__111unique_lockINS_5mutexEED1Ev.exit, label %if.then.i.i

if.then.i.i:                                      ; preds = %if.end
  %7 = load ptr, ptr %__lk, align 8, !tbaa !5
  call void @_ZNSt3__15mutex6unlockEv(ptr %7) #5
  br label %_ZNSt3__111unique_lockINS_5mutexEED1Ev.exit

_ZNSt3__111unique_lockINS_5mutexEED1Ev.exit:      ; preds = %if.then.i.i, %if.end
  ret void

ehcleanup:                                        ; preds = %lpad3, %lpad
  %exn.slot.0 = phi ptr [ %4, %lpad3 ], [ %1, %lpad ]
  %ehselector.slot.0 = phi i32 [ %5, %lpad3 ], [ %2, %lpad ]
  %8 = load i8, ptr %__owns_.i.i, align 8, !tbaa !6, !range !4
  %tobool.i.i9 = icmp eq i8 %8, 0
  br i1 %tobool.i.i9, label %_ZNSt3__111unique_lockINS_5mutexEED1Ev.exit12, label %if.then.i.i11

if.then.i.i11:                                    ; preds = %ehcleanup
  %9 = load ptr, ptr %__lk, align 8, !tbaa !5
  call void @_ZNSt3__15mutex6unlockEv(ptr %9) #5
  br label %_ZNSt3__111unique_lockINS_5mutexEED1Ev.exit12

_ZNSt3__111unique_lockINS_5mutexEED1Ev.exit12:    ; preds = %if.then.i.i11, %ehcleanup
  %lpad.val = insertvalue { ptr, i32 } undef, ptr %exn.slot.0, 0
  %lpad.val5 = insertvalue { ptr, i32 } %lpad.val, i32 %ehselector.slot.0, 1
  resume { ptr, i32 } %lpad.val5
}

; Function Attrs: optsize
declare void @_ZNSt3__117__assoc_sub_state10__sub_waitERNS_11unique_lockINS_5mutexEEE(ptr, ptr) #0 align 2

; Function Attrs: nounwind optsize
declare zeroext i1 @_ZNSt15__exception_ptrneERKNS_13exception_ptrES2_(ptr, ptr) #1

; Function Attrs: nounwind optsize
declare void @_ZNSt15__exception_ptr13exception_ptrC1EMS0_FvvE(ptr, ptr byval({ i64, i64 })) #1

; Function Attrs: nounwind optsize
declare void @_ZNSt15__exception_ptr13exception_ptrD1Ev(ptr) #1

; Function Attrs: noreturn optsize
declare void @_ZSt17rethrow_exceptionNSt15__exception_ptr13exception_ptrE(ptr) #2

; Function Attrs: nounwind optsize
declare void @_ZNSt15__exception_ptr13exception_ptrC1ERKS0_(ptr, ptr) #1

; Function Attrs: nounwind optsize
declare void @_ZNSt3__15mutex6unlockEv(ptr) #1

; Function Attrs: optsize
declare void @_ZNSt3__15mutex4lockEv(ptr) #0

; Function Attrs: nounwind
declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) #3

attributes #0 = { optsize "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind optsize "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noreturn optsize "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { optsize }
attributes #5 = { nounwind optsize }
attributes #6 = { noreturn optsize }

!0 = !{!"any pointer", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!"bool", !1}
!4 = !{i8 0, i8 2}
!5 = !{!0, !0, i64 0}
!6 = !{!3, !3, i64 0}
