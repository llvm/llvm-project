; RUN: opt < %s -passes=partial-inliner -skip-partial-inlining-cost-analysis -inline-threshold=0 -disable-output

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

%0 = type { i32 (...)**, %1, %1, %3, %3, %3, i8, float, %4*, %5*, %5*, i32, i32, i32, i32, float, float, float, i8*, i32, float, float, float, i8, [7 x i8] }
%1 = type { %2, %3 }
%2 = type { [3 x %3] }
%3 = type { [4 x float] }
%4 = type <{ i8*, i16, i16, [4 x i8], i8*, i32, %3, %3, [4 x i8] }>
%5 = type { i32 (...)**, i32, i8* }
%6 = type <{ %7, [4 x i8], %19*, %20*, %30, %35, %3, float, i8, i8, i8, i8, %37, i32, [4 x i8] }>
%7 = type <{ %8, [7 x i8], void (%16*, float)*, void (%16*, float)*, i8*, %17 }>
%8 = type <{ i32 (...)**, %9, %11*, %12, %13*, %14*, %15*, i8 }>
%9 = type <{ i8, [3 x i8], i32, i32, [4 x i8], %0**, i8, [7 x i8] }>
%11 = type { i32 (...)** }
%12 = type { float, i32, i32, float, i8, %15*, i8, i8, i8, float, i8, float, %13* }
%13 = type opaque
%14 = type { i32 (...)** }
%15 = type { i32 (...)** }
%16 = type <{ %8, [7 x i8], void (%16*, float)*, void (%16*, float)*, i8*, %17, [4 x i8] }>
%17 = type { %18 }
%18 = type { float, float, float, float, float, i32, float, float, float, float, float, i32, float, float, float, i32, i32 }
%19 = type { i32 (...)** }
%20 = type <{ i32 (...)**, %21, %25, %9, i8, [7 x i8] }>
%21 = type { %22 }
%22 = type <{ i8, [3 x i8], i32, i32, [4 x i8], %24*, i8, [7 x i8] }>
%24 = type { i32, i32 }
%25 = type <{ i8, [3 x i8], i32, i32, [4 x i8], %27**, i8, [7 x i8] }>
%27 = type { i32, [4 x i8], [4 x %29], i8*, i8*, i32, float, float, i32 }
%29 = type <{ %3, %3, %3, %3, %3, float, float, float, i32, i32, i32, i32, [4 x i8], i8*, float, i8, [3 x i8], float, float, i32, %3, %3, [4 x i8] }>
%30 = type <{ i8, [3 x i8], i32, i32, [4 x i8], %32**, i8, [7 x i8] }>
%32 = type { i32 (...)**, i32, i32, i32, i8, %33*, %33*, float, float, %3, %3, %3 }
%33 = type <{ %0, %2, %3, %3, float, %3, %3, %3, %3, %3, %3, %3, float, float, i8, [3 x i8], float, float, float, float, float, float, %34*, %30, i32, i32, i32, [4 x i8] }>
%34 = type { i32 (...)** }
%35 = type <{ i8, [3 x i8], i32, i32, [4 x i8], %33**, i8, [7 x i8] }>
%37 = type <{ i8, [3 x i8], i32, i32, [4 x i8], %39**, i8, [7 x i8] }>
%39 = type { i32 (...)** }
%40 = type <{ i32 (...)**, %9, %11*, %12, %13*, %14*, %15*, i8, [7 x i8] }>

@gDisableDeactivation = external local_unnamed_addr global i8, align 1
@0 = external dso_local unnamed_addr constant [29 x i8], align 1
@1 = external dso_local unnamed_addr constant [14 x i8], align 1
@2 = external dso_local unnamed_addr constant [22 x i8], align 1
@gDeactivationTime = external local_unnamed_addr global float, align 4

declare void @_ZN15CProfileManager12Stop_ProfileEv() local_unnamed_addr

declare void @_ZN15CProfileManager13Start_ProfileEPKc(i8*) local_unnamed_addr

declare void @_ZN17btCollisionObject18setActivationStateEi(%0*, i32 signext) local_unnamed_addr

declare hidden void @__clang_call_terminate(i8*) local_unnamed_addr

declare i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #0

define void @_ZN23btDiscreteDynamicsWorld28internalSingleStepSimulationEf(%6*, float) unnamed_addr align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) !prof !27 {
  invoke void null(%6* nonnull %0, float %1)
          to label %5 unwind label %3

; <label>:3:                                      ; preds = %2
  %4 = landingpad { i8*, i32 }
          cleanup
  br label %16

; <label>:5:                                      ; preds = %2
  %6 = invoke %15* null(%40* null)
          to label %11 unwind label %13

; <label>:7:                                      ; preds = %5
  invoke void null(%40* null)
          to label %8 unwind label %13

; <label>:8:                                      ; preds = %7
  invoke void null(%6* nonnull %0)
          to label %9 unwind label %13

; <label>:9:                                      ; preds = %8
  invoke void null(%6* nonnull %0, %17* nonnull dereferenceable(68) null)
          to label %10 unwind label %13

; <label>:10:                                     ; preds = %9
  invoke void null(%6* nonnull %0, float %1)
          to label %11 unwind label %13

; <label>:11:
  invoke void @_ZN23btDiscreteDynamicsWorld21updateActivationStateEf(%6* nonnull %0, float %1)
          to label %12 unwind label %13

; <label>:12:
  ret void

; <label>:13:
  %14 = landingpad { i8*, i32 }
          cleanup
  %15 = extractvalue { i8*, i32 } %14, 0
  br label %16


; <label>:16:
  call void @_ZN15CProfileManager12Stop_ProfileEv()
  resume { i8*, i32 } zeroinitializer
}

define void @_ZN23btDiscreteDynamicsWorld21updateActivationStateEf(%6* nocapture readonly, float) local_unnamed_addr align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) !prof !27 {
  %3 = icmp sgt i32 0, 0
  br i1 %3, label %4, label %5, !prof !29

; <label>:4:                                      ; preds = %2
  br i1 false, label %5, label %6, !prof !30

; <label>:5:                                      ; preds = %7, %4, %2
  ret void

; <label>:6:                                      ; preds = %4
  invoke void @_ZN17btCollisionObject18setActivationStateEi(%0* nonnull null, i32 signext 0)
          to label %7 unwind label %8

; <label>:7:                                      ; preds = %6
  invoke void @_ZN17btCollisionObject18setActivationStateEi(%0* nonnull null, i32 signext 1)
          to label %5 unwind label %8

; <label>:8:                                      ; preds = %7, %6
  %9 = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } %9
}

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #1

attributes #0 = { argmemonly nounwind }
attributes #1 = { noreturn nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 6540578580}
!4 = !{!"MaxCount", i64 629805108}
!5 = !{!"MaxInternalCount", i64 40670372}
!6 = !{!"MaxFunctionCount", i64 629805108}
!7 = !{!"NumCounts", i64 8554}
!8 = !{!"NumFunctions", i64 3836}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13, !14, !15, !16, !16, !17, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26}
!11 = !{i32 10000, i64 629805108, i32 1}
!12 = !{i32 100000, i64 366853677, i32 2}
!13 = !{i32 200000, i64 196816893, i32 4}
!14 = !{i32 300000, i64 192575561, i32 7}
!15 = !{i32 400000, i64 130688163, i32 11}
!16 = !{i32 500000, i64 74857169, i32 19}
!17 = !{i32 600000, i64 48184151, i32 30}
!18 = !{i32 700000, i64 21298588, i32 49}
!19 = !{i32 800000, i64 10721033, i32 90}
!20 = !{i32 900000, i64 3301634, i32 202}
!21 = !{i32 950000, i64 1454952, i32 362}
!22 = !{i32 990000, i64 343872, i32 675}
!23 = !{i32 999000, i64 46009, i32 1112}
!24 = !{i32 999900, i64 6067, i32 1435}
!25 = !{i32 999990, i64 700, i32 1721}
!26 = !{i32 999999, i64 72, i32 1955}
!27 = !{!"function_entry_count", i64 700}
!28 = !{!"branch_weights", i32 701, i32 1}
!29 = !{!"branch_weights", i32 954001, i32 701}
!30 = !{!"branch_weights", i32 1, i32 954001}
