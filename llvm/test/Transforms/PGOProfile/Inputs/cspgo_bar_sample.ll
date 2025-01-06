target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$__llvm_profile_filename = comdat any
$__llvm_profile_raw_version = comdat any
$__llvm_profile_sampling = comdat any

@odd = common dso_local local_unnamed_addr global i32 0, align 4
@even = common dso_local local_unnamed_addr global i32 0, align 4
@__llvm_profile_filename = local_unnamed_addr constant [25 x i8] c"pass2/default_%m.profraw\00", comdat
@__llvm_profile_raw_version = local_unnamed_addr constant i64 216172782113783812, comdat
@__llvm_profile_sampling = thread_local global i16 0, comdat
@llvm.used = appending global [1 x i8*] [i8* bitcast (i64* @__llvm_profile_sampling to i8*)], section "llvm.metadata"

define dso_local void @bar(i32 %n) !prof !30 {
entry:
  %call = tail call fastcc i32 @cond(i32 %n)
  %tobool = icmp eq i32 %call, 0
  br i1 %tobool, label %if.else, label %if.then, !prof !31

if.then:
  %0 = load i32, i32* @odd, align 4, !tbaa !32
  %inc = add i32 %0, 1
  store i32 %inc, i32* @odd, align 4, !tbaa !32
  br label %if.end

if.else:
  %1 = load i32, i32* @even, align 4, !tbaa !32
  %inc1 = add i32 %1, 1
  store i32 %inc1, i32* @even, align 4, !tbaa !32
  br label %if.end

if.end:
  ret void
}

define internal fastcc i32 @cond(i32 %i) #1 !prof !30 !PGOFuncName !36 {
entry:
  %rem = srem i32 %i, 2
  ret i32 %rem
}

attributes #1 = { inlinehint noinline }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"EnableSplitLTOUnit", i32 0}
!2 = !{i32 1, !"ProfileSummary", !3}
!3 = !{!4, !5, !6, !7, !8, !9, !10, !11}
!4 = !{!"ProfileFormat", !"InstrProf"}
!5 = !{!"TotalCount", i64 500002}
!6 = !{!"MaxCount", i64 200000}
!7 = !{!"MaxInternalCount", i64 100000}
!8 = !{!"MaxFunctionCount", i64 200000}
!9 = !{!"NumCounts", i64 6}
!10 = !{!"NumFunctions", i64 4}
!11 = !{!"DetailedSummary", !12}
!12 = !{!13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28}
!13 = !{i32 10000, i64 200000, i32 1}
!14 = !{i32 100000, i64 200000, i32 1}
!15 = !{i32 200000, i64 200000, i32 1}
!16 = !{i32 300000, i64 200000, i32 1}
!17 = !{i32 400000, i64 200000, i32 1}
!18 = !{i32 500000, i64 100000, i32 4}
!19 = !{i32 600000, i64 100000, i32 4}
!20 = !{i32 700000, i64 100000, i32 4}
!21 = !{i32 800000, i64 100000, i32 4}
!22 = !{i32 900000, i64 100000, i32 4}
!23 = !{i32 950000, i64 100000, i32 4}
!24 = !{i32 990000, i64 100000, i32 4}
!25 = !{i32 999000, i64 100000, i32 4}
!26 = !{i32 999900, i64 100000, i32 4}
!27 = !{i32 999990, i64 100000, i32 4}
!28 = !{i32 999999, i64 1, i32 6}
!30 = !{!"function_entry_count", i64 200000}
!31 = !{!"branch_weights", i32 100000, i32 100000}
!32 = !{!33, !33, i64 0}
!33 = !{!"int", !34, i64 0}
!34 = !{!"omnipotent char", !35, i64 0}
!35 = !{!"Simple C/C++ TBAA"}
!36 = !{!"cspgo_bar.c:cond"}
