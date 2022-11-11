; RUN: opt < %s -passes='require<profile-summary>,cgscc(inline)' -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@p = dso_local global ptr null, align 8

; After _Z3goov is inlined into _Z3foov, the value profile of the indirect
; call in _Z3goov and _Z3foov need to be scaled. The test is to make sure
; the magic number NOMORE_ICP_MAGICNUM used for prevent recursive indirect
; call will be kept during the scaling.
;
; CHECK-LABEL: @_Z3goov(
; CHECK: call void %t0(), {{.*}} !prof ![[PROF_ID1:[0-9]+]]
; CHECK-NEXT: ret void
;
; CHECK-LABEL: @_Z3foov(
; CHECK: call void %t0.i(), {{.*}} !prof ![[PROF_ID2:[0-9]+]]
; CHECK-NEXT: ret void
;
; Function Attrs: uwtable mustprogress
define dso_local void @_Z3goov() #0 !dbg !11 !prof !23 {
entry:
  %t0 = load ptr, ptr @p, align 8, !dbg !12, !tbaa !13
  call void %t0(), !dbg !17, !prof !22
  ret void, !dbg !18
}

; Function Attrs: uwtable mustprogress
define dso_local void @_Z3foov() #0 !dbg !19 {
entry:
  call void @_Z3goov(), !dbg !20, !prof !24
  ret void, !dbg !21
}

attributes #0 = { uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-sample-profile" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !25}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "1.cc", directory: "")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!""}
!8 = !DISubroutineType(types: !2)
!11 = distinct !DISubprogram(name: "goo", linkageName: "_Z3goov", scope: !1, file: !1, line: 6, type: !8, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!12 = !DILocation(line: 7, column: 5, scope: !11)
!13 = !{!14, !14, i64 0}
!14 = !{!"any pointer", !15, i64 0}
!15 = !{!"omnipotent char", !16, i64 0}
!16 = !{!"Simple C++ TBAA"}
!17 = !DILocation(line: 7, column: 3, scope: !11)
!18 = !DILocation(line: 8, column: 1, scope: !11)
!19 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 10, type: !8, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!20 = !DILocation(line: 11, column: 3, scope: !19)
!21 = !DILocation(line: 12, column: 3, scope: !19)
; CHECK: ![[PROF_ID1]] = !{!"VP", i32 0, i64 7200, i64 -7383239051784516332, i64 -1, i64 -3834823603621627078, i64 7200}
; CHECK: ![[PROF_ID2]] = !{!"VP", i32 0, i64 800, i64 -7383239051784516332, i64 -1, i64 -3834823603621627078, i64 800}
!22 = !{!"VP", i32 0, i64 8000, i64 -7383239051784516332, i64 -1, i64 125292384912345234234, i64 8000}
!23 = !{!"function_entry_count", i64 1000} 
!24 = !{!"branch_weights", i32 100}
!25 = !{i32 1, !"ProfileSummary", !26}
!26 = !{!27, !28, !29, !30, !31, !32, !33, !34}
!27 = !{!"ProfileFormat", !"SampleProfile"}
!28 = !{!"TotalCount", i64 10000}
!29 = !{!"MaxCount", i64 1000}
!30 = !{!"MaxInternalCount", i64 1}
!31 = !{!"MaxFunctionCount", i64 1000}
!32 = !{!"NumCounts", i64 3}
!33 = !{!"NumFunctions", i64 3}
!34 = !{!"DetailedSummary", !35}
!35 = !{!36, !37, !38}
!36 = !{i32 10000, i64 100, i32 1}
!37 = !{i32 999000, i64 100, i32 1}
!38 = !{i32 999999, i64 1, i32 2}
