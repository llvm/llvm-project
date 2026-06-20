; This test verifies that indirect call promotion (ICP) under sample profiling
; correctly checks target feature compatibility. Specifically, a callee with
; target features incompatible with the caller's features (e.g., "_Z3moov" requiring
; "+avx512f" while caller "_Z3goov" does not support it) should not be promoted,
; while a compatible callee (e.g., "_Z3hoov") should be promoted successfully.
; Note that under Sample PGO, the promotion candidates (e.g., _Z3hoov and _Z3moov)
; are retrieved directly from the sample profile file (Inputs/norepeated-icp-2.prof)
; at the corresponding line offset (1) rather than using value profile !prof metadata.
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/norepeated-icp-2.prof -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@p = dso_local global ptr null, align 8

; Function Attrs: uwtable mustprogress
define dso_local void @_Z3moov() #0 !dbg !7 {
entry:
  ret void
}

; Function Attrs: uwtable mustprogress
define dso_local void @_Z3hoov() #1 !dbg !11 {
entry:
  store ptr @_Z3moov, ptr @p, align 8
  ret void
}

; Function Attrs: uwtable mustprogress
define dso_local void @_Z3goov() #1 !dbg !24 {
entry:
  %t0 = load ptr, ptr @p, align 8
  ; Here, _Z3moov requires "+avx512f" (attributes #0) which the caller lacks (attributes #1).
  ; Therefore, _Z3moov is not promoted.
  ; CHECK-NOT: icmp eq ptr %t0, @_Z3moov
  ; On the other hand, _Z3hoov has compatible target features and is promoted successfully.
  ; CHECK: icmp eq ptr %t0, @_Z3hoov
  ; CHECK-NOT: icmp eq ptr %t0, @_Z3moov
  call void %t0(), !dbg !26
  ret void
}

attributes #0 = { uwtable mustprogress "use-sample-profile" "target-features"="+avx512f" }
attributes #1 = { uwtable mustprogress "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "1.cc", directory: "")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "moo", linkageName: "_Z3moov", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!11 = distinct !DISubprogram(name: "hoo", linkageName: "_Z3hoov", scope: !1, file: !1, line: 9, type: !8, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!24 = distinct !DISubprogram(name: "goo", linkageName: "_Z3goov", scope: !1, file: !1, line: 15, type: !8, scopeLine: 15, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!26 = !DILocation(line: 16, column: 3, scope: !24)
