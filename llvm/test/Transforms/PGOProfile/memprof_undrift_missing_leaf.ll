;; Tests memprof undrifting when the leaf frame is missing in the profile.
;; This test case is taken from memprof_missing_leaf.ll with the profile
;; drifted.

; RUN: split-file %s %t
; RUN: llvm-profdata merge %t/memprof_missing_leaf.yaml -o %t/memprof_missing_leaf.memprofdata
; RUN: opt < %t/memprof_missing_leaf.ll -passes='memprof-use<profile-filename=%t/memprof_missing_leaf.memprofdata>' -memprof-salvage-stale-profile -S | FileCheck %s

;--- memprof_missing_leaf.yaml
---
HeapProfileRecords:
  - GUID:            main
    AllocSites:
      - Callstack:
          - { Function: main, LineOffset: 2, Column: 21, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       1
    CallSites:       []
...
;--- memprof_missing_leaf.ll
; CHECK: call {{.*}} @_Znam{{.*}} #[[ATTR:[0-9]+]]
; CHECK: attributes #[[ATTR]] = {{.*}} "memprof"="notcold"

; ModuleID = '<stdin>'
source_filename = "memprof_missing_leaf.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) #0

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() #1 !dbg !8 {
entry:
  %s.addr.i = alloca i64, align 8
  %retval = alloca i32, align 4
  %a = alloca ptr, align 8
  store i32 0, ptr %retval, align 4
  store i64 1, ptr %s.addr.i, align 8, !tbaa !11
  %0 = load i64, ptr %s.addr.i, align 8, !dbg !15, !tbaa !11
  %call.i = call noalias noundef nonnull ptr @_Znam(i64 noundef %0) #3, !dbg !18
  store ptr %call.i, ptr %a, align 8, !dbg !19, !tbaa !20
  %1 = load ptr, ptr %a, align 8, !dbg !22, !tbaa !20
  %isnull = icmp eq ptr %1, null, !dbg !23
  br i1 %isnull, label %delete.end, label %delete.notnull, !dbg !23

delete.notnull:                                   ; preds = %entry
  call void @_ZdlPv(ptr noundef %1) #4, !dbg !23
  br label %delete.end, !dbg !23

delete.end:                                       ; preds = %delete.notnull, %entry
  ret i32 0, !dbg !24
}

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(ptr noundef) #2

attributes #0 = { nobuiltin allocsize(0) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress norecurse uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nobuiltin nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { builtin allocsize(0) }
attributes #4 = { builtin nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 18.0.0 (git@github.com:llvm/llvm-project.git 71bf052ec90e77cb4aa66505d47cbc4b6016ac1d)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "memprof_missing_leaf.cc", directory: ".", checksumkind: CSK_MD5, checksum: "f1445a8699406a6b826128704d257677")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 15, type: !9, scopeLine: 15, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{}
!11 = !{!12, !12, i64 0}
!12 = !{!"long", !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C++ TBAA"}
!15 = !DILocation(line: 11, column: 19, scope: !16, inlinedAt: !17)
!16 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barm", scope: !1, file: !1, line: 7, type: !9, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!17 = distinct !DILocation(line: 16, column: 21, scope: !8)
!18 = !DILocation(line: 11, column: 10, scope: !16, inlinedAt: !17)
!19 = !DILocation(line: 16, column: 9, scope: !8)
!20 = !{!21, !21, i64 0}
!21 = !{!"any pointer", !13, i64 0}
!22 = !DILocation(line: 17, column: 10, scope: !8)
!23 = !DILocation(line: 17, column: 3, scope: !8)
!24 = !DILocation(line: 18, column: 3, scope: !8)
