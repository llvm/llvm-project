; The pre-inliner can keep a local function's profile top-level (R below) while
; ThinLTO pre-link still inlines it into its single caller (G) and deletes it.
; R's nested decision to inline ext, defined in another module, then has no
; owner: its import GUID is attached to R and deleted with it, and in post-link
; G's profile has no context for R. With -sample-profile-orphaned-local-roots,
; pre-link copies R's import GUIDs to G, and post-link resolves frames inlined
; from R using R's top-level profile.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt pre.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -sample-profile-file=prof.txt -S | FileCheck %s --check-prefix=PRE-OFF
; RUN: opt pre.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -sample-profile-file=prof.txt -sample-profile-orphaned-local-roots -S \
; RUN:   | FileCheck %s --check-prefix=PRE-ON
;; R survives when it is not local, so it keeps its own import GUIDs.
; RUN: sed 's/define internal i64 @R(/define dso_local i64 @R(/' pre.ll > pre-extern.ll
; RUN: opt pre-extern.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -sample-profile-file=prof.txt -sample-profile-orphaned-local-roots -S \
; RUN:   | FileCheck %s --check-prefix=PRE-EXTERN

; RUN: opt post.ll -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -sample-profile-file=prof.txt -pass-remarks=sample-profile-inline -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=POST-OFF --allow-empty
; RUN: opt post.ll -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -sample-profile-file=prof.txt -sample-profile-orphaned-local-roots \
; RUN:   -pass-remarks=sample-profile-inline -o /dev/null 2>&1 | FileCheck %s --check-prefix=POST-ON
;; Only a local frame can be orphaned: a non-local function may have an
;; out-of-line copy elsewhere that uses its top-level profile.
; RUN: sed 's/spFlags: DISPFlagLocalToUnit | DISPFlagDefinition/spFlags: DISPFlagDefinition/' \
; RUN:   post.ll > post-nonlocal.ll
; RUN: opt post-nonlocal.ll -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -sample-profile-file=prof.txt -sample-profile-orphaned-local-roots \
; RUN:   -pass-remarks=sample-profile-inline -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=POST-OFF --allow-empty

;; If the call site does not account for all entries of R, R may have other
;; inlined copies, so this copy must not get all of R's samples.
; RUN: sed 's/^R:4600000:100000$/R:4600000:200000/' prof.txt > prof-multi.txt
; RUN: opt post.ll -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -sample-profile-file=prof-multi.txt -sample-profile-orphaned-local-roots \
; RUN:   -pass-remarks=sample-profile-inline -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=POST-OFF --allow-empty
;; A top-level profile whose checksum does not match R is not used.
; RUN: sed '15s/281479271677951/1/' prof.txt > prof-stale.txt
; RUN: opt post.ll -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -sample-profile-file=prof-stale.txt -sample-profile-orphaned-local-roots \
; RUN:   -pass-remarks=sample-profile-inline -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=POST-OFF --allow-empty

; PRE-OFF: define {{.*}} @G({{.*}} !prof ![[#GPROF:]]
; PRE-OFF: ![[#GPROF]] = !{!"function_entry_count", i64 100000}

; PRE-ON: define {{.*}} @G({{.*}} !prof ![[#GPROF:]]
; PRE-ON-NOT: @R(
; PRE-ON: ![[#GPROF]] = !{!"function_entry_count", i64 100000, i64 -2953423517159131221}

; PRE-EXTERN: define {{.*}} @G({{.*}} !prof ![[#GPROF:]]
; PRE-EXTERN: define {{.*}} @R({{.*}} !prof ![[#RPROF:]]
; PRE-EXTERN-DAG: ![[#GPROF]] = !{!"function_entry_count", i64 100000}
; PRE-EXTERN-DAG: ![[#RPROF]] = !{!"function_entry_count", i64 100000, i64 -2953423517159131221}

; POST-OFF-NOT: 'ext' inlined into 'G'
; POST-ON: remark: a.c:2:47: 'ext' inlined into 'G' to match profiling context with (cost=always): preinliner

;--- prof.txt
G:200000:100000
 1: 100000
 2: 100000 R:100000
 !CFGChecksum: 281479271677951
R:4600000:100000
 1: 100000
 2: ext:4500000
  1: 100000
  2: 1000000
  3: 1000000
  4: 1000000
  5: 100000
  !CFGChecksum: 88680961901
  !Attributes: 2
 !CFGChecksum: 281479271677951

;--- pre.ll
source_filename = "a.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local i64 @G(ptr noundef %v, i64 noundef %n) #0 !dbg !10 {
entry:
  %v.addr = alloca ptr, align 8
  %n.addr = alloca i64, align 8
  store ptr %v, ptr %v.addr, align 8
  store i64 %n, ptr %n.addr, align 8
  call void @llvm.pseudoprobe(i64 -6455552227143004193, i64 1, i32 0, i64 -1), !dbg !13
  %0 = load ptr, ptr %v.addr, align 8, !dbg !13
  %1 = load i64, ptr %n.addr, align 8, !dbg !14
  %call = call i64 @R(ptr noundef %0, i64 noundef %1), !dbg !15
  %add = add nsw i64 %call, 1, !dbg !17
  ret i64 %add, !dbg !18
}

define internal i64 @R(ptr noundef %v, i64 noundef %n) #0 !dbg !19 {
entry:
  %v.addr = alloca ptr, align 8
  %n.addr = alloca i64, align 8
  store ptr %v, ptr %v.addr, align 8
  store i64 %n, ptr %n.addr, align 8
  call void @llvm.pseudoprobe(i64 9084449867242398177, i64 1, i32 0, i64 -1), !dbg !20
  %0 = load ptr, ptr %v.addr, align 8, !dbg !20
  %1 = load i64, ptr %n.addr, align 8, !dbg !21
  %call = call i64 @ext(ptr noundef %0, i64 noundef %1), !dbg !22
  %xor = xor i64 %call, 85, !dbg !24
  ret i64 %xor, !dbg !25
}

declare dso_local i64 @ext(ptr noundef, i64 noundef) #1

declare void @llvm.pseudoprobe(i64 immarg, i64 immarg, i32 immarg, i64 immarg) #2

attributes #0 = { nounwind uwtable "use-sample-profile" }
attributes #1 = { nounwind }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.pseudo_probe_desc = !{!8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "a.c", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 2, !"debugger-tuning", !"gdb"}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{!"clang version 24.0.0git (https://git.internal.tfbnw.net/repos/git/rw/osmeta/external/llvm-project 361dea8dbc903d0c906651c9c54daed779976243)"}
!8 = !{i64 -6455552227143004193, i64 281479271677951, !"G"}
!9 = !{i64 9084449867242398177, i64 281479271677951, !"R"}
!10 = distinct !DISubprogram(name: "G", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !DILocation(line: 3, column: 68, scope: !10)
!14 = !DILocation(line: 3, column: 71, scope: !10)
!15 = !DILocation(line: 3, column: 66, scope: !16)
!16 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 455082007)
!17 = !DILocation(line: 3, column: 74, scope: !10)
!18 = !DILocation(line: 3, column: 59, scope: !10)
!19 = distinct !DISubprogram(name: "R", scope: !1, file: !1, line: 2, type: !11, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0)
!20 = !DILocation(line: 2, column: 51, scope: !19)
!21 = !DILocation(line: 2, column: 54, scope: !19)
!22 = !DILocation(line: 2, column: 47, scope: !23)
!23 = !DILexicalBlockFile(scope: !19, file: !1, discriminator: 455082007)
!24 = !DILocation(line: 2, column: 57, scope: !19)
!25 = !DILocation(line: 2, column: 40, scope: !19)

;--- post.ll
source_filename = "llvm-link"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local range(i64 -9223372036854775807, -9223372036854775808) i64 @G(ptr noundef %v, i64 noundef %n) local_unnamed_addr #0 !dbg !41 !prof !44 !guid !45 {
entry:
  call void @llvm.pseudoprobe(i64 -6455552227143004193, i64 1, i32 0, i64 -1), !dbg !46
  call void @llvm.pseudoprobe(i64 9084449867242398177, i64 1, i32 0, i64 -1), !dbg !47
  %call.i = tail call i64 @ext(ptr noundef %v, i64 noundef %n), !dbg !51, !prof !53
  %xor.i = xor i64 %call.i, 85, !dbg !54
  %add = add nsw i64 %xor.i, 1, !dbg !55
  ret i64 %add, !dbg !56
}

declare void @llvm.pseudoprobe(i64 immarg, i64 immarg, i32 immarg, i64 immarg) #1

define available_externally i64 @ext(ptr nofree noundef readonly captures(none) %v, i64 noundef %n) local_unnamed_addr #2 !dbg !57 !prof !58 !guid !59 {
entry:
  call void @llvm.pseudoprobe(i64 -2953423517159131221, i64 1, i32 0, i64 -1), !dbg !60
  call void @llvm.pseudoprobe(i64 -2953423517159131221, i64 2, i32 0, i64 -1), !dbg !61
  %cmp4 = icmp sgt i64 %n, 0, !dbg !62
  br i1 %cmp4, label %for.body, label %for.end, !dbg !64

for.body:                                         ; preds = %for.body, %entry
  %i.06 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %r.05 = phi i64 [ %add, %for.body ], [ 0, %entry ]
  call void @llvm.pseudoprobe(i64 -2953423517159131221, i64 3, i32 0, i64 -1), !dbg !65
  %arrayidx = getelementptr inbounds nuw [8 x i8], ptr %v, i64 %i.06, !dbg !65
  %0 = load i64, ptr %arrayidx, align 8, !dbg !65
  %add = add nsw i64 %0, %r.05, !dbg !66
  call void @llvm.pseudoprobe(i64 -2953423517159131221, i64 4, i32 0, i64 -1), !dbg !67
  %inc = add nuw nsw i64 %i.06, 1, !dbg !68
  call void @llvm.pseudoprobe(i64 -2953423517159131221, i64 2, i32 0, i64 -1), !dbg !61
  %exitcond.not = icmp eq i64 %inc, %n, !dbg !62
  br i1 %exitcond.not, label %for.end, label %for.body, !dbg !64, !llvm.loop !70

for.end:                                          ; preds = %for.body, %entry
  %r.0.lcssa = phi i64 [ 0, %entry ], [ %add, %for.body ], !dbg !74
  call void @llvm.pseudoprobe(i64 -2953423517159131221, i64 5, i32 0, i64 -1), !dbg !75
  ret i64 %r.0.lcssa, !dbg !76
}

attributes #0 = { nounwind uwtable "use-sample-profile" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #2 = { nofree norecurse nosync nounwind memory(argmem: read) uwtable "use-sample-profile" }

!llvm.dbg.cu = !{!0, !2}
!llvm.pseudo_probe_desc = !{!4, !5, !6}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "a.c", directory: "")
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "b.c", directory: "")
!4 = !{i64 -6455552227143004193, i64 281479271677951, !"G"}
!5 = !{i64 9084449867242398177, i64 281479271677951, !"R"}
!6 = !{i64 -2953423517159131221, i64 88680961901, !"ext"}
!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 2, !"debugger-tuning", !"gdb"}
!10 = !{i32 7, !"uwtable", i32 2}
!11 = !{i32 7, !"frame-pointer", i32 2}
!12 = !{i32 1, !"ProfileSummary", !13}
!13 = !{!14, !15, !16, !17, !18, !19, !20, !21, !22, !23}
!14 = !{!"ProfileFormat", !"SampleProfile"}
!15 = !{!"TotalCount", i64 3500000}
!16 = !{!"MaxCount", i64 1000000}
!17 = !{!"MaxInternalCount", i64 0}
!18 = !{!"MaxFunctionCount", i64 100000}
!19 = !{!"NumCounts", i64 8}
!20 = !{!"NumFunctions", i64 2}
!21 = !{!"IsPartialProfile", i64 0}
!22 = !{!"PartialProfileRatio", double 0.000000e+00}
!23 = !{!"DetailedSummary", !24}
!24 = !{!25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40}
!25 = !{i32 10000, i64 1000000, i32 3}
!26 = !{i32 100000, i64 1000000, i32 3}
!27 = !{i32 200000, i64 1000000, i32 3}
!28 = !{i32 300000, i64 1000000, i32 3}
!29 = !{i32 400000, i64 1000000, i32 3}
!30 = !{i32 500000, i64 1000000, i32 3}
!31 = !{i32 600000, i64 1000000, i32 3}
!32 = !{i32 700000, i64 1000000, i32 3}
!33 = !{i32 800000, i64 1000000, i32 3}
!34 = !{i32 900000, i64 100000, i32 8}
!35 = !{i32 950000, i64 100000, i32 8}
!36 = !{i32 990000, i64 100000, i32 8}
!37 = !{i32 999000, i64 100000, i32 8}
!38 = !{i32 999900, i64 100000, i32 8}
!39 = !{i32 999990, i64 100000, i32 8}
!40 = !{i32 999999, i64 100000, i32 8}
!41 = distinct !DISubprogram(name: "G", scope: !1, file: !1, line: 3, type: !42, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!42 = !DISubroutineType(types: !43)
!43 = !{}
!44 = !{!"function_entry_count", i64 100000}
!45 = !{i64 -6455552227143004193}
!46 = !DILocation(line: 3, column: 68, scope: !41)
!47 = distinct !DILocation(line: 2, column: 51, scope: !48, inlinedAt: !49)
!48 = distinct !DISubprogram(name: "R", scope: !1, file: !1, line: 2, type: !42, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0)
!49 = distinct !DILocation(line: 3, column: 66, scope: !50)
!50 = !DILexicalBlockFile(scope: !41, file: !1, discriminator: 455082007)
!51 = distinct !DILocation(line: 2, column: 47, scope: !52, inlinedAt: !49)
!52 = !DILexicalBlockFile(scope: !48, file: !1, discriminator: 455082007)
!53 = !{!"branch_weights", i32 100000}
!54 = distinct !DILocation(line: 2, column: 57, scope: !48, inlinedAt: !49)
!55 = !DILocation(line: 3, column: 74, scope: !41)
!56 = !DILocation(line: 3, column: 59, scope: !41)
!57 = distinct !DISubprogram(name: "ext", scope: !3, file: !3, line: 1, type: !42, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!58 = !{!"function_entry_count", i64 -1}
!59 = !{i64 -2953423517159131221}
!60 = !DILocation(line: 2, column: 8, scope: !57)
!61 = !DILocation(line: 3, column: 20, scope: !57)
!62 = !DILocation(line: 3, column: 22, scope: !63)
!63 = !DILexicalBlockFile(scope: !57, file: !3, discriminator: 2)
!64 = !DILocation(line: 3, column: 3, scope: !63)
!65 = !DILocation(line: 4, column: 10, scope: !57)
!66 = !DILocation(line: 4, column: 7, scope: !57)
!67 = !DILocation(line: 3, column: 27, scope: !57)
!68 = !DILocation(line: 3, column: 27, scope: !69)
!69 = !DILexicalBlockFile(scope: !57, file: !3, discriminator: 4)
!70 = distinct !{!70, !71, !72, !73}
!71 = !DILocation(line: 3, column: 3, scope: !57)
!72 = !DILocation(line: 4, column: 13, scope: !57)
!73 = !{!"llvm.loop.mustprogress"}
!74 = !DILocation(line: 0, scope: !57)
!75 = !DILocation(line: 5, column: 10, scope: !57)
!76 = !DILocation(line: 5, column: 3, scope: !57)
