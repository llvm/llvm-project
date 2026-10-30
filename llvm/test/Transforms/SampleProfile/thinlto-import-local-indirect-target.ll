; In ThinLTO pre-link the sample loader defers promotion of indirect calls to
; the backend. When the caller is imported into another module, its callees
; are imported by following its summary edges. Check that
; -sample-profile-import-local-indirect-targets adds hot profiled targets of a
; deferred indirect call that are defined in this module, and that the caller's
; summary then imports them together with the caller.
;
; dispatch makes an indirect call at probe 2 with profiled targets hot,
; local_hot (internal linkage), rare (2 samples), pre (2 samples, should be
; inlined) and zero (0 samples), and residual value-profile target cold.
; dispatch2 also calls `called` directly. That call is inlined before summary
; generation; its deferred indirect target still needs an import hint.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llvm-profdata merge --sample --extbinary nested.txt -o nested.prof
; RUN: llvm-profdata merge --sample --extbinary cs.txt -o cs.prof
; RUN: opt dispatch.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -profile-summary-cold-count=10 \
; RUN:   -sample-profile-file=nested.prof -S | FileCheck %s --check-prefix=OFF

;; Pre-inlined nested profile: the pre-inliner decision sets the threshold to
;; zero, as findInlinedFunctions does for out-of-module targets, so rare and pre
;; are added; zero is not.
; RUN: opt dispatch.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -profile-summary-cold-count=10 \
; RUN:   -sample-profile-file=nested.prof -sample-profile-import-local-indirect-targets \
; RUN:   -S | FileCheck %s --check-prefix=NESTED
;; A stored GUID can differ from the current name's hash. Preserve that
;; identity in the import hint and in the resulting summary edge.
; RUN: sed '/^define.* @hot(/s/ {$/ !guid !900 {/' dispatch.ll > assigned-guid.ll
; RUN: echo '!900 = !{i64 42}' >> assigned-guid.ll
; RUN: opt assigned-guid.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -profile-summary-cold-count=10 \
; RUN:   -sample-profile-file=nested.prof -sample-profile-import-local-indirect-targets \
; RUN:   -thinlto-bc -o assigned-guid.bc
; RUN: llvm-dis assigned-guid.bc -o - | FileCheck %s --check-prefix=ASSIGNED

;; Exercise the non-prioritized sample-loader path as well.
; RUN: opt dispatch.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -profile-summary-cold-count=10 \
; RUN:   -sample-profile-file=nested.prof -sample-profile-import-local-indirect-targets \
; RUN:   -sample-profile-prioritized-inline=false -S | FileCheck %s --check-prefix=NESTED
;; Without the pre-inliner decision only total samples above the hot threshold count.
; RUN: opt dispatch.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -profile-summary-cold-count=10 \
; RUN:   -sample-profile-file=nested.prof -sample-profile-import-local-indirect-targets \
; RUN:   -sample-profile-use-preinliner=false -S | FileCheck %s --check-prefix=HOT
;; Context-sensitive profile: head samples above the hot threshold, or, with
;; the pre-inliner decision (the default for this pre-inlined profile), contexts
;; marked as should-be-inlined.
; RUN: opt dispatch.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -profile-summary-cold-count=10 \
; RUN:   -sample-profile-file=cs.prof -sample-profile-import-local-indirect-targets \
; RUN:   -S | FileCheck %s --check-prefix=CSPRE
; RUN: opt dispatch.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -profile-summary-cold-count=10 \
; RUN:   -sample-profile-file=cs.prof -sample-profile-import-local-indirect-targets \
; RUN:   -sample-profile-use-preinliner=false -S | FileCheck %s --check-prefix=HOT

;; Import the caller's callees into caller.ll.
; RUN: opt caller.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -profile-summary-cold-count=10 \
; RUN:   -sample-profile-file=nested.prof -thinlto-bc -o caller.bc
; RUN: opt dispatch.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -profile-summary-cold-count=10 \
; RUN:   -sample-profile-file=nested.prof -thinlto-bc -o dispatch-off.bc
; RUN: opt dispatch.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -profile-summary-cold-count=10 \
; RUN:   -sample-profile-file=nested.prof -sample-profile-import-local-indirect-targets \
; RUN:   -thinlto-bc -o dispatch-on.bc
; RUN: llvm-lto -thinlto-action=thinlink -o index-off.bc dispatch-off.bc caller.bc
; RUN: llvm-lto -thinlto-action=thinlink -o index-on.bc dispatch-on.bc caller.bc
; RUN: opt -passes=function-import -summary-file=index-off.bc caller.bc -print-imports \
; RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=IMPORT-OFF
; RUN: opt -passes=function-import -summary-file=index-on.bc caller.bc -print-imports \
; RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=IMPORT-ON
;; Exactly the imports listed below, no others.
; RUN: opt -passes=function-import -summary-file=index-off.bc caller.bc -print-imports \
; RUN:   -o /dev/null 2>&1 | grep Import | count 3
; RUN: opt -passes=function-import -summary-file=index-on.bc caller.bc -print-imports \
; RUN:   -o /dev/null 2>&1 | grep Import | count 8

;; Local import hints are only added for deferred indirect calls. Making the
;; calls direct must not add hints, even with the option enabled.
; RUN: sed -e 's/call noundef i32 %5(/call noundef i32 @hot(/g' -e 's/@hot(i32 noundef %0) #0/@hot(i32 noundef %0) noinline #0/' dispatch.ll > direct.ll
; RUN: opt direct.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -profile-summary-cold-count=10 \
; RUN:   -sample-profile-file=nested.prof -sample-profile-import-local-indirect-targets \
; RUN:   -S | FileCheck %s --check-prefix=DIRECT
; RUN: opt direct.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -profile-summary-cold-count=10 \
; RUN:   -sample-profile-file=nested.prof -sample-profile-import-local-indirect-targets \
; RUN:   -sample-profile-prioritized-inline=false -S | FileCheck %s --check-prefix=DIRECT

; DIRECT: define {{.*}} @dispatch({{.*}} !prof ![[#D:]]
; DIRECT: call {{.*}} @hot(
; DIRECT: ![[#D]] = !{!"function_entry_count", i64 1000000}

;; A retained direct call must not create duplicate summary edges. Its
;; noinline target remains ineligible for body import even with the hint.
; RUN: sed 's/@called(i32 noundef %0) #0/@called(i32 noundef %0) noinline #0/' dispatch.ll > dispatch-retained.ll
; RUN: opt dispatch-retained.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -profile-summary-cold-count=10 \
; RUN:   -sample-profile-file=nested.prof -sample-profile-import-local-indirect-targets \
; RUN:   -thinlto-bc -o dispatch-retained.bc
; RUN: llvm-dis dispatch-retained.bc -o - | FileCheck %s --check-prefix=RETAINED
; RUN: llvm-lto -thinlto-action=thinlink -o index-retained.bc dispatch-retained.bc caller.bc
; RUN: opt -passes=function-import -summary-file=index-retained.bc caller.bc -print-imports \
; RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=IMPORT-RETAINED
; RUN: opt -passes=function-import -summary-file=index-retained.bc caller.bc -print-imports \
; RUN:   -o /dev/null 2>&1 | grep Import | count 7

; RETAINED-LABEL: define {{.*}} @dispatch2(
; RETAINED: call {{.*}} @called(
; RETAINED: name: "dispatch2", {{.*}}calls: ((callee: ^[[#CALLED:]], hotness: critical, tail: 1))
; RETAINED: ^[[#CALLED]] = gv: (name: "called",
; IMPORT-RETAINED-NOT: Import called from
; IMPORT-RETAINED: Import
; IMPORT-RETAINED-NOT: Import called from

;; GUIDs (unsigned, as sorted in the metadata): local_hot 5110348091542198759
;; (includes the source file name), hot 10177652421713147431,
;; pre 13441716699865282923, rare 16988177133866688935.

;; Both functions share one uniqued entry-count node.
; OFF:     define {{.*}} @dispatch({{.*}} !prof ![[#D:]]
; OFF:     define {{.*}} @dispatch2({{.*}} !prof ![[#D]]
; OFF:     ![[#D]] = !{!"function_entry_count", i64 1000000}
; OFF-NOT: !"function_entry_count", i64 {{[0-9]+}}, i64

; ASSIGNED: define {{.*}} @dispatch({{.*}} !prof ![[#AD:]]
; ASSIGNED: ![[#AD]] = !{!"function_entry_count", i64 1000000, i64 42, i64 5110348091542198759, i64 -5005027373844268693, i64 -1458566939842862681}
; ASSIGNED: ^[[#ASSIGNED_TARGET:]] = gv: (name: "hot", {{.*}}; guid = 42
; ASSIGNED: name: "dispatch", {{.*}}callee: ^[[#ASSIGNED_TARGET]], hotness: critical

; NESTED: define {{.*}} @dispatch({{.*}} !prof ![[#D:]]
; NESTED: define {{.*}} @dispatch2({{.*}} !prof ![[#D2:]]
; NESTED-NOT: call {{.*}} @called(
; NESTED: ![[#D]] = !{!"function_entry_count", i64 1000000, i64 5110348091542198759, i64 -8269091651996404185, i64 -5005027373844268693, i64 -1458566939842862681}
; NESTED: ![[#D2]] = !{!"function_entry_count", i64 1000000, i64 -8705695290094177457}

; HOT:    define {{.*}} @dispatch({{.*}} !prof ![[#D:]]
; HOT:    define {{.*}} @dispatch2({{.*}} !prof ![[#D2:]]
; HOT:    ![[#D]] = !{!"function_entry_count", i64 1000000, i64 5110348091542198759, i64 -8269091651996404185}
; HOT:    ![[#D2]] = !{!"function_entry_count", i64 1000000, i64 -8705695290094177457}

; CSPRE:  define {{.*}} @dispatch({{.*}} !prof ![[#D:]]
; CSPRE:  define {{.*}} @dispatch2({{.*}} !prof ![[#D2:]]
; CSPRE:  ![[#D]] = !{!"function_entry_count", i64 1000000, i64 5110348091542198759, i64 -8269091651996404185, i64 -5005027373844268693}
; CSPRE:  ![[#D2]] = !{!"function_entry_count", i64 1000000, i64 -8705695290094177457}

;; cold is imported in both cases through the value-profile edge; zero is
;; never imported. The hint preserves called after its direct call disappears.
; IMPORT-OFF-DAG: Import dispatch from
; IMPORT-OFF-DAG: Import dispatch2 from
; IMPORT-OFF-DAG: Import cold from

; IMPORT-ON-DAG:  Import dispatch from
; IMPORT-ON-DAG:  Import dispatch2 from
; IMPORT-ON-DAG:  Import cold from
; IMPORT-ON-DAG:  Import hot from
; IMPORT-ON-DAG:  Import _ZL9local_hoti.__uniq.{{[0-9]+}}.llvm.{{[0-9]+}} from
; IMPORT-ON-DAG:  Import rare from
; IMPORT-ON-DAG:  Import pre from
; IMPORT-ON-DAG:  Import called from

;--- dispatch.ll
source_filename = "dispatch.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"

@g_local = dso_local global ptr @_ZL9local_hoti.__uniq.205233850391932887563395888172984341499, align 8, !dbg !0

; Function Attrs: mustprogress nounwind uwtable
define dso_local i32 @hot(i32 noundef %0) #0 !dbg !25 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4, !tbaa !28
    #dbg_declare(ptr %2, !27, !DIExpression(), !32)
  call void @llvm.pseudoprobe(i64 -8269091651996404185, i64 1, i32 0, i64 -1), !dbg !33
  %3 = load i32, ptr %2, align 4, !dbg !33, !tbaa !28
  %4 = mul nsw i32 3, %3, !dbg !34
  %5 = add nsw i32 %4, 1, !dbg !35
  ret i32 %5, !dbg !36
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local i32 @cold(i32 noundef %0) #0 !dbg !37 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4, !tbaa !28
    #dbg_declare(ptr %2, !39, !DIExpression(), !40)
  call void @llvm.pseudoprobe(i64 -6778568560291945099, i64 1, i32 0, i64 -1), !dbg !41
  %3 = load i32, ptr %2, align 4, !dbg !41, !tbaa !28
  %4 = mul nsw i32 7, %3, !dbg !42
  %5 = sub nsw i32 %4, 1, !dbg !43
  ret i32 %5, !dbg !44
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local i32 @called(i32 noundef %0) #0 !dbg !45 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4, !tbaa !28
    #dbg_declare(ptr %2, !47, !DIExpression(), !48)
  call void @llvm.pseudoprobe(i64 -8705695290094177457, i64 1, i32 0, i64 -1), !dbg !49
  %3 = load i32, ptr %2, align 4, !dbg !49, !tbaa !28
  %4 = mul nsw i32 5, %3, !dbg !50
  %5 = add nsw i32 %4, 2, !dbg !51
  ret i32 %5, !dbg !52
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local i32 @rare(i32 noundef %0) #0 !dbg !53 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4, !tbaa !28
    #dbg_declare(ptr %2, !55, !DIExpression(), !56)
  call void @llvm.pseudoprobe(i64 -1458566939842862681, i64 1, i32 0, i64 -1), !dbg !57
  %3 = load i32, ptr %2, align 4, !dbg !57, !tbaa !28
  %4 = mul nsw i32 13, %3, !dbg !58
  %5 = add nsw i32 %4, 5, !dbg !59
  ret i32 %5, !dbg !60
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local i32 @zero(i32 noundef %0) #0 !dbg !61 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4, !tbaa !28
    #dbg_declare(ptr %2, !63, !DIExpression(), !64)
  call void @llvm.pseudoprobe(i64 7126800030518488272, i64 1, i32 0, i64 -1), !dbg !65
  %3 = load i32, ptr %2, align 4, !dbg !65, !tbaa !28
  %4 = mul nsw i32 17, %3, !dbg !66
  %5 = add nsw i32 %4, 7, !dbg !67
  ret i32 %5, !dbg !68
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local i32 @pre(i32 noundef %0) #0 !dbg !69 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4, !tbaa !28
    #dbg_declare(ptr %2, !71, !DIExpression(), !72)
  call void @llvm.pseudoprobe(i64 -5005027373844268693, i64 1, i32 0, i64 -1), !dbg !73
  %3 = load i32, ptr %2, align 4, !dbg !73, !tbaa !28
  %4 = mul nsw i32 19, %3, !dbg !74
  %5 = add nsw i32 %4, 9, !dbg !75
  ret i32 %5, !dbg !76
}

; Function Attrs: mustprogress nounwind uwtable
define internal noundef i32 @_ZL9local_hoti.__uniq.205233850391932887563395888172984341499(i32 noundef %0) #1 !dbg !77 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4, !tbaa !28
    #dbg_declare(ptr %2, !79, !DIExpression(), !80)
  call void @llvm.pseudoprobe(i64 -933772253614566521, i64 1, i32 0, i64 -1), !dbg !81
  %3 = load i32, ptr %2, align 4, !dbg !81, !tbaa !28
  %4 = mul nsw i32 11, %3, !dbg !82
  %5 = add nsw i32 %4, 3, !dbg !83
  ret i32 %5, !dbg !84
}

; Function Attrs: mustprogress uwtable
define dso_local i32 @dispatch(i32 noundef %0, ptr noundef %1) #2 !dbg !85 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store i32 %0, ptr %3, align 4, !tbaa !28
    #dbg_declare(ptr %3, !89, !DIExpression(), !91)
  store ptr %1, ptr %4, align 8, !tbaa !92
    #dbg_declare(ptr %4, !90, !DIExpression(), !94)
  call void @llvm.pseudoprobe(i64 -7762558725055652451, i64 1, i32 0, i64 -1), !dbg !95
  %5 = load ptr, ptr %4, align 8, !dbg !95, !tbaa !92
  %6 = load i32, ptr %3, align 4, !dbg !96, !tbaa !28
  %7 = call noundef i32 %5(i32 noundef %6), !dbg !97
  ret i32 %7, !dbg !99
}

; Function Attrs: mustprogress uwtable
define dso_local i32 @dispatch2(i32 noundef %0, ptr noundef %1) #2 !dbg !100 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store i32 %0, ptr %3, align 4, !tbaa !28
    #dbg_declare(ptr %3, !102, !DIExpression(), !104)
  store ptr %1, ptr %4, align 8, !tbaa !92
    #dbg_declare(ptr %4, !103, !DIExpression(), !105)
  call void @llvm.pseudoprobe(i64 707185974311385933, i64 1, i32 0, i64 -1), !dbg !106
  %5 = load ptr, ptr %4, align 8, !dbg !106, !tbaa !92
  %6 = load i32, ptr %3, align 4, !dbg !107, !tbaa !28
  %7 = call noundef i32 %5(i32 noundef %6), !dbg !108
  %8 = load i32, ptr %3, align 4, !dbg !110, !tbaa !28
  %9 = call i32 @called(i32 noundef %8), !dbg !111
  %10 = add nsw i32 %7, %9, !dbg !113
  ret i32 %10, !dbg !114
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #3

attributes #0 = { "use-sample-profile" mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { "use-sample-profile" mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "sample-profile-suffix-elision-policy"="selected" "stack-protector-buffer-size"="8" }
attributes #2 = { "use-sample-profile" mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { "use-sample-profile" nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !13, !14}
!llvm.pseudo_probe_desc = !{!16, !17, !18, !19, !20, !21, !22, !23, !24}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g_local", scope: !2, file: !3, line: 10, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!3 = !DIFile(filename: "dispatch.cpp", directory: "/", checksumkind: CSK_MD5, checksum: "e2761f98609a1a3e4cd44e2bf948b20c")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !6)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 7, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 7, !"uwtable", i32 2}
!14 = !{i32 7, !"frame-pointer", i32 1}
!15 = !{!"clang version 21.1.0 (ssh://git.vip.facebook.com/data/gitrepos/osmeta/external/llvm-project 912e119fd9606d5816cce0d2f42ff0f988242ef6)"}
!16 = !{i64 -8269091651996404185, i64 4294967295, !"hot"}
!17 = !{i64 -6778568560291945099, i64 4294967295, !"cold"}
!18 = !{i64 -8705695290094177457, i64 4294967295, !"called"}
!19 = !{i64 -1458566939842862681, i64 4294967295, !"rare"}
!20 = !{i64 7126800030518488272, i64 4294967295, !"zero"}
!21 = !{i64 -5005027373844268693, i64 4294967295, !"pre"}
!22 = !{i64 -933772253614566521, i64 4294967295, !"_ZL9local_hoti.__uniq.205233850391932887563395888172984341499"}
!23 = !{i64 -7762558725055652451, i64 281479271677951, !"dispatch"}
!24 = !{i64 707185974311385933, i64 562954248388607, !"dispatch2"}
!25 = distinct !DISubprogram(name: "hot", scope: !3, file: !3, line: 3, type: !7, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !26)
!26 = !{!27}
!27 = !DILocalVariable(name: "x", arg: 1, scope: !25, file: !3, line: 3, type: !9)
!28 = !{!29, !29, i64 0}
!29 = !{!"int", !30, i64 0}
!30 = !{!"omnipotent char", !31, i64 0}
!31 = !{!"Simple C++ TBAA"}
!32 = !DILocation(line: 3, column: 24, scope: !25)
!33 = !DILocation(line: 3, column: 40, scope: !25)
!34 = !DILocation(line: 3, column: 38, scope: !25)
!35 = !DILocation(line: 3, column: 42, scope: !25)
!36 = !DILocation(line: 3, column: 29, scope: !25)
!37 = distinct !DISubprogram(name: "cold", scope: !3, file: !3, line: 4, type: !7, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !38)
!38 = !{!39}
!39 = !DILocalVariable(name: "x", arg: 1, scope: !37, file: !3, line: 4, type: !9)
!40 = !DILocation(line: 4, column: 25, scope: !37)
!41 = !DILocation(line: 4, column: 41, scope: !37)
!42 = !DILocation(line: 4, column: 39, scope: !37)
!43 = !DILocation(line: 4, column: 43, scope: !37)
!44 = !DILocation(line: 4, column: 30, scope: !37)
!45 = distinct !DISubprogram(name: "called", scope: !3, file: !3, line: 5, type: !7, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !46)
!46 = !{!47}
!47 = !DILocalVariable(name: "x", arg: 1, scope: !45, file: !3, line: 5, type: !9)
!48 = !DILocation(line: 5, column: 27, scope: !45)
!49 = !DILocation(line: 5, column: 43, scope: !45)
!50 = !DILocation(line: 5, column: 41, scope: !45)
!51 = !DILocation(line: 5, column: 45, scope: !45)
!52 = !DILocation(line: 5, column: 32, scope: !45)
!53 = distinct !DISubprogram(name: "rare", scope: !3, file: !3, line: 6, type: !7, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !54)
!54 = !{!55}
!55 = !DILocalVariable(name: "x", arg: 1, scope: !53, file: !3, line: 6, type: !9)
!56 = !DILocation(line: 6, column: 25, scope: !53)
!57 = !DILocation(line: 6, column: 42, scope: !53)
!58 = !DILocation(line: 6, column: 40, scope: !53)
!59 = !DILocation(line: 6, column: 44, scope: !53)
!60 = !DILocation(line: 6, column: 30, scope: !53)
!61 = distinct !DISubprogram(name: "zero", scope: !3, file: !3, line: 7, type: !7, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !62)
!62 = !{!63}
!63 = !DILocalVariable(name: "x", arg: 1, scope: !61, file: !3, line: 7, type: !9)
!64 = !DILocation(line: 7, column: 25, scope: !61)
!65 = !DILocation(line: 7, column: 42, scope: !61)
!66 = !DILocation(line: 7, column: 40, scope: !61)
!67 = !DILocation(line: 7, column: 44, scope: !61)
!68 = !DILocation(line: 7, column: 30, scope: !61)
!69 = distinct !DISubprogram(name: "pre", scope: !3, file: !3, line: 8, type: !7, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !70)
!70 = !{!71}
!71 = !DILocalVariable(name: "x", arg: 1, scope: !69, file: !3, line: 8, type: !9)
!72 = !DILocation(line: 8, column: 24, scope: !69)
!73 = !DILocation(line: 8, column: 41, scope: !69)
!74 = !DILocation(line: 8, column: 39, scope: !69)
!75 = !DILocation(line: 8, column: 43, scope: !69)
!76 = !DILocation(line: 8, column: 29, scope: !69)
!77 = distinct !DISubprogram(name: "local_hot", linkageName: "_ZL9local_hoti.__uniq.205233850391932887563395888172984341499", scope: !3, file: !3, line: 9, type: !7, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !78)
!78 = !{!79}
!79 = !DILocalVariable(name: "x", arg: 1, scope: !77, file: !3, line: 9, type: !9)
!80 = !DILocation(line: 9, column: 26, scope: !77)
!81 = !DILocation(line: 9, column: 43, scope: !77)
!82 = !DILocation(line: 9, column: 41, scope: !77)
!83 = !DILocation(line: 9, column: 45, scope: !77)
!84 = !DILocation(line: 9, column: 31, scope: !77)
!85 = distinct !DISubprogram(name: "dispatch", scope: !3, file: !3, line: 11, type: !86, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !88)
!86 = !DISubroutineType(types: !87)
!87 = !{!9, !9, !6}
!88 = !{!89, !90}
!89 = !DILocalVariable(name: "x", arg: 1, scope: !85, file: !3, line: 11, type: !9)
!90 = !DILocalVariable(name: "t", arg: 2, scope: !85, file: !3, line: 11, type: !6)
!91 = !DILocation(line: 11, column: 29, scope: !85)
!92 = !{!93, !93, i64 0}
!93 = !{!"any pointer", !30, i64 0}
!94 = !DILocation(line: 11, column: 38, scope: !85)
!95 = !DILocation(line: 11, column: 56, scope: !85)
!96 = !DILocation(line: 11, column: 58, scope: !85)
!97 = !DILocation(line: 11, column: 56, scope: !98)
!98 = !DILexicalBlockFile(scope: !85, file: !3, discriminator: 387973143)
!99 = !DILocation(line: 11, column: 49, scope: !85)
!100 = distinct !DISubprogram(name: "dispatch2", scope: !3, file: !3, line: 12, type: !86, scopeLine: 12, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !101)
!101 = !{!102, !103}
!102 = !DILocalVariable(name: "x", arg: 1, scope: !100, file: !3, line: 12, type: !9)
!103 = !DILocalVariable(name: "t", arg: 2, scope: !100, file: !3, line: 12, type: !6)
!104 = !DILocation(line: 12, column: 30, scope: !100)
!105 = !DILocation(line: 12, column: 39, scope: !100)
!106 = !DILocation(line: 12, column: 57, scope: !100)
!107 = !DILocation(line: 12, column: 59, scope: !100)
!108 = !DILocation(line: 12, column: 57, scope: !109)
!109 = !DILexicalBlockFile(scope: !100, file: !3, discriminator: 387973143)
!110 = !DILocation(line: 12, column: 71, scope: !100)
!111 = !DILocation(line: 12, column: 64, scope: !112)
!112 = !DILexicalBlockFile(scope: !100, file: !3, discriminator: 455082015)
!113 = !DILocation(line: 12, column: 62, scope: !100)
!114 = !DILocation(line: 12, column: 50, scope: !100)

;--- caller.ll
source_filename = "caller.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"

; Function Attrs: mustprogress uwtable
define dso_local i32 @caller(i32 noundef %0, ptr noundef %1) #0 !dbg !9 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store i32 %0, ptr %3, align 4, !tbaa !19
    #dbg_declare(ptr %3, !17, !DIExpression(), !23)
  store ptr %1, ptr %4, align 8, !tbaa !24
    #dbg_declare(ptr %4, !18, !DIExpression(), !26)
  call void @llvm.pseudoprobe(i64 -1768971689307247648, i64 1, i32 0, i64 -1), !dbg !27
  %5 = load i32, ptr %3, align 4, !dbg !27, !tbaa !19
  %6 = load ptr, ptr %4, align 8, !dbg !28, !tbaa !24
  %7 = call i32 @dispatch(i32 noundef %5, ptr noundef %6), !dbg !29
  %8 = load i32, ptr %3, align 4, !dbg !31, !tbaa !19
  %9 = load ptr, ptr %4, align 8, !dbg !32, !tbaa !24
  %10 = call i32 @dispatch2(i32 noundef %8, ptr noundef %9), !dbg !33
  %11 = add nsw i32 %7, %10, !dbg !35
  ret i32 %11, !dbg !36
}

declare !dbg !37 dso_local i32 @dispatch(i32 noundef, ptr noundef) #1

declare !dbg !38 dso_local i32 @dispatch2(i32 noundef, ptr noundef) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #2

attributes #0 = { "use-sample-profile" mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { "use-sample-profile" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { "use-sample-profile" nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.pseudo_probe_desc = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "caller.cpp", directory: "/", checksumkind: CSK_MD5, checksum: "4842e70420ef6581a6289dd77c51b0a4")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{i32 7, !"frame-pointer", i32 1}
!7 = !{!"clang version 21.1.0 (ssh://git.vip.facebook.com/data/gitrepos/osmeta/external/llvm-project 912e119fd9606d5816cce0d2f42ff0f988242ef6)"}
!8 = !{i64 -1768971689307247648, i64 562954248388607, !"caller"}
!9 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12, !13}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DISubroutineType(types: !15)
!15 = !{!12, !12}
!16 = !{!17, !18}
!17 = !DILocalVariable(name: "x", arg: 1, scope: !9, file: !1, line: 3, type: !12)
!18 = !DILocalVariable(name: "t", arg: 2, scope: !9, file: !1, line: 3, type: !13)
!19 = !{!20, !20, i64 0}
!20 = !{!"int", !21, i64 0}
!21 = !{!"omnipotent char", !22, i64 0}
!22 = !{!"Simple C++ TBAA"}
!23 = !DILocation(line: 3, column: 27, scope: !9)
!24 = !{!25, !25, i64 0}
!25 = !{!"any pointer", !21, i64 0}
!26 = !DILocation(line: 3, column: 36, scope: !9)
!27 = !DILocation(line: 3, column: 63, scope: !9)
!28 = !DILocation(line: 3, column: 66, scope: !9)
!29 = !DILocation(line: 3, column: 54, scope: !30)
!30 = !DILexicalBlockFile(scope: !9, file: !1, discriminator: 455082007)
!31 = !DILocation(line: 3, column: 81, scope: !9)
!32 = !DILocation(line: 3, column: 84, scope: !9)
!33 = !DILocation(line: 3, column: 71, scope: !34)
!34 = !DILexicalBlockFile(scope: !9, file: !1, discriminator: 455082015)
!35 = !DILocation(line: 3, column: 69, scope: !9)
!36 = !DILocation(line: 3, column: 47, scope: !9)
!37 = !DISubprogram(name: "dispatch", scope: !1, file: !1, line: 1, type: !10, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!38 = !DISubprogram(name: "dispatch2", scope: !1, file: !1, line: 2, type: !10, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)

;--- nested.txt
caller:3000000:1000000
 1: 1000000
 2: 1000000 dispatch:1000000
 3: 1000000 dispatch2:1000000
 !CFGChecksum: 562954248388607
dispatch:2000000:1000000
 1: 1000000
 2: 100000 cold:100000
 2: hot:600000
  1: 600000
  !CFGChecksum: 4294967295
  !Attributes: 3
 2: _ZL9local_hoti.__uniq.205233850391932887563395888172984341499:300000
  1: 300000
  !CFGChecksum: 4294967295
  !Attributes: 3
 2: rare:2
  1: 2
  !CFGChecksum: 4294967295
  !Attributes: 1
 2: zero:0
  1: 0
  !CFGChecksum: 4294967295
  !Attributes: 3
 2: pre:2
  1: 2
  !CFGChecksum: 4294967295
  !Attributes: 3
 !CFGChecksum: 281479271677951
 !Attributes: 1
dispatch2:2000000:1000000
 1: 1000000
 3: 800000 called:800000
 2: called:700000
  1: 700000
  !CFGChecksum: 4294967295
  !Attributes: 3
 !CFGChecksum: 562954248388607
 !Attributes: 1
cold:100000:100000
 1: 100000
 !CFGChecksum: 4294967295
;--- cs.txt
[caller]:3000000:1000000
 1: 1000000
 2: 1000000 dispatch:1000000
 3: 1000000 dispatch2:1000000
 !CFGChecksum: 562954248388607
[dispatch]:1100000:1000000
 1: 1000000
 2: 100000 cold:100000
 !CFGChecksum: 281479271677951
[dispatch:2 @ hot]:600000:600000
 1: 600000
 !CFGChecksum: 4294967295
[dispatch:2 @ _ZL9local_hoti.__uniq.205233850391932887563395888172984341499]:300000:300000
 1: 300000
 !CFGChecksum: 4294967295
[dispatch:2 @ rare]:2:2
 1: 2
 !CFGChecksum: 4294967295
[dispatch:2 @ pre]:2:2
 1: 2
 !CFGChecksum: 4294967295
 !Attributes: 2
[dispatch:2 @ zero]:0:0
 1: 0
 !CFGChecksum: 4294967295
 !Attributes: 2
[dispatch2]:1800000:1000000
 1: 1000000
 3: 800000 called:800000
 !CFGChecksum: 562954248388607
[dispatch2:2 @ called]:700000:700000
 1: 700000
 !CFGChecksum: 4294967295
[cold]:100000:100000
 1: 100000
 !CFGChecksum: 4294967295
