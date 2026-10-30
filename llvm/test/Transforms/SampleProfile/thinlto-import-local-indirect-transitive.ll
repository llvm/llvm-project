; Deferred local targets can themselves have only contextual profiles. Walk
; those profiles transitively, including hot residual targets. Do not depend
; on a separate base profile or on a surviving direct call to create edges.
; RUN: split-file %s %t && cd %t
; RUN: llvm-profdata merge --sample --extbinary nested.txt -o p.prof
; RUN: opt lim.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=p.prof -profile-summary-hot-count=1000 -profile-summary-cold-count=10 -sample-profile-import-local-indirect-targets -thinlto-bc -o lim.bc
; RUN: llvm-dis lim.bc -o - | FileCheck %s --check-prefix=ALL
; RUN: opt user.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=p.prof -profile-summary-hot-count=1000 -profile-summary-cold-count=10 -thinlto-bc -o user.bc
; RUN: llvm-lto -thinlto-action=thinlink -o index.bc lim.bc user.bc
; RUN: opt -passes=function-import -summary-file=index.bc user.bc -print-imports -o /dev/null 2>&1 | FileCheck %s --check-prefix=IMPORTS
; RUN: opt -passes=function-import -summary-file=index.bc user.bc -print-imports -o /dev/null 2>&1 | grep Import | count 5
;
; A repeated function name is another finite context, not an IR-graph cycle.
; Its GUID must appear only once, and its nested leaf must still be traversed.
; RUN: llvm-profdata merge --sample --extbinary recursive.txt -o recursive.prof
; RUN: opt lim.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=recursive.prof -profile-summary-hot-count=1000 -profile-summary-cold-count=10 -sample-profile-import-local-indirect-targets -S | FileCheck %s --check-prefix=ALL
;
; With pre-inliner decisions disabled, prune a cold nested leaf. Retain the
; hot residual target even though it has no nested FunctionSamples here.
; RUN: llvm-profdata merge --sample --extbinary cold-leaf.txt -o cold-leaf.prof
; RUN: opt lim.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=cold-leaf.prof -profile-summary-hot-count=1000 -profile-summary-cold-count=10 -sample-profile-import-local-indirect-targets -sample-profile-use-preinliner=false -S | FileCheck %s --check-prefix=RESIDUAL
;
; CS profiles use context-entry hotness and prune descendants when their
; ancestor is cold and not selected. A hot grandchild cannot bypass that gate.
; RUN: llvm-profdata merge --sample --extbinary cs-hot.txt -o cs-hot.prof
; RUN: opt lim.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=cs-hot.prof -profile-summary-hot-count=1000 -profile-summary-cold-count=10 -sample-profile-import-local-indirect-targets -S | FileCheck %s --check-prefix=ALL
; RUN: llvm-profdata merge --sample --extbinary cs-cold.txt -o cs-cold.prof
; RUN: opt lim.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=cs-cold.prof -profile-summary-hot-count=1000 -profile-summary-cold-count=10 -sample-profile-import-local-indirect-targets -sample-profile-use-preinliner=false -S | FileCheck %s --check-prefix=COLD

; ALL: define {{.*}} @dispatch4({{.*}} !prof ![[#D:]]
; ALL: ![[#D]] = !{!"function_entry_count", i64 1000000, i64 1969880780693113886, i64 4395024877212874786, i64 7399155579282764986}
; RESIDUAL: define {{.*}} @dispatch4({{.*}} !prof ![[#D:]]
; RESIDUAL: ![[#D]] = !{!"function_entry_count", i64 1000000, i64 1969880780693113886, i64 4395024877212874786}
; COLD: define {{.*}} @dispatch4({{.*}} !prof ![[#D:]]
; COLD: ![[#D]] = !{!"function_entry_count", i64 1000000}
; IMPORTS-DAG: Import dispatch3 from
; IMPORTS-DAG: Import dispatch4 from
; IMPORTS-DAG: Import called2 from
; IMPORTS-DAG: Import mid from
; IMPORTS-DAG: Import leaf from

;--- lim.ll
source_filename = "lim.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"

@g_t = external dso_local global ptr, align 8
@g_n = external dso_local global i32, align 4

; Function Attrs: mustprogress nounwind uwtable
define dso_local i32 @called2(i32 noundef %0) #0 !dbg !13 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store i32 %0, ptr %2, align 4, !tbaa !28
    #dbg_declare(ptr %2, !18, !DIExpression(), !32)
  call void @llvm.lifetime.start.p0(i64 4, ptr %3) #4, !dbg !33
    #dbg_declare(ptr %3, !19, !DIExpression(), !34)
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 1, i32 0, i64 -1), !dbg !34
  store i32 0, ptr %3, align 4, !dbg !34, !tbaa !28
  call void @llvm.lifetime.start.p0(i64 4, ptr %4) #4, !dbg !35
    #dbg_declare(ptr %4, !20, !DIExpression(), !36)
  store i32 0, ptr %4, align 4, !dbg !36, !tbaa !28
  br label %8, !dbg !35

8:                                                ; preds = %21, %1
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 2, i32 0, i64 -1), !dbg !37
  %9 = load i32, ptr %4, align 4, !dbg !37, !tbaa !28
  %10 = load i32, ptr %2, align 4, !dbg !39, !tbaa !28
  %11 = icmp slt i32 %9, %10, !dbg !40
  br i1 %11, label %13, label %12, !dbg !41

12:                                               ; preds = %8
  call void @llvm.lifetime.end.p0(i64 4, ptr %4) #4, !dbg !42
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 3, i32 0, i64 -1), !dbg !43
  br label %24

13:                                               ; preds = %8
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 4, i32 0, i64 -1), !dbg !44
  %14 = load i32, ptr %4, align 4, !dbg !44, !tbaa !28
  %15 = mul nsw i32 %14, 7, !dbg !45
  %16 = load i32, ptr %3, align 4, !dbg !46, !tbaa !28
  %17 = ashr i32 %16, 3, !dbg !47
  %18 = xor i32 %15, %17, !dbg !48
  %19 = load i32, ptr %3, align 4, !dbg !49, !tbaa !28
  %20 = add nsw i32 %19, %18, !dbg !49
  store i32 %20, ptr %3, align 4, !dbg !49, !tbaa !28
  br label %21, !dbg !50

21:                                               ; preds = %13
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 5, i32 0, i64 -1), !dbg !51
  %22 = load i32, ptr %4, align 4, !dbg !51, !tbaa !28
  %23 = add nsw i32 %22, 1, !dbg !51
  store i32 %23, ptr %4, align 4, !dbg !51, !tbaa !28
  br label %8, !dbg !42, !llvm.loop !52

24:                                               ; preds = %12
  call void @llvm.lifetime.start.p0(i64 4, ptr %5) #4, !dbg !55
    #dbg_declare(ptr %5, !22, !DIExpression(), !56)
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 6, i32 0, i64 -1), !dbg !56
  store i32 0, ptr %5, align 4, !dbg !56, !tbaa !28
  br label %25, !dbg !55

25:                                               ; preds = %40, %24
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 7, i32 0, i64 -1), !dbg !57
  %26 = load i32, ptr %5, align 4, !dbg !57, !tbaa !28
  %27 = load i32, ptr %2, align 4, !dbg !59, !tbaa !28
  %28 = icmp slt i32 %26, %27, !dbg !60
  br i1 %28, label %30, label %29, !dbg !61

29:                                               ; preds = %25
  call void @llvm.lifetime.end.p0(i64 4, ptr %5) #4, !dbg !62
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 8, i32 0, i64 -1), !dbg !43
  br label %43

30:                                               ; preds = %25
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 9, i32 0, i64 -1), !dbg !63
  %31 = load i32, ptr %3, align 4, !dbg !63, !tbaa !28
  %32 = shl i32 %31, 5, !dbg !64
  %33 = load i32, ptr %5, align 4, !dbg !65, !tbaa !28
  %34 = load i32, ptr %2, align 4, !dbg !66, !tbaa !28
  %35 = or i32 %34, 1, !dbg !67
  %36 = sdiv i32 %33, %35, !dbg !68
  %37 = add nsw i32 %32, %36, !dbg !69
  %38 = load i32, ptr %3, align 4, !dbg !70, !tbaa !28
  %39 = xor i32 %38, %37, !dbg !70
  store i32 %39, ptr %3, align 4, !dbg !70, !tbaa !28
  br label %40, !dbg !71

40:                                               ; preds = %30
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 10, i32 0, i64 -1), !dbg !72
  %41 = load i32, ptr %5, align 4, !dbg !72, !tbaa !28
  %42 = add nsw i32 %41, 1, !dbg !72
  store i32 %42, ptr %5, align 4, !dbg !72, !tbaa !28
  br label %25, !dbg !62, !llvm.loop !73

43:                                               ; preds = %29
  call void @llvm.lifetime.start.p0(i64 4, ptr %6) #4, !dbg !75
    #dbg_declare(ptr %6, !24, !DIExpression(), !76)
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 11, i32 0, i64 -1), !dbg !76
  store i32 0, ptr %6, align 4, !dbg !76, !tbaa !28
  br label %44, !dbg !75

44:                                               ; preds = %59, %43
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 12, i32 0, i64 -1), !dbg !77
  %45 = load i32, ptr %6, align 4, !dbg !77, !tbaa !28
  %46 = load i32, ptr %2, align 4, !dbg !79, !tbaa !28
  %47 = icmp slt i32 %45, %46, !dbg !80
  br i1 %47, label %49, label %48, !dbg !81

48:                                               ; preds = %44
  call void @llvm.lifetime.end.p0(i64 4, ptr %6) #4, !dbg !82
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 13, i32 0, i64 -1), !dbg !43
  br label %62

49:                                               ; preds = %44
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 14, i32 0, i64 -1), !dbg !83
  %50 = load i32, ptr %3, align 4, !dbg !83, !tbaa !28
  %51 = srem i32 %50, 13, !dbg !84
  %52 = load i32, ptr %6, align 4, !dbg !85, !tbaa !28
  %53 = mul nsw i32 %51, %52, !dbg !86
  %54 = load i32, ptr %6, align 4, !dbg !87, !tbaa !28
  %55 = ashr i32 %54, 2, !dbg !88
  %56 = sub nsw i32 %53, %55, !dbg !89
  %57 = load i32, ptr %3, align 4, !dbg !90, !tbaa !28
  %58 = add nsw i32 %57, %56, !dbg !90
  store i32 %58, ptr %3, align 4, !dbg !90, !tbaa !28
  br label %59, !dbg !91

59:                                               ; preds = %49
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 15, i32 0, i64 -1), !dbg !92
  %60 = load i32, ptr %6, align 4, !dbg !92, !tbaa !28
  %61 = add nsw i32 %60, 1, !dbg !92
  store i32 %61, ptr %6, align 4, !dbg !92, !tbaa !28
  br label %44, !dbg !82, !llvm.loop !93

62:                                               ; preds = %48
  call void @llvm.lifetime.start.p0(i64 4, ptr %7) #4, !dbg !95
    #dbg_declare(ptr %7, !26, !DIExpression(), !96)
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 16, i32 0, i64 -1), !dbg !96
  store i32 0, ptr %7, align 4, !dbg !96, !tbaa !28
  br label %63, !dbg !95

63:                                               ; preds = %76, %62
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 17, i32 0, i64 -1), !dbg !97
  %64 = load i32, ptr %7, align 4, !dbg !97, !tbaa !28
  %65 = load i32, ptr %2, align 4, !dbg !99, !tbaa !28
  %66 = icmp slt i32 %64, %65, !dbg !100
  br i1 %66, label %68, label %67, !dbg !101

67:                                               ; preds = %63
  call void @llvm.lifetime.end.p0(i64 4, ptr %7) #4, !dbg !102
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 18, i32 0, i64 -1), !dbg !43
  br label %79

68:                                               ; preds = %63
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 19, i32 0, i64 -1), !dbg !103
  %69 = load i32, ptr %3, align 4, !dbg !103, !tbaa !28
  %70 = mul nsw i32 %69, 31, !dbg !104
  %71 = load i32, ptr %7, align 4, !dbg !105, !tbaa !28
  %72 = srem i32 %71, 7, !dbg !106
  %73 = xor i32 %70, %72, !dbg !107
  %74 = load i32, ptr %3, align 4, !dbg !108, !tbaa !28
  %75 = sub nsw i32 %74, %73, !dbg !108
  store i32 %75, ptr %3, align 4, !dbg !108, !tbaa !28
  br label %76, !dbg !109

76:                                               ; preds = %68
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 20, i32 0, i64 -1), !dbg !110
  %77 = load i32, ptr %7, align 4, !dbg !110, !tbaa !28
  %78 = add nsw i32 %77, 1, !dbg !110
  store i32 %78, ptr %7, align 4, !dbg !110, !tbaa !28
  br label %63, !dbg !102, !llvm.loop !111

79:                                               ; preds = %67
  call void @llvm.pseudoprobe(i64 1969880780693113886, i64 21, i32 0, i64 -1), !dbg !113
  %80 = load i32, ptr %3, align 4, !dbg !113, !tbaa !28
  call void @llvm.lifetime.end.p0(i64 4, ptr %3) #4, !dbg !114
  ret i32 %80, !dbg !115
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #1

; Function Attrs: mustprogress nounwind uwtable
define dso_local i32 @leaf(i32 noundef %0) #0 !dbg !116 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4, !tbaa !28
    #dbg_declare(ptr %2, !118, !DIExpression(), !119)
  call void @llvm.pseudoprobe(i64 7399155579282764986, i64 1, i32 0, i64 -1), !dbg !120
  %3 = load i32, ptr %2, align 4, !dbg !120, !tbaa !28
  %4 = mul nsw i32 23, %3, !dbg !121
  %5 = add nsw i32 %4, 1, !dbg !122
  ret i32 %5, !dbg !123
}

; Function Attrs: mustprogress uwtable
define dso_local i32 @mid(i32 noundef %0) #2 !dbg !124 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4, !tbaa !28
    #dbg_declare(ptr %2, !126, !DIExpression(), !127)
  call void @llvm.pseudoprobe(i64 4395024877212874786, i64 1, i32 0, i64 -1), !dbg !128
  %3 = load volatile ptr, ptr @g_t, align 8, !dbg !128, !tbaa !129
  %4 = load i32, ptr %2, align 4, !dbg !131, !tbaa !28
  %5 = call i32 %3(i32 noundef %4), !dbg !132
  %6 = add nsw i32 %5, 1, !dbg !134
  ret i32 %6, !dbg !135
}

; Function Attrs: mustprogress uwtable
define dso_local i32 @dispatch3(i32 noundef %0, ptr noundef %1) #2 !dbg !136 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store i32 %0, ptr %3, align 4, !tbaa !28
    #dbg_declare(ptr %3, !141, !DIExpression(), !144)
  store ptr %1, ptr %4, align 8, !tbaa !129
    #dbg_declare(ptr %4, !142, !DIExpression(), !145)
  call void @llvm.lifetime.start.p0(i64 4, ptr %5) #4, !dbg !146
    #dbg_declare(ptr %5, !143, !DIExpression(), !147)
  call void @llvm.pseudoprobe(i64 2018282615097108723, i64 1, i32 0, i64 -1), !dbg !148
  %6 = load ptr, ptr %4, align 8, !dbg !148, !tbaa !129
  %7 = load i32, ptr %3, align 4, !dbg !149, !tbaa !28
  %8 = call noundef i32 %6(i32 noundef %7), !dbg !150
  store i32 %8, ptr %5, align 4, !dbg !147, !tbaa !28
  %9 = load i32, ptr %3, align 4, !dbg !152, !tbaa !28
  %10 = icmp eq i32 %9, 12345, !dbg !154
  br i1 %10, label %11, label %16, !dbg !154

11:                                               ; preds = %2
  call void @llvm.pseudoprobe(i64 2018282615097108723, i64 3, i32 0, i64 -1), !dbg !155
  %12 = load volatile i32, ptr @g_n, align 4, !dbg !155, !tbaa !28
  %13 = call i32 @called2(i32 noundef %12), !dbg !156
  %14 = load i32, ptr %5, align 4, !dbg !158, !tbaa !28
  %15 = add nsw i32 %14, %13, !dbg !158
  store i32 %15, ptr %5, align 4, !dbg !158, !tbaa !28
  br label %16, !dbg !159

16:                                               ; preds = %11, %2
  call void @llvm.pseudoprobe(i64 2018282615097108723, i64 5, i32 0, i64 -1), !dbg !160
  %17 = load i32, ptr %5, align 4, !dbg !160, !tbaa !28
  call void @llvm.lifetime.end.p0(i64 4, ptr %5) #4, !dbg !161
  ret i32 %17, !dbg !162
}

; Function Attrs: mustprogress uwtable
define dso_local i32 @dispatch4(i32 noundef %0, ptr noundef %1) #2 !dbg !163 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store i32 %0, ptr %3, align 4, !tbaa !28
    #dbg_declare(ptr %3, !165, !DIExpression(), !167)
  store ptr %1, ptr %4, align 8, !tbaa !129
    #dbg_declare(ptr %4, !166, !DIExpression(), !168)
  call void @llvm.pseudoprobe(i64 -192514282485742112, i64 1, i32 0, i64 -1), !dbg !169
  %5 = load ptr, ptr %4, align 8, !dbg !169, !tbaa !129
  %6 = load i32, ptr %3, align 4, !dbg !170, !tbaa !28
  %7 = call noundef i32 %5(i32 noundef %6), !dbg !171
  ret i32 %7, !dbg !173
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #3

attributes #0 = { "use-sample-profile" mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { "use-sample-profile" nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { "use-sample-profile" mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { "use-sample-profile" nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #4 = { "use-sample-profile" nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}
!llvm.pseudo_probe_desc = !{!8, !9, !10, !11, !12}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "lim.cpp", directory: "/data/users/kupathak/llvm21-importhint/limits", checksumkind: CSK_MD5, checksum: "633a57f6c1c82f1c3e2e4437aa596262")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{i32 7, !"frame-pointer", i32 1}
!7 = !{!"clang version 21.1.0 (ssh://git.vip.facebook.com/data/gitrepos/osmeta/external/llvm-project 912e119fd9606d5816cce0d2f42ff0f988242ef6)"}
!8 = !{i64 1969880780693113886, i64 415320791528, !"called2"}
!9 = !{i64 7399155579282764986, i64 4294967295, !"leaf"}
!10 = !{i64 4395024877212874786, i64 281479271677951, !"mid"}
!11 = !{i64 2018282615097108723, i64 563003850888502, !"dispatch3"}
!12 = !{i64 -192514282485742112, i64 281479271677951, !"dispatch4"}
!13 = distinct !DISubprogram(name: "called2", scope: !1, file: !1, line: 6, type: !14, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !17)
!14 = !DISubroutineType(types: !15)
!15 = !{!16, !16}
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !{!18, !19, !20, !22, !24, !26}
!18 = !DILocalVariable(name: "x", arg: 1, scope: !13, file: !1, line: 6, type: !16)
!19 = !DILocalVariable(name: "s", scope: !13, file: !1, line: 7, type: !16)
!20 = !DILocalVariable(name: "i", scope: !21, file: !1, line: 8, type: !16)
!21 = distinct !DILexicalBlock(scope: !13, file: !1, line: 8, column: 3)
!22 = !DILocalVariable(name: "i", scope: !23, file: !1, line: 9, type: !16)
!23 = distinct !DILexicalBlock(scope: !13, file: !1, line: 9, column: 3)
!24 = !DILocalVariable(name: "i", scope: !25, file: !1, line: 10, type: !16)
!25 = distinct !DILexicalBlock(scope: !13, file: !1, line: 10, column: 3)
!26 = !DILocalVariable(name: "i", scope: !27, file: !1, line: 11, type: !16)
!27 = distinct !DILexicalBlock(scope: !13, file: !1, line: 11, column: 3)
!28 = !{!29, !29, i64 0}
!29 = !{!"int", !30, i64 0}
!30 = !{!"omnipotent char", !31, i64 0}
!31 = !{!"Simple C++ TBAA"}
!32 = !DILocation(line: 6, column: 28, scope: !13)
!33 = !DILocation(line: 7, column: 3, scope: !13)
!34 = !DILocation(line: 7, column: 7, scope: !13)
!35 = !DILocation(line: 8, column: 8, scope: !21)
!36 = !DILocation(line: 8, column: 12, scope: !21)
!37 = !DILocation(line: 8, column: 19, scope: !38)
!38 = distinct !DILexicalBlock(scope: !21, file: !1, line: 8, column: 3)
!39 = !DILocation(line: 8, column: 23, scope: !38)
!40 = !DILocation(line: 8, column: 21, scope: !38)
!41 = !DILocation(line: 8, column: 3, scope: !21)
!42 = !DILocation(line: 8, column: 3, scope: !38)
!43 = !DILocation(line: 0, scope: !13)
!44 = !DILocation(line: 8, column: 37, scope: !38)
!45 = !DILocation(line: 8, column: 39, scope: !38)
!46 = !DILocation(line: 8, column: 47, scope: !38)
!47 = !DILocation(line: 8, column: 49, scope: !38)
!48 = !DILocation(line: 8, column: 44, scope: !38)
!49 = !DILocation(line: 8, column: 33, scope: !38)
!50 = !DILocation(line: 8, column: 31, scope: !38)
!51 = !DILocation(line: 8, column: 26, scope: !38)
!52 = distinct !{!52, !41, !53, !54}
!53 = !DILocation(line: 8, column: 53, scope: !21)
!54 = !{!"llvm.loop.mustprogress"}
!55 = !DILocation(line: 9, column: 8, scope: !23)
!56 = !DILocation(line: 9, column: 12, scope: !23)
!57 = !DILocation(line: 9, column: 19, scope: !58)
!58 = distinct !DILexicalBlock(scope: !23, file: !1, line: 9, column: 3)
!59 = !DILocation(line: 9, column: 23, scope: !58)
!60 = !DILocation(line: 9, column: 21, scope: !58)
!61 = !DILocation(line: 9, column: 3, scope: !23)
!62 = !DILocation(line: 9, column: 3, scope: !58)
!63 = !DILocation(line: 9, column: 37, scope: !58)
!64 = !DILocation(line: 9, column: 39, scope: !58)
!65 = !DILocation(line: 9, column: 47, scope: !58)
!66 = !DILocation(line: 9, column: 52, scope: !58)
!67 = !DILocation(line: 9, column: 54, scope: !58)
!68 = !DILocation(line: 9, column: 49, scope: !58)
!69 = !DILocation(line: 9, column: 45, scope: !58)
!70 = !DILocation(line: 9, column: 33, scope: !58)
!71 = !DILocation(line: 9, column: 31, scope: !58)
!72 = !DILocation(line: 9, column: 26, scope: !58)
!73 = distinct !{!73, !61, !74, !54}
!74 = !DILocation(line: 9, column: 57, scope: !23)
!75 = !DILocation(line: 10, column: 8, scope: !25)
!76 = !DILocation(line: 10, column: 12, scope: !25)
!77 = !DILocation(line: 10, column: 19, scope: !78)
!78 = distinct !DILexicalBlock(scope: !25, file: !1, line: 10, column: 3)
!79 = !DILocation(line: 10, column: 23, scope: !78)
!80 = !DILocation(line: 10, column: 21, scope: !78)
!81 = !DILocation(line: 10, column: 3, scope: !25)
!82 = !DILocation(line: 10, column: 3, scope: !78)
!83 = !DILocation(line: 10, column: 37, scope: !78)
!84 = !DILocation(line: 10, column: 39, scope: !78)
!85 = !DILocation(line: 10, column: 47, scope: !78)
!86 = !DILocation(line: 10, column: 45, scope: !78)
!87 = !DILocation(line: 10, column: 52, scope: !78)
!88 = !DILocation(line: 10, column: 54, scope: !78)
!89 = !DILocation(line: 10, column: 49, scope: !78)
!90 = !DILocation(line: 10, column: 33, scope: !78)
!91 = !DILocation(line: 10, column: 31, scope: !78)
!92 = !DILocation(line: 10, column: 26, scope: !78)
!93 = distinct !{!93, !81, !94, !54}
!94 = !DILocation(line: 10, column: 58, scope: !25)
!95 = !DILocation(line: 11, column: 8, scope: !27)
!96 = !DILocation(line: 11, column: 12, scope: !27)
!97 = !DILocation(line: 11, column: 19, scope: !98)
!98 = distinct !DILexicalBlock(scope: !27, file: !1, line: 11, column: 3)
!99 = !DILocation(line: 11, column: 23, scope: !98)
!100 = !DILocation(line: 11, column: 21, scope: !98)
!101 = !DILocation(line: 11, column: 3, scope: !27)
!102 = !DILocation(line: 11, column: 3, scope: !98)
!103 = !DILocation(line: 11, column: 37, scope: !98)
!104 = !DILocation(line: 11, column: 39, scope: !98)
!105 = !DILocation(line: 11, column: 48, scope: !98)
!106 = !DILocation(line: 11, column: 50, scope: !98)
!107 = !DILocation(line: 11, column: 45, scope: !98)
!108 = !DILocation(line: 11, column: 33, scope: !98)
!109 = !DILocation(line: 11, column: 31, scope: !98)
!110 = !DILocation(line: 11, column: 26, scope: !98)
!111 = distinct !{!111, !101, !112, !54}
!112 = !DILocation(line: 11, column: 53, scope: !27)
!113 = !DILocation(line: 12, column: 10, scope: !13)
!114 = !DILocation(line: 13, column: 1, scope: !13)
!115 = !DILocation(line: 12, column: 3, scope: !13)
!116 = distinct !DISubprogram(name: "leaf", scope: !1, file: !1, line: 14, type: !14, scopeLine: 14, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !117)
!117 = !{!118}
!118 = !DILocalVariable(name: "x", arg: 1, scope: !116, file: !1, line: 14, type: !16)
!119 = !DILocation(line: 14, column: 25, scope: !116)
!120 = !DILocation(line: 14, column: 42, scope: !116)
!121 = !DILocation(line: 14, column: 40, scope: !116)
!122 = !DILocation(line: 14, column: 44, scope: !116)
!123 = !DILocation(line: 14, column: 30, scope: !116)
!124 = distinct !DISubprogram(name: "mid", scope: !1, file: !1, line: 15, type: !14, scopeLine: 15, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !125)
!125 = !{!126}
!126 = !DILocalVariable(name: "x", arg: 1, scope: !124, file: !1, line: 15, type: !16)
!127 = !DILocation(line: 15, column: 24, scope: !124)
!128 = !DILocation(line: 15, column: 36, scope: !124)
!129 = !{!130, !130, i64 0}
!130 = !{!"any pointer", !30, i64 0}
!131 = !DILocation(line: 15, column: 40, scope: !124)
!132 = !DILocation(line: 15, column: 36, scope: !133)
!133 = !DILexicalBlockFile(scope: !124, file: !1, discriminator: 387973143)
!134 = !DILocation(line: 15, column: 43, scope: !124)
!135 = !DILocation(line: 15, column: 29, scope: !124)
!136 = distinct !DISubprogram(name: "dispatch3", scope: !1, file: !1, line: 16, type: !137, scopeLine: 16, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !140)
!137 = !DISubroutineType(types: !138)
!138 = !{!16, !16, !139}
!139 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!140 = !{!141, !142, !143}
!141 = !DILocalVariable(name: "x", arg: 1, scope: !136, file: !1, line: 16, type: !16)
!142 = !DILocalVariable(name: "t", arg: 2, scope: !136, file: !1, line: 16, type: !139)
!143 = !DILocalVariable(name: "r", scope: !136, file: !1, line: 17, type: !16)
!144 = !DILocation(line: 16, column: 30, scope: !136)
!145 = !DILocation(line: 16, column: 39, scope: !136)
!146 = !DILocation(line: 17, column: 3, scope: !136)
!147 = !DILocation(line: 17, column: 7, scope: !136)
!148 = !DILocation(line: 17, column: 11, scope: !136)
!149 = !DILocation(line: 17, column: 13, scope: !136)
!150 = !DILocation(line: 17, column: 11, scope: !151)
!151 = !DILexicalBlockFile(scope: !136, file: !1, discriminator: 387973143)
!152 = !DILocation(line: 18, column: 7, scope: !153)
!153 = distinct !DILexicalBlock(scope: !136, file: !1, line: 18, column: 7)
!154 = !DILocation(line: 18, column: 9, scope: !153)
!155 = !DILocation(line: 19, column: 18, scope: !153)
!156 = !DILocation(line: 19, column: 10, scope: !157)
!157 = !DILexicalBlockFile(scope: !153, file: !1, discriminator: 455082023)
!158 = !DILocation(line: 19, column: 7, scope: !153)
!159 = !DILocation(line: 19, column: 5, scope: !153)
!160 = !DILocation(line: 20, column: 10, scope: !136)
!161 = !DILocation(line: 21, column: 1, scope: !136)
!162 = !DILocation(line: 20, column: 3, scope: !136)
!163 = distinct !DISubprogram(name: "dispatch4", scope: !1, file: !1, line: 22, type: !137, scopeLine: 22, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !164)
!164 = !{!165, !166}
!165 = !DILocalVariable(name: "x", arg: 1, scope: !163, file: !1, line: 22, type: !16)
!166 = !DILocalVariable(name: "t", arg: 2, scope: !163, file: !1, line: 22, type: !139)
!167 = !DILocation(line: 22, column: 30, scope: !163)
!168 = !DILocation(line: 22, column: 39, scope: !163)
!169 = !DILocation(line: 22, column: 57, scope: !163)
!170 = !DILocation(line: 22, column: 59, scope: !163)
!171 = !DILocation(line: 22, column: 57, scope: !172)
!172 = !DILexicalBlockFile(scope: !163, file: !1, discriminator: 387973143)
!173 = !DILocation(line: 22, column: 50, scope: !163)

;--- user.ll
source_filename = "user.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"

; Function Attrs: mustprogress uwtable
define dso_local i32 @user(i32 noundef %0, ptr noundef %1) #0 !dbg !9 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store i32 %0, ptr %3, align 4, !tbaa !19
    #dbg_declare(ptr %3, !17, !DIExpression(), !23)
  store ptr %1, ptr %4, align 8, !tbaa !24
    #dbg_declare(ptr %4, !18, !DIExpression(), !26)
  call void @llvm.pseudoprobe(i64 856900610518880750, i64 1, i32 0, i64 -1), !dbg !27
  %5 = load i32, ptr %3, align 4, !dbg !27, !tbaa !19
  %6 = load ptr, ptr %4, align 8, !dbg !28, !tbaa !24
  %7 = call i32 @dispatch3(i32 noundef %5, ptr noundef %6), !dbg !29
  %8 = load i32, ptr %3, align 4, !dbg !31, !tbaa !19
  %9 = load ptr, ptr %4, align 8, !dbg !32, !tbaa !24
  %10 = call i32 @dispatch4(i32 noundef %8, ptr noundef %9), !dbg !33
  %11 = add nsw i32 %7, %10, !dbg !35
  ret i32 %11, !dbg !36
}

declare !dbg !37 dso_local i32 @dispatch3(i32 noundef, ptr noundef) #1

declare !dbg !38 dso_local i32 @dispatch4(i32 noundef, ptr noundef) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #2

attributes #0 = { "use-sample-profile" mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { "use-sample-profile" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { "use-sample-profile" nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}
!llvm.pseudo_probe_desc = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "user.cpp", directory: "/data/users/kupathak/llvm21-importhint/limits", checksumkind: CSK_MD5, checksum: "356a21457fa455c0e9440d69eb856dcf")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{i32 7, !"frame-pointer", i32 1}
!7 = !{!"clang version 21.1.0 (ssh://git.vip.facebook.com/data/gitrepos/osmeta/external/llvm-project 912e119fd9606d5816cce0d2f42ff0f988242ef6)"}
!8 = !{i64 856900610518880750, i64 562954248388607, !"user"}
!9 = distinct !DISubprogram(name: "user", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
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
!23 = !DILocation(line: 3, column: 25, scope: !9)
!24 = !{!25, !25, i64 0}
!25 = !{!"any pointer", !21, i64 0}
!26 = !DILocation(line: 3, column: 34, scope: !9)
!27 = !DILocation(line: 3, column: 62, scope: !9)
!28 = !DILocation(line: 3, column: 65, scope: !9)
!29 = !DILocation(line: 3, column: 52, scope: !30)
!30 = !DILexicalBlockFile(scope: !9, file: !1, discriminator: 455082007)
!31 = !DILocation(line: 3, column: 80, scope: !9)
!32 = !DILocation(line: 3, column: 83, scope: !9)
!33 = !DILocation(line: 3, column: 70, scope: !34)
!34 = !DILexicalBlockFile(scope: !9, file: !1, discriminator: 455082015)
!35 = !DILocation(line: 3, column: 68, scope: !9)
!36 = !DILocation(line: 3, column: 45, scope: !9)
!37 = !DISubprogram(name: "dispatch3", scope: !1, file: !1, line: 1, type: !10, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!38 = !DISubprogram(name: "dispatch4", scope: !1, file: !1, line: 2, type: !10, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)

;--- nested.txt
user:3000000:1000000
 1: 1000000
 2: 1000000 dispatch3:1000000
 3: 1000000 dispatch4:1000000
 !CFGChecksum: 562954248388607
dispatch3:2000000:1000000
 1: 1000000
 3: 5
 4: 5 called2:5
 5: 1000000
 2: called2:900000
  1: 900000
  !CFGChecksum: 415320791528
  !Attributes: 3
 !CFGChecksum: 563003850888502
 !Attributes: 1
dispatch4:1000000:1000000
 1: 1000000
 2: mid:500000
  1: 500000
  2: 500000 called2:2000
  2: leaf:400000
   1: 400000
   !CFGChecksum: 4294967295
   !Attributes: 3
  !CFGChecksum: 281479271677951
  !Attributes: 3
 !CFGChecksum: 281479271677951
 !Attributes: 1

;--- recursive.txt
user:3000000:1000000
 1: 1000000
 2: 1000000 dispatch3:1000000
 3: 1000000 dispatch4:1000000
 !CFGChecksum: 562954248388607
dispatch3:2000000:1000000
 1: 1000000
 3: 5
 4: 5 called2:5
 5: 1000000
 2: called2:900000
  1: 900000
  !CFGChecksum: 415320791528
  !Attributes: 3
 !CFGChecksum: 563003850888502
 !Attributes: 1
dispatch4:1000000:1000000
 1: 1000000
 2: mid:500000
  1: 500000
  2: 500000 called2:2000
  2: mid:450000
   1: 450000
   2: leaf:400000
    1: 400000
    !CFGChecksum: 4294967295
    !Attributes: 3
   !CFGChecksum: 281479271677951
   !Attributes: 3
  !CFGChecksum: 281479271677951
  !Attributes: 3
 !CFGChecksum: 281479271677951
 !Attributes: 1

;--- cold-leaf.txt
user:3000000:1000000
 1: 1000000
 2: 1000000 dispatch3:1000000
 3: 1000000 dispatch4:1000000
 !CFGChecksum: 562954248388607
dispatch3:2000000:1000000
 1: 1000000
 3: 5
 4: 5 called2:5
 5: 1000000
 2: called2:900000
  1: 900000
  !CFGChecksum: 415320791528
  !Attributes: 3
 !CFGChecksum: 563003850888502
 !Attributes: 1
dispatch4:1000000:1000000
 1: 1000000
 2: mid:500000
  1: 500000
  2: 500000 called2:2000
  2: leaf:2
   1: 2
   !CFGChecksum: 4294967295
   !Attributes: 1
  !CFGChecksum: 281479271677951
  !Attributes: 3
 !CFGChecksum: 281479271677951
 !Attributes: 1

;--- cs-hot.txt
[dispatch4]:1000000:1000000
 1: 1000000
 !CFGChecksum: 281479271677951
 !Attributes: 1
[dispatch4:2 @ mid]:500000:500000
 1: 500000
 2: 500000 called2:2000
 !CFGChecksum: 281479271677951
 !Attributes: 3
[dispatch4:2 @ mid:2 @ leaf]:400000:400000
 1: 400000
 !CFGChecksum: 4294967295
 !Attributes: 3

;--- cs-cold.txt
[dispatch4]:1000000:1000000
 1: 1000000
 !CFGChecksum: 281479271677951
 !Attributes: 1
[dispatch4:2 @ mid]:500000:2
 1: 500000
 2: 500000 called2:2000
 !CFGChecksum: 281479271677951
 !Attributes: 1
[dispatch4:2 @ mid:2 @ leaf]:400000:400000
 1: 400000
 !CFGChecksum: 4294967295
 !Attributes: 3
