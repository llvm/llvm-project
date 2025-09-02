; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-icp-factor.prof -S -sample-profile-prioritized-inline=1 2>&1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

; Function Attrs: nounwind uwtable
define dso_local i32 @bar(i32 %arg) #0 !dbg !13 {
bb:
  %i = alloca i32, align 4
  store i32 %arg, ptr %i, align 4, !tbaa !19
  call void @llvm.dbg.declare(metadata ptr %i, metadata !18, metadata !DIExpression()), !dbg !23
  call void @llvm.pseudoprobe(i64 -2012135647395072713, i64 1, i32 0, i64 -1), !dbg !24
  %i1 = load i32, ptr %i, align 4, !dbg !24, !tbaa !19
  %i2 = add nsw i32 %i1, 1, !dbg !24
  store i32 %i2, ptr %i, align 4, !dbg !24, !tbaa !19
  %i3 = load i32, ptr %i, align 4, !dbg !25, !tbaa !19
  %i4 = add nsw i32 %i3, 1, !dbg !26
  ret i32 %i4, !dbg !27
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define dso_local i32 @baz(i32 %arg) #0 !dbg !28 {
bb:
  %i = alloca i32, align 4
  store i32 %arg, ptr %i, align 4, !tbaa !19
  call void @llvm.dbg.declare(metadata ptr %i, metadata !30, metadata !DIExpression()), !dbg !31
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 1, i32 0, i64 -1), !dbg !32
  %i1 = load i32, ptr %i, align 4, !dbg !32, !tbaa !19
  %i2 = add nsw i32 %i1, 10, !dbg !33
  ret i32 %i2, !dbg !34
}

; Function Attrs: nounwind uwtable
define dso_local i32 @foo(i32 %arg, ptr %arg1) #0 !dbg !35 {
bb:
  %i = alloca i32, align 4
  %i2 = alloca ptr, align 8
  store i32 %arg, ptr %i, align 4, !tbaa !19
  call void @llvm.dbg.declare(metadata ptr %i, metadata !42, metadata !DIExpression()), !dbg !44
  store ptr %arg1, ptr %i2, align 8, !tbaa !45
  call void @llvm.dbg.declare(metadata ptr %i2, metadata !43, metadata !DIExpression()), !dbg !47
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 1, i32 0, i64 -1), !dbg !48
  %i3 = load ptr, ptr %i2, align 8, !dbg !48, !tbaa !45
  %i4 = load i32, ptr %i, align 4, !dbg !49, !tbaa !19
  %i6 = call i32 (i32, ...) %i3(i32 %i4), !dbg !50
  ret i32 %i6, !dbg !52
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #0 !dbg !53 {
bb:
  %i = alloca i32, align 4
  %i1 = alloca ptr, align 8
  %i2 = alloca i32, align 4
  %i3 = alloca i32, align 4
  store i32 0, ptr %i, align 4
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 1, i32 0, i64 -1), !dbg !62
  call void @llvm.lifetime.start.p0(ptr %i1), !dbg !62
  call void @llvm.dbg.declare(metadata ptr %i1, metadata !57, metadata !DIExpression()), !dbg !63
  call void @llvm.lifetime.start.p0(ptr %i2), !dbg !64
  call void @llvm.dbg.declare(metadata ptr %i2, metadata !59, metadata !DIExpression()), !dbg !65
  store i32 0, ptr %i2, align 4, !dbg !65, !tbaa !19
  call void @llvm.lifetime.start.p0(ptr %i3), !dbg !66
  call void @llvm.dbg.declare(metadata ptr %i3, metadata !60, metadata !DIExpression()), !dbg !67
  store i32 0, ptr %i3, align 4, !dbg !67, !tbaa !19
  br label %bb7, !dbg !66

bb7:                                              ; preds = %bb25, %bb
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 2, i32 0, i64 -1), !dbg !68
  %i8 = load i32, ptr %i3, align 4, !dbg !68, !tbaa !19
  %i9 = icmp slt i32 %i8, 1000000000, !dbg !70
  br i1 %i9, label %bb12, label %bb10, !dbg !71

bb10:                                             ; preds = %bb7
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 3, i32 0, i64 -1), !dbg !72
  call void @llvm.lifetime.end.p0(ptr %i3), !dbg !72
  br label %bb28

bb12:                                             ; preds = %bb7
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 4, i32 0, i64 -1), !dbg !73
  %i13 = load i32, ptr %i3, align 4, !dbg !73, !tbaa !19
  %i14 = srem i32 %i13, 100, !dbg !76
  %i15 = icmp eq i32 %i14, 0, !dbg !77
  br i1 %i15, label %bb16, label %bb17, !dbg !78

bb16:                                             ; preds = %bb12
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 5, i32 0, i64 -1), !dbg !79
  store ptr @bar, ptr %i1, align 8, !dbg !79, !tbaa !45
  br label %bb18, !dbg !80

bb17:                                             ; preds = %bb12
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 6, i32 0, i64 -1), !dbg !81
  store ptr @baz, ptr %i1, align 8, !dbg !81, !tbaa !45
  br label %bb18

bb18:                                             ; preds = %bb17, %bb16
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 7, i32 0, i64 -1), !dbg !82
  %i19 = load i32, ptr %i3, align 4, !dbg !82, !tbaa !19
  %i20 = load ptr, ptr %i1, align 8, !dbg !83, !tbaa !45
  %i22 = call i32 @foo(i32 %i19, ptr %i20), !dbg !84
  %i23 = load i32, ptr %i2, align 4, !dbg !86, !tbaa !19
  %i24 = add nsw i32 %i23, %i22, !dbg !86
  store i32 %i24, ptr %i2, align 4, !dbg !86, !tbaa !19
  br label %bb25, !dbg !87

bb25:                                             ; preds = %bb18
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 8, i32 0, i64 -1), !dbg !88
  %i26 = load i32, ptr %i3, align 4, !dbg !88, !tbaa !19
  %i27 = add nsw i32 %i26, 1, !dbg !88
  store i32 %i27, ptr %i3, align 4, !dbg !88, !tbaa !19
  br label %bb7, !dbg !72, !llvm.loop !89

bb28:                                             ; preds = %bb10
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 9, i32 0, i64 -1), !dbg !92
  %i29 = load i32, ptr %i2, align 4, !dbg !92, !tbaa !19
  %i30 = call i32 (ptr, ...) @printf(ptr @.str, i32 %i29), !dbg !93
  call void @llvm.lifetime.end.p0(ptr %i2), !dbg !95
  call void @llvm.lifetime.end.p0(ptr %i1), !dbg !95
  ret i32 0, !dbg !96
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr nocapture) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr nocapture) #2

declare dso_local i32 @printf(ptr, ...)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #3

attributes #0 = { nounwind uwtable "disable-tail-calls"="true" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-sample-profile" "use-soft-float"="false" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6, !7}
!llvm.ident = !{!8}
!llvm.pseudo_probe_desc = !{!9, !10, !11, !12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 13.0.06)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"ThinLTO", i32 0}
!7 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!8 = !{!"clang version 13.0.0 "}
!9 = !{i64 -2012135647395072713, i64 4294967295, !"bar", null}
!10 = !{i64 7546896869197086323, i64 4294967295, !"baz", null}
!11 = !{i64 6699318081062747564, i64 281479271677951, !"foo", null}
!12 = !{i64 -2624081020897602054, i64 563125815542069, !"main", null}
!13 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 2, type: !14, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !17)
!14 = !DISubroutineType(types: !15)
!15 = !{!16, !16}
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !{!18}
!18 = !DILocalVariable(name: "x", arg: 1, scope: !13, file: !1, line: 2, type: !16)
!19 = !{!20, !20, i64 0}
!20 = !{!"int", !21, i64 0}
!21 = !{!"omnipotent char", !22, i64 0}
!22 = !{!"Simple C/C++ TBAA"}
!23 = !DILocation(line: 2, column: 13, scope: !13)
!24 = !DILocation(line: 4, column: 7, scope: !13)
!25 = !DILocation(line: 5, column: 12, scope: !13)
!26 = !DILocation(line: 5, column: 14, scope: !13)
!27 = !DILocation(line: 5, column: 5, scope: !13)
!28 = distinct !DISubprogram(name: "baz", scope: !1, file: !1, line: 9, type: !14, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !29)
!29 = !{!30}
!30 = !DILocalVariable(name: "x", arg: 1, scope: !28, file: !1, line: 9, type: !16)
!31 = !DILocation(line: 9, column: 13, scope: !28)
!32 = !DILocation(line: 10, column: 10, scope: !28)
!33 = !DILocation(line: 10, column: 12, scope: !28)
!34 = !DILocation(line: 10, column: 3, scope: !28)
!35 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 13, type: !36, scopeLine: 13, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !41)
!36 = !DISubroutineType(types: !37)
!37 = !{!16, !16, !38}
!38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !39, size: 64)
!39 = !DISubroutineType(types: !40)
!40 = !{!16, null}
!41 = !{!42, !43}
!42 = !DILocalVariable(name: "x", arg: 1, scope: !35, file: !1, line: 13, type: !16)
!43 = !DILocalVariable(name: "f", arg: 2, scope: !35, file: !1, line: 13, type: !38)
!44 = !DILocation(line: 13, column: 13, scope: !35)
!45 = !{!46, !46, i64 0}
!46 = !{!"any pointer", !21, i64 0}
!47 = !DILocation(line: 13, column: 22, scope: !35)
!48 = !DILocation(line: 14, column: 10, scope: !35)
!49 = !DILocation(line: 14, column: 12, scope: !35)
!50 = !DILocation(line: 14, column: 10, scope: !51)
;; A discriminator of 108527639 which is 0x6780017 in hexdecimal, stands for an indirect call probe
;; with an index of 2 and probe factor of 0.79.
!51 = !DILexicalBlockFile(scope: !35, file: !1, discriminator: 108527639)
!52 = !DILocation(line: 14, column: 3, scope: !35)
!53 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 17, type: !54, scopeLine: 18, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !56)
!54 = !DISubroutineType(types: !55)
!55 = !{!16}
!56 = !{!57, !59, !60}
!57 = !DILocalVariable(name: "x", scope: !53, file: !1, line: 19, type: !58)
!58 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!59 = !DILocalVariable(name: "sum", scope: !53, file: !1, line: 25, type: !16)
!60 = !DILocalVariable(name: "i", scope: !61, file: !1, line: 26, type: !16)
!61 = distinct !DILexicalBlock(scope: !53, file: !1, line: 26, column: 5)
!62 = !DILocation(line: 19, column: 3, scope: !53)
!63 = !DILocation(line: 19, column: 9, scope: !53)
!64 = !DILocation(line: 25, column: 5, scope: !53)
!65 = !DILocation(line: 25, column: 9, scope: !53)
!66 = !DILocation(line: 26, column: 10, scope: !61)
!67 = !DILocation(line: 26, column: 14, scope: !61)
!68 = !DILocation(line: 26, column: 21, scope: !69)
!69 = distinct !DILexicalBlock(scope: !61, file: !1, line: 26, column: 5)
!70 = !DILocation(line: 26, column: 23, scope: !69)
!71 = !DILocation(line: 26, column: 5, scope: !61)
!72 = !DILocation(line: 26, column: 5, scope: !69)
!73 = !DILocation(line: 27, column: 10, scope: !74)
!74 = distinct !DILexicalBlock(scope: !75, file: !1, line: 27, column: 10)
!75 = distinct !DILexicalBlock(scope: !69, file: !1, line: 26, column: 45)
!76 = !DILocation(line: 27, column: 12, scope: !74)
!77 = !DILocation(line: 27, column: 19, scope: !74)
!78 = !DILocation(line: 27, column: 10, scope: !75)
!79 = !DILocation(line: 28, column: 11, scope: !74)
!80 = !DILocation(line: 28, column: 9, scope: !74)
!81 = !DILocation(line: 30, column: 11, scope: !74)
!82 = !DILocation(line: 32, column: 17, scope: !75)
!83 = !DILocation(line: 32, column: 20, scope: !75)
!84 = !DILocation(line: 32, column: 13, scope: !85)
;; A discriminator of 116916311 which is 0x6f80057 in hexdecimal, stands for an indirect call probe
;; with an index of 10 and probe factor of 0.95.
!85 = !DILexicalBlockFile(scope: !75, file: !1, discriminator: 116916311)
!86 = !DILocation(line: 32, column: 11, scope: !75)
!87 = !DILocation(line: 33, column: 5, scope: !75)
!88 = !DILocation(line: 26, column: 41, scope: !69)
!89 = distinct !{!89, !71, !90, !91}
!90 = !DILocation(line: 33, column: 5, scope: !61)
!91 = !{!"llvm.loop.mustprogress"}
!92 = !DILocation(line: 34, column: 21, scope: !53)
!93 = !DILocation(line: 34, column: 5, scope: !94)
!94 = !DILexicalBlockFile(scope: !53, file: !1, discriminator: 104333335)
!95 = !DILocation(line: 36, column: 1, scope: !53)
!96 = !DILocation(line: 35, column: 5, scope: !53)

; CHECK: define dso_local i32 @main
; CHECK: %{{.+}} = call i32 (i32, ...) %{{.+}}(i32 %{{.+}}) #[[#]], !dbg ![[#DBGID:]], !prof ![[#PROF:]]

;; A discriminator of 106430487 which is 0x6580017 in hexdecimal, stands for an indirect call probe
;; with an index of 2 and probe factor of 0.75, which is from 0.95 * 0.79.
; CHECK: ![[#DBGID]] = !DILocation(line: [[#]], column: [[#]], scope: ![[#SCOPE:]], inlinedAt: ![[#]])
; CHECK: ![[#SCOPE]] = !DILexicalBlockFile(scope: ![[#]], file: ![[#]], discriminator: 106430487)

;; The remaining count of the second target (bar) should be from the original count multiplied by two callsite
;; factors, i.e, roughly 11259 * 0.95 * 0.79 = 8444.
; CHECK: ![[#PROF]] = !{!"VP", i32 0, i64 8444, i64 7546896869197086323, i64 -1, i64 -2012135647395072713, i64 8444}
