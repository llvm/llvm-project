; REQUIRES: x86_64-linux
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/profile-mismatch.prof -report-profile-staleness -persist-profile-staleness -S 2>%t -o %t.ll
; RUN: FileCheck %s --input-file %t
; RUN: FileCheck %s --input-file %t.ll -check-prefix=CHECK-MD
; RUN: llc < %t.ll -filetype=obj -o %t.obj
; RUN: llvm-objdump --section-headers %t.obj | FileCheck %s --check-prefix=CHECK-OBJ
; RUN: llc < %t.ll -filetype=asm -o - | FileCheck %s --check-prefix=CHECK-ASM

; CHECK: (1/3) of callsites' profile are invalid and (15/50) of samples are discarded due to callsite location mismatch.

; CHECK-MD: ![[#]] = !{!"NumMismatchedCallsites", i64 1, !"NumRecoveredCallsites", i64 0, !"TotalProfiledCallsites", i64 3, !"MismatchedCallsiteSamples", i64 15, !"RecoveredCallsiteSamples", i64 0}

; CHECK-OBJ: .llvm_stats

; CHECK-ASM: .ascii	"NumMismatchedCallsites"
; CHECK-ASM: .byte	4
; CHECK-ASM: .ascii	"MQ=="
; CHECK-ASM: .byte	21
; CHECK-ASM: .ascii	"NumRecoveredCallsites"
; CHECK-ASM: .byte	4
; CHECK-ASM: .ascii	"MA=="
; CHECK-ASM: .byte	22
; CHECK-ASM: .ascii	"TotalProfiledCallsites"
; CHECK-ASM: .byte	4
; CHECK-ASM: .ascii	"Mw=="
; CHECK-ASM: .byte	25
; CHECK-ASM: .ascii	"MismatchedCallsiteSamples"
; CHECK-ASM: .byte	4
; CHECK-ASM: .ascii	"MTU="
; CHECK-ASM: .byte	24
; CHECK-ASM: .ascii	"RecoveredCallsiteSamples"
; CHECK-ASM: .byte	4
; CHECK-ASM: .ascii	"MA=="


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = dso_local global i32 0, align 4, !dbg !0

; Function Attrs: nounwind uwtable
define dso_local i32 @foo(i32 noundef %x) #0 !dbg !12 {
entry:
  %y = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 %x, metadata !16, metadata !DIExpression()), !dbg !18
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %y), !dbg !19
  call void @llvm.dbg.declare(metadata ptr %y, metadata !17, metadata !DIExpression()), !dbg !20
  %add = add nsw i32 %x, 1, !dbg !21
  store volatile i32 %add, ptr %y, align 4, !dbg !20, !tbaa !22
  %y.0. = load volatile i32, ptr %y, align 4, !dbg !26, !tbaa !22
  %add1 = add nsw i32 %y.0., 1, !dbg !27
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %y), !dbg !28
  ret i32 %add1, !dbg !29
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @bar(i32 noundef %x) #3 !dbg !30 {
entry:
  call void @llvm.dbg.value(metadata i32 %x, metadata !32, metadata !DIExpression()), !dbg !33
  %add = add nsw i32 %x, 2, !dbg !34
  ret i32 %add, !dbg !35
}

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @matched(i32 noundef %x) #3 !dbg !36 {
entry:
  call void @llvm.dbg.value(metadata i32 %x, metadata !38, metadata !DIExpression()), !dbg !39
  %add = add nsw i32 %x, 3, !dbg !40
  ret i32 %add, !dbg !41
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #0 !dbg !42 {
entry:
  call void @llvm.dbg.value(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !52
  br label %for.cond, !dbg !53

for.cond:                                         ; preds = %for.cond.cleanup3, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc8, %for.cond.cleanup3 ], !dbg !52
  call void @llvm.dbg.value(metadata i32 %i.0, metadata !46, metadata !DIExpression()), !dbg !52
  %cmp = icmp ult i32 %i.0, 1000, !dbg !54
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !56

for.cond.cleanup:                                 ; preds = %for.cond
  ret i32 0, !dbg !58

for.body:                                         ; preds = %for.cond
  call void @llvm.dbg.value(metadata i32 0, metadata !48, metadata !DIExpression()), !dbg !59
  br label %for.cond1, !dbg !60

for.cond1:                                        ; preds = %for.body4, %for.body
  %a.0 = phi i32 [ 0, %for.body ], [ %inc, %for.body4 ], !dbg !59
  call void @llvm.dbg.value(metadata i32 %a.0, metadata !48, metadata !DIExpression()), !dbg !59
  %cmp2 = icmp ult i32 %a.0, 10000, !dbg !61
  br i1 %cmp2, label %for.body4, label %for.cond.cleanup3, !dbg !64

for.cond.cleanup3:                                ; preds = %for.cond1
  %inc8 = add nuw nsw i32 %i.0, 1, !dbg !66
  call void @llvm.dbg.value(metadata i32 %inc8, metadata !46, metadata !DIExpression()), !dbg !52
  br label %for.cond, !dbg !68, !llvm.loop !69

for.body4:                                        ; preds = %for.cond1
  %0 = load volatile i32, ptr @x, align 4, !dbg !73, !tbaa !22
  %call = call i32 @matched(i32 noundef %0), !dbg !75
  store volatile i32 %call, ptr @x, align 4, !dbg !76, !tbaa !22
  %1 = load volatile i32, ptr @x, align 4, !dbg !77, !tbaa !22
  %call5 = call i32 @foo(i32 noundef %1), !dbg !78
  store volatile i32 %call5, ptr @x, align 4, !dbg !79, !tbaa !22
  %2 = load volatile i32, ptr @x, align 4, !dbg !80, !tbaa !22
  %call6 = call i32 @bar(i32 noundef %2), !dbg !81
  store volatile i32 %call6, ptr @x, align 4, !dbg !82, !tbaa !22
  %inc = add nuw nsw i32 %a.0, 1, !dbg !83
  call void @llvm.dbg.value(metadata i32 %inc, metadata !48, metadata !DIExpression()), !dbg !59
  br label %for.cond1, !dbg !85, !llvm.loop !86
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #4

attributes #0 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { argmemonly mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { noinline nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #4 = { nocallback nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "test")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !6)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 7, !"uwtable", i32 2}
!11 = !{!""}
!12 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 2, type: !13, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{!6, !6}
!15 = !{!16, !17}
!16 = !DILocalVariable(name: "x", arg: 1, scope: !12, file: !3, line: 2, type: !6)
!17 = !DILocalVariable(name: "y", scope: !12, file: !3, line: 3, type: !5)
!18 = !DILocation(line: 0, scope: !12)
!19 = !DILocation(line: 3, column: 3, scope: !12)
!20 = !DILocation(line: 3, column: 16, scope: !12)
!21 = !DILocation(line: 3, column: 22, scope: !12)
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C/C++ TBAA"}
!26 = !DILocation(line: 4, column: 10, scope: !12)
!27 = !DILocation(line: 4, column: 12, scope: !12)
!28 = !DILocation(line: 5, column: 1, scope: !12)
!29 = !DILocation(line: 4, column: 3, scope: !12)
!30 = distinct !DISubprogram(name: "bar", scope: !3, file: !3, line: 7, type: !13, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !31)
!31 = !{!32}
!32 = !DILocalVariable(name: "x", arg: 1, scope: !30, file: !3, line: 7, type: !6)
!33 = !DILocation(line: 0, scope: !30)
!34 = !DILocation(line: 8, column: 12, scope: !30)
!35 = !DILocation(line: 8, column: 3, scope: !30)
!36 = distinct !DISubprogram(name: "matched", scope: !3, file: !3, line: 11, type: !13, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !37)
!37 = !{!38}
!38 = !DILocalVariable(name: "x", arg: 1, scope: !36, file: !3, line: 11, type: !6)
!39 = !DILocation(line: 0, scope: !36)
!40 = !DILocation(line: 12, column: 12, scope: !36)
!41 = !DILocation(line: 12, column: 3, scope: !36)
!42 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 15, type: !43, scopeLine: 15, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !45)
!43 = !DISubroutineType(types: !44)
!44 = !{!6}
!45 = !{!46, !48}
!46 = !DILocalVariable(name: "i", scope: !47, file: !3, line: 16, type: !6)
!47 = distinct !DILexicalBlock(scope: !42, file: !3, line: 16, column: 3)
!48 = !DILocalVariable(name: "a", scope: !49, file: !3, line: 17, type: !6)
!49 = distinct !DILexicalBlock(scope: !50, file: !3, line: 17, column: 5)
!50 = distinct !DILexicalBlock(scope: !51, file: !3, line: 16, column: 34)
!51 = distinct !DILexicalBlock(scope: !47, file: !3, line: 16, column: 3)
!52 = !DILocation(line: 0, scope: !47)
!53 = !DILocation(line: 16, column: 8, scope: !47)
!54 = !DILocation(line: 16, column: 21, scope: !55)
!55 = !DILexicalBlockFile(scope: !51, file: !3, discriminator: 2)
!56 = !DILocation(line: 16, column: 3, scope: !57)
!57 = !DILexicalBlockFile(scope: !47, file: !3, discriminator: 2)
!58 = !DILocation(line: 23, column: 1, scope: !42)
!59 = !DILocation(line: 0, scope: !49)
!60 = !DILocation(line: 17, column: 10, scope: !49)
!61 = !DILocation(line: 17, column: 23, scope: !62)
!62 = !DILexicalBlockFile(scope: !63, file: !3, discriminator: 2)
!63 = distinct !DILexicalBlock(scope: !49, file: !3, line: 17, column: 5)
!64 = !DILocation(line: 17, column: 5, scope: !65)
!65 = !DILexicalBlockFile(scope: !49, file: !3, discriminator: 2)
!66 = !DILocation(line: 16, column: 30, scope: !67)
!67 = !DILexicalBlockFile(scope: !51, file: !3, discriminator: 4)
!68 = !DILocation(line: 16, column: 3, scope: !67)
!69 = distinct !{!69, !70, !71, !72}
!70 = !DILocation(line: 16, column: 3, scope: !47)
!71 = !DILocation(line: 22, column: 3, scope: !47)
!72 = !{!"llvm.loop.mustprogress"}
!73 = !DILocation(line: 18, column: 19, scope: !74)
!74 = distinct !DILexicalBlock(scope: !63, file: !3, line: 17, column: 37)
!75 = !DILocation(line: 18, column: 11, scope: !74)
!76 = !DILocation(line: 18, column: 9, scope: !74)
!77 = !DILocation(line: 19, column: 15, scope: !74)
!78 = !DILocation(line: 19, column: 11, scope: !74)
!79 = !DILocation(line: 19, column: 9, scope: !74)
!80 = !DILocation(line: 20, column: 15, scope: !74)
!81 = !DILocation(line: 20, column: 11, scope: !74)
!82 = !DILocation(line: 20, column: 9, scope: !74)
!83 = !DILocation(line: 17, column: 33, scope: !84)
!84 = !DILexicalBlockFile(scope: !63, file: !3, discriminator: 4)
!85 = !DILocation(line: 17, column: 5, scope: !84)
!86 = distinct !{!86, !87, !88, !72}
!87 = !DILocation(line: 17, column: 5, scope: !49)
!88 = !DILocation(line: 21, column: 5, scope: !49)
