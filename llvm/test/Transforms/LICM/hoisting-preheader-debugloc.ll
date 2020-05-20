; RUN: opt -S -licm < %s | FileCheck %s

; CHECK: %arrayidx4.promoted = load i32, i32* %arrayidx4, align 1, !tbaa !59

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i16 0, align 2, !dbg !0
@b = dso_local local_unnamed_addr global i16 0, align 2, !dbg !6

define dso_local void @c(i32 %d) local_unnamed_addr #0 !dbg !13 {
entry:
  call void @llvm.dbg.value(metadata i32 undef, metadata !18, metadata !DIExpression()), !dbg !19
  ret void, !dbg !20
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

define dso_local signext i16 @e(i32 %d) local_unnamed_addr #0 !dbg !21 {
entry:
  call void @llvm.dbg.value(metadata i32 %d, metadata !25, metadata !DIExpression()), !dbg !26
  %conv = trunc i32 %d to i16, !dbg !27
  ret i16 %conv, !dbg !28
}

define dso_local signext i16 @g() local_unnamed_addr #2 !dbg !29 {
entry:
  %l_284 = alloca [2 x [3 x [6 x i32]]], align 16
  %0 = bitcast [2 x [3 x [6 x i32]]]* %l_284 to i8*, !dbg !39
  call void @llvm.lifetime.start.p0i8(i64 144, i8* nonnull %0) #4, !dbg !39
  call void @llvm.dbg.declare(metadata [2 x [3 x [6 x i32]]]* %l_284, metadata !33, metadata !DIExpression()), !dbg !40
  %1 = load i16, i16* @a, align 2, !dbg !41, !tbaa !44
  %cmp11 = icmp sgt i16 %1, -1, !dbg !48
  br i1 %cmp11, label %for.body.lr.ph, label %cleanup, !dbg !49

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body, !dbg !49

for.body:                                         ; preds = %for.body.lr.ph, %for.cond2
  %call = call signext i16 @e(i32 1), !dbg !50
  %tobool = icmp eq i16 %call, 0, !dbg !50
  br i1 %tobool, label %for.cond2, label %for.body.cleanup_crit_edge, !dbg !53

for.cond2:                                        ; preds = %for.body
  %arrayidx4 = getelementptr inbounds [2 x [3 x [6 x i32]]], [2 x [3 x [6 x i32]]]* %l_284, i64 0, i64 1, i64 2, i64 5, !dbg !54
  store i32 0, i32* %arrayidx4, align 4, !dbg !58, !tbaa !59
  %arrayidx8 = getelementptr inbounds [2 x [3 x [6 x i32]]], [2 x [3 x [6 x i32]]]* %l_284, i64 0, i64 1, i64 1, i64 4, !dbg !61
  %2 = load i32, i32* %arrayidx8, align 8, !dbg !61, !tbaa !59
  %conv9 = trunc i32 %2 to i16, !dbg !61
  store i16 %conv9, i16* @b, align 2, !dbg !62, !tbaa !44
  %3 = load i16, i16* @a, align 2, !dbg !41, !tbaa !44
  %cmp = icmp sgt i16 %3, -1, !dbg !48
  br i1 %cmp, label %for.body, label %for.cond.cleanup_crit_edge, !dbg !49, !llvm.loop !63

for.cond.cleanup_crit_edge:                       ; preds = %for.cond2
  br label %cleanup, !dbg !49

for.body.cleanup_crit_edge:                       ; preds = %for.body
  br label %cleanup, !dbg !53

cleanup:                                          ; preds = %for.body.cleanup_crit_edge, %for.cond.cleanup_crit_edge, %entry
  call void @llvm.lifetime.end.p0i8(i64 144, i8* nonnull %0) #4, !dbg !66
  ret i16 1, !dbg !66
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #3

; Function Attrs: norecurse nounwind readnone uwtable
define internal signext i16 @f(i32 %d) #0 !dbg !67 {
entry:
  call void @llvm.dbg.value(metadata i32 %d, metadata !69, metadata !DIExpression()), !dbg !70
  ret i16 undef, !dbg !71
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #3

; Function Attrs: nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #2 !dbg !72 {
entry:
  %call = call signext i16 @g(), !dbg !75
  ret i32 0, !dbg !76
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

attributes #0 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind willreturn }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 237d0e3c0416abf9919406bcc92874cfd15f5e0c)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "a.c", directory: "/home/davide/lldb-build/bin")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 237d0e3c0416abf9919406bcc92874cfd15f5e0c)"}
!13 = distinct !DISubprogram(name: "c", scope: !3, file: !3, line: 2, type: !14, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !17)
!14 = !DISubroutineType(types: !15)
!15 = !{null, !16}
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !{!18}
!18 = !DILocalVariable(name: "d", arg: 1, scope: !13, file: !3, line: 2, type: !16)
!19 = !DILocation(line: 0, scope: !13)
!20 = !DILocation(line: 2, column: 13, scope: !13)
!21 = distinct !DISubprogram(name: "e", scope: !3, file: !3, line: 3, type: !22, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !24)
!22 = !DISubroutineType(types: !23)
!23 = !{!8, !16}
!24 = !{!25}
!25 = !DILocalVariable(name: "d", arg: 1, scope: !21, file: !3, line: 3, type: !16)
!26 = !DILocation(line: 0, scope: !21)
!27 = !DILocation(line: 3, column: 22, scope: !21)
!28 = !DILocation(line: 3, column: 15, scope: !21)
!29 = distinct !DISubprogram(name: "g", scope: !3, file: !3, line: 5, type: !30, scopeLine: 5, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !32)
!30 = !DISubroutineType(types: !31)
!31 = !{!8}
!32 = !{!33}
!33 = !DILocalVariable(name: "l_284", scope: !29, file: !3, line: 6, type: !34)
!34 = !DICompositeType(tag: DW_TAG_array_type, baseType: !16, size: 1152, elements: !35)
!35 = !{!36, !37, !38}
!36 = !DISubrange(count: 2)
!37 = !DISubrange(count: 3)
!38 = !DISubrange(count: 6)
!39 = !DILocation(line: 6, column: 3, scope: !29)
!40 = !DILocation(line: 6, column: 7, scope: !29)
!41 = !DILocation(line: 7, column: 10, scope: !42)
!42 = distinct !DILexicalBlock(scope: !43, file: !3, line: 7, column: 3)
!43 = distinct !DILexicalBlock(scope: !29, file: !3, line: 7, column: 3)
!44 = !{!45, !45, i64 0}
!45 = !{!"short", !46, i64 0}
!46 = !{!"omnipotent char", !47, i64 0}
!47 = !{!"Simple C/C++ TBAA"}
!48 = !DILocation(line: 7, column: 12, scope: !42)
!49 = !DILocation(line: 7, column: 3, scope: !43)
!50 = !DILocation(line: 8, column: 9, scope: !51)
!51 = distinct !DILexicalBlock(scope: !52, file: !3, line: 8, column: 9)
!52 = distinct !DILexicalBlock(scope: !42, file: !3, line: 7, column: 19)
!53 = !DILocation(line: 8, column: 9, scope: !52)
!54 = !DILocation(line: 11, column: 7, scope: !55)
!55 = distinct !DILexicalBlock(scope: !56, file: !3, line: 10, column: 14)
!56 = distinct !DILexicalBlock(scope: !57, file: !3, line: 10, column: 5)
!57 = distinct !DILexicalBlock(scope: !52, file: !3, line: 10, column: 5)
!58 = !DILocation(line: 11, column: 22, scope: !55)
!59 = !{!60, !60, i64 0}
!60 = !{!"int", !46, i64 0}
!61 = !DILocation(line: 16, column: 9, scope: !52)
!62 = !DILocation(line: 16, column: 7, scope: !52)
!63 = distinct !{!63, !49, !64, !65}
!64 = !DILocation(line: 17, column: 3, scope: !43)
!65 = !{!"llvm.loop.unroll.disable"}
!66 = !DILocation(line: 18, column: 1, scope: !29)
!67 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 4, type: !22, scopeLine: 4, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !68)
!68 = !{!69}
!69 = !DILocalVariable(name: "d", arg: 1, scope: !67, file: !3, line: 4, type: !16)
!70 = !DILocation(line: 0, scope: !67)
!71 = !DILocation(line: 4, column: 21, scope: !67)
!72 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 19, type: !73, scopeLine: 19, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!73 = !DISubroutineType(types: !74)
!74 = !{!16}
!75 = !DILocation(line: 19, column: 14, scope: !72)
!76 = !DILocation(line: 19, column: 19, scope: !72)
