; Just run it through opt, no passes needed.
; RUN: opt < %s -S | FileCheck %s

; ModuleID = 'various_ir_values.c'
source_filename = "various_ir_values.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local void @foo(ptr %A) #0 !dbg !7 {
entry:
  %A.addr = alloca ptr, align 8, !DIAssignID !62
  %i = alloca i32, align 4
  tail call void @llvm.dbg.assign(metadata i1 undef, metadata !13, metadata !DIExpression(), metadata !62, metadata ptr %A.addr, metadata !DIExpression()), !dbg !20
  store ptr %A, ptr %A.addr, align 8, !tbaa !16
  tail call void @llvm.dbg.declare(metadata ptr %A.addr, metadata !13, metadata !DIExpression()), !dbg !20
  call void @llvm.lifetime.start.p0(i64 4, ptr %i) #3, !dbg !21
  tail call void @llvm.dbg.declare(metadata ptr %i, metadata !14, metadata !DIExpression()), !dbg !22
  store i32 0, ptr %i, align 4, !dbg !22, !tbaa !23
  br label %for.cond, !dbg !21

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4, !dbg !25, !tbaa !23
  %1 = load ptr, ptr %A.addr, align 8, !dbg !27, !tbaa !16
  %2 = load i32, ptr %1, align 4, !dbg !28, !tbaa !23
  %cmp = icmp slt i32 %0, %2, !dbg !29
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !30, !prof !61

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0(i64 4, ptr %i) #3, !dbg !31
  br label %for.end

for.body:                                         ; preds = %for.cond
  %3 = load ptr, ptr %A.addr, align 8, !dbg !32, !tbaa !16
  %4 = load i32, ptr %i, align 4, !dbg !33, !tbaa !23
  %idxprom = sext i32 %4 to i64, !dbg !32
  %arrayidx = getelementptr inbounds i32, ptr %3, i64 %idxprom, !dbg !32
  store i32 0, ptr %arrayidx, align 4, !dbg !34, !tbaa !23
  br label %for.inc, !dbg !32

for.inc:                                          ; preds = %for.body
  %5 = load i32, ptr %i, align 4, !dbg !35, !tbaa !23
  %inc = add nsw i32 %5, 1, !dbg !35
  store i32 %inc, ptr %i, align 4, !dbg !35, !tbaa !23
  br label %for.cond, !dbg !31, !llvm.loop !36

for.end:                                          ; preds = %for.cond.cleanup
  ret void, !dbg !38
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: nounwind uwtable
define dso_local void @bar(ptr %A) #0 !dbg !39 {
entry:
  %A.addr = alloca ptr, align 8
  %i = alloca i32, align 4
  store ptr %A, ptr %A.addr, align 8, !tbaa !16
  tail call void @llvm.dbg.declare(metadata ptr %A.addr, metadata !41, metadata !DIExpression()), !dbg !44
  call void @llvm.lifetime.start.p0(i64 4, ptr %i) #3, !dbg !45
  tail call void @llvm.dbg.declare(metadata ptr %i, metadata !42, metadata !DIExpression()), !dbg !46
  store i32 0, ptr %i, align 4, !dbg !46, !tbaa !23
  br label %for.cond, !dbg !45

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4, !dbg !47, !tbaa !23
  %1 = load ptr, ptr %A.addr, align 8, !dbg !49, !tbaa !16
  %2 = load i32, ptr %1, align 4, !dbg !50, !tbaa !23
  %cmp = icmp slt i32 %0, %2, !dbg !51
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !52

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0(i64 4, ptr %i) #3, !dbg !53
  br label %for.end

for.body:                                         ; preds = %for.cond
  %3 = load ptr, ptr %A.addr, align 8, !dbg !54, !tbaa !16
  %4 = load i32, ptr %i, align 4, !dbg !55, !tbaa !23
  %idxprom = sext i32 %4 to i64, !dbg !54
  %arrayidx = getelementptr inbounds i32, ptr %3, i64 %idxprom, !dbg !54
  store i32 0, ptr %arrayidx, align 4, !dbg !56, !tbaa !23
  br label %for.inc, !dbg !54

for.inc:                                          ; preds = %for.body
  %5 = load i32, ptr %i, align 4, !dbg !57, !tbaa !23
  %inc = add nsw i32 %5, 1, !dbg !57
  store i32 %inc, ptr %i, align 4, !dbg !57, !tbaa !23
  br label %for.cond, !dbg !53, !llvm.loop !58

for.end:                                          ; preds = %for.cond.cleanup
  ret void, !dbg !60
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "denormal-fp-math"="ieee,ieee" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { argmemonly nounwind willreturn }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 (git@github.com:llvm/llvm-project.git 1d5da8cd30fce1c0a2c2fa6ba656dbfaa36192c8)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "various_ir_values.c", directory: "/data/build/llvm-project")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 11.0.0 (git@github.com:llvm/llvm-project.git 1d5da8cd30fce1c0a2c2fa6ba656dbfaa36192c8)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14}
!13 = !DILocalVariable(name: "A", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!14 = !DILocalVariable(name: "i", scope: !15, file: !1, line: 3, type: !11)
!15 = distinct !DILexicalBlock(scope: !7, file: !1, line: 3, column: 3)
!16 = !{!17, !17, i64 0}
!17 = !{!"any pointer", !18, i64 0}
!18 = !{!"omnipotent char", !19, i64 0}
!19 = !{!"Simple C/C++ TBAA"}
!20 = !DILocation(line: 1, column: 15, scope: !7)
!21 = !DILocation(line: 3, column: 8, scope: !15)
!22 = !DILocation(line: 3, column: 12, scope: !15)
!23 = !{!24, !24, i64 0}
!24 = !{!"int", !18, i64 0}
!25 = !DILocation(line: 3, column: 19, scope: !26)
!26 = distinct !DILexicalBlock(scope: !15, file: !1, line: 3, column: 3)
!27 = !DILocation(line: 3, column: 24, scope: !26)
!28 = !DILocation(line: 3, column: 23, scope: !26)
!29 = !DILocation(line: 3, column: 21, scope: !26)
!30 = !DILocation(line: 3, column: 3, scope: !15)
!31 = !DILocation(line: 3, column: 3, scope: !26)
!32 = !DILocation(line: 4, column: 5, scope: !26)
!33 = !DILocation(line: 4, column: 7, scope: !26)
!34 = !DILocation(line: 4, column: 10, scope: !26)
!35 = !DILocation(line: 3, column: 27, scope: !26)
!36 = distinct !{!36, !30, !37}
!37 = !DILocation(line: 4, column: 12, scope: !15)
!38 = !DILocation(line: 5, column: 1, scope: !7)
!39 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 7, type: !8, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !40)
!40 = !{!41, !42}
!41 = !DILocalVariable(name: "A", arg: 1, scope: !39, file: !1, line: 7, type: !10)
!42 = !DILocalVariable(name: "i", scope: !43, file: !1, line: 9, type: !11)
!43 = distinct !DILexicalBlock(scope: !39, file: !1, line: 9, column: 3)
!44 = !DILocation(line: 7, column: 15, scope: !39)
!45 = !DILocation(line: 9, column: 8, scope: !43)
!46 = !DILocation(line: 9, column: 12, scope: !43)
!47 = !DILocation(line: 9, column: 19, scope: !48)
!48 = distinct !DILexicalBlock(scope: !43, file: !1, line: 9, column: 3)
!49 = !DILocation(line: 9, column: 24, scope: !48)
!50 = !DILocation(line: 9, column: 23, scope: !48)
!51 = !DILocation(line: 9, column: 21, scope: !48)
!52 = !DILocation(line: 9, column: 3, scope: !43)
!53 = !DILocation(line: 9, column: 3, scope: !48)
!54 = !DILocation(line: 10, column: 5, scope: !48)
!55 = !DILocation(line: 10, column: 7, scope: !48)
!56 = !DILocation(line: 10, column: 10, scope: !48)
!57 = !DILocation(line: 9, column: 27, scope: !48)
!58 = distinct !{!58, !52, !59}
!59 = !DILocation(line: 10, column: 12, scope: !43)
!60 = !DILocation(line: 11, column: 1, scope: !39)
!61 = !{!"branch_weights", i32 1, i32 1048575}
!62 = distinct !DIAssignID()
