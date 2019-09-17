; RUN: opt -disable-basicaa -loop-rotate -enable-mssa-loop-dependency -verify-memoryssa -S %s | FileCheck %s
; REQUIRES: asserts

; CHECK-LABEL: @f_w4_i2
define void @f_w4_i2() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i16 [ 0, %entry ], [ %inc, %for.body ]
  call void @llvm.dbg.value(metadata i16 %i.0, metadata !32, metadata !DIExpression()), !dbg !31
  br i1 undef, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  store i32 undef, i32* undef, align 1
  %inc = add i16 %i.0, 1
  call void @llvm.dbg.value(metadata i16 %inc, metadata !32, metadata !DIExpression()), !dbg !31
  br label %for.cond
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0s", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "vec.c", directory: "test")
!2 = !{}
!3 = !{!4}
!4 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = distinct !DISubprogram(name: "f_w4_i2", scope: !1, file: !1, line: 36, type: !16, scopeLine: 38, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !23)
!16 = !DISubroutineType(types: !17)
!17 = !{null}
!23 = !{}
!30 = distinct !DILexicalBlock(scope: !15, file: !1, line: 43, column: 5)
!31 = !DILocation(line: 0, scope: !30)
!32 = !DILocalVariable(name: "i", scope: !30, file: !1, line: 43, type: !4)
