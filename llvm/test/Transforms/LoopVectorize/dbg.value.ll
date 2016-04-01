; RUN: opt < %s -S -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine | FileCheck %s
; Make sure we vectorize with debugging turned on.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

@A = global [1024 x i32] zeroinitializer, align 16
@B = global [1024 x i32] zeroinitializer, align 16
@C = global [1024 x i32] zeroinitializer, align 16

; CHECK-LABEL: @test(
define i32 @test() #0 !dbg !3 {
entry:
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !9, metadata !DIExpression()), !dbg !18
  br label %for.body, !dbg !18

for.body:
  ;CHECK: load <4 x i32>
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv, !dbg !19
  %0 = load i32, i32* %arrayidx, align 4, !dbg !19
  %arrayidx2 = getelementptr inbounds [1024 x i32], [1024 x i32]* @C, i64 0, i64 %indvars.iv, !dbg !19
  %1 = load i32, i32* %arrayidx2, align 4, !dbg !19
  %add = add nsw i32 %1, %0, !dbg !19
  %arrayidx4 = getelementptr inbounds [1024 x i32], [1024 x i32]* @A, i64 0, i64 %indvars.iv, !dbg !19
  store i32 %add, i32* %arrayidx4, align 4, !dbg !19
  %indvars.iv.next = add i64 %indvars.iv, 1, !dbg !18
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !9, metadata !DIExpression()), !dbg !18
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !18
  %exitcond = icmp ne i32 %lftr.wideiv, 1024, !dbg !18
  br i1 %exitcond, label %for.body, label %for.end, !dbg !18

for.end:
  ret i32 0, !dbg !24
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "fp-contract-model"="standard" "no-frame-pointer-elim" "no-frame-pointer-elim-non-leaf" "relocation-model"="pic" "ssp-buffers-size"="8" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!26}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang", isOptimized: true, emissionKind: FullDebug, file: !25, enums: !1, retainedTypes: !1, subprograms: !2, globals: !11)
!1 = !{}
!2 = !{!3}
!3 = distinct !DISubprogram(name: "test", linkageName: "test", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 5, file: !25, scope: !4, type: !5, variables: !8)
!4 = !DIFile(filename: "test", directory: "/path/to/somewhere")
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DILocalVariable(name: "i", line: 6, scope: !10, file: !4, type: !7)
!10 = distinct !DILexicalBlock(line: 6, column: 0, file: !25, scope: !3)
!11 = !{!12, !16, !17}
!12 = !DIGlobalVariable(name: "A", line: 1, isLocal: false, isDefinition: true, scope: null, file: !4, type: !13, variable: [1024 x i32]* @A)
!13 = !DICompositeType(tag: DW_TAG_array_type, size: 32768, align: 32, baseType: !7, elements: !14)
!14 = !{!15}
!15 = !{i32 786465, i64 0, i64 1024}
!16 = !DIGlobalVariable(name: "B", line: 2, isLocal: false, isDefinition: true, scope: null, file: !4, type: !13, variable: [1024 x i32]* @B)
!17 = !DIGlobalVariable(name: "C", line: 3, isLocal: false, isDefinition: true, scope: null, file: !4, type: !13, variable: [1024 x i32]* @C)
!18 = !DILocation(line: 6, scope: !10)
!19 = !DILocation(line: 7, scope: !20)
!20 = distinct !DILexicalBlock(line: 6, column: 0, file: !25, scope: !10)
!24 = !DILocation(line: 9, scope: !3)
!25 = !DIFile(filename: "test", directory: "/path/to/somewhere")
!26 = !{i32 1, !"Debug Info Version", i32 3}
