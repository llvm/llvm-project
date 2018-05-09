; RUN: opt < %s -S -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine | FileCheck %s
; Make sure we vectorize with debugging turned on.

source_filename = "test/Transforms/LoopVectorize/dbg.value.ll"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@A = global [1024 x i32] zeroinitializer, align 16, !dbg !0
@B = global [1024 x i32] zeroinitializer, align 16, !dbg !7
@C = global [1024 x i32] zeroinitializer, align 16, !dbg !9
; CHECK-LABEL: @test(

; Function Attrs: nounwind ssp uwtable
define i32 @test() #0 !dbg !15 {
entry:
  tail call void @llvm.dbg.value(metadata i32 0, metadata !19, metadata !21), !dbg !22
  br label %for.body, !dbg !22

for.body:                                         ; preds = %for.body, %entry
  ;CHECK: load <4 x i32>
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv, !dbg !23
  %0 = load i32, i32* %arrayidx, align 4, !dbg !23
  %arrayidx2 = getelementptr inbounds [1024 x i32], [1024 x i32]* @C, i64 0, i64 %indvars.iv, !dbg !23
  %1 = load i32, i32* %arrayidx2, align 4, !dbg !23
  %add = add nsw i32 %1, %0, !dbg !23
  %arrayidx4 = getelementptr inbounds [1024 x i32], [1024 x i32]* @A, i64 0, i64 %indvars.iv, !dbg !23
  store i32 %add, i32* %arrayidx4, align 4, !dbg !23
  %indvars.iv.next = add i64 %indvars.iv, 1, !dbg !22
  tail call void @llvm.dbg.value(metadata !12, metadata !19, metadata !21), !dbg !22
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !22
  %exitcond = icmp ne i32 %lftr.wideiv, 1024, !dbg !22
  br i1 %exitcond, label %for.body, label %for.end, !dbg !22

for.end:                                          ; preds = %for.body
  ret i32 0, !dbg !25
}

; Function Attrs: nounwind readnone

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "fp-contract-model"="standard" "no-frame-pointer-elim" "no-frame-pointer-elim-non-leaf" "relocation-model"="pic" "ssp-buffers-size"="8" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!11}
!llvm.module.flags = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "A", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "test", directory: "/path/to/somewhere")
!3 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 32768, align: 32, elements: !5)
!4 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!5 = !{!6}
!6 = !{i32 786465, i64 0, i64 1024}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = !DIGlobalVariable(name: "B", scope: null, file: !2, line: 2, type: !3, isLocal: false, isDefinition: true)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = !DIGlobalVariable(name: "C", scope: null, file: !2, line: 3, type: !3, isLocal: false, isDefinition: true)
!11 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !12, retainedTypes: !12, globals: !13)
!12 = !{}
!13 = !{!0, !7, !9}
!14 = !{i32 1, !"Debug Info Version", i32 3}
!15 = distinct !DISubprogram(name: "test", linkageName: "test", scope: !2, file: !2, line: 5, type: !16, isLocal: false, isDefinition: true, scopeLine: 5, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !11, retainedNodes: !18)
!16 = !DISubroutineType(types: !17)
!17 = !{!4}
!18 = !{!19}
!19 = !DILocalVariable(name: "i", scope: !20, file: !2, line: 6, type: !4)
!20 = distinct !DILexicalBlock(scope: !15, file: !2, line: 6)
!21 = !DIExpression()
!22 = !DILocation(line: 6, scope: !20)
!23 = !DILocation(line: 7, scope: !24)
!24 = distinct !DILexicalBlock(scope: !20, file: !2, line: 6)
!25 = !DILocation(line: 9, scope: !15)

