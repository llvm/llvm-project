; RUN: opt %s -passes=indvars -S -o - | FileCheck %s
source_filename = "/Data/llvm/test/Transforms/IndVarSimplify/scev-phi-debug-info.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.status = type { i32, ptr }

@status = internal unnamed_addr global [32 x %struct.status] zeroinitializer, align 16, !dbg !0

define void @f0() local_unnamed_addr !dbg !20 {
entry:
  tail call void @llvm.dbg.value(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !24
  br label %for.cond, !dbg !24

for.cond:                                         ; preds = %for.body, %entry
  ; CHECK: %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  ; CHECK: call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !23, metadata !DIExpression()), !dbg !24
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  tail call void @llvm.dbg.value(metadata i32 %i.0, metadata !23, metadata !DIExpression()), !dbg !24
  %cmp = icmp slt i32 %i.0, 32, !dbg !24
  br i1 %cmp, label %for.body, label %for.end, !dbg !24

for.body:                                         ; preds = %for.cond
  %idxprom = sext i32 %i.0 to i64, !dbg !24
  %value = getelementptr inbounds [32 x %struct.status], ptr @status, i64 0, i64 %idxprom, i32 0, !dbg !24
  store i32 42, ptr %value, align 16, !dbg !24
  tail call void @use(i32 %i.0), !dbg !24
  %inc = add nsw i32 %i.0, 1, !dbg !24
  tail call void @llvm.dbg.value(metadata i32 %inc, metadata !23, metadata !DIExpression()), !dbg !24
  br label %for.cond, !dbg !24

for.end:                                          ; preds = %for.cond
  ret void, !dbg !24
}

declare void @use(i32)

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!16, !17, !18}
!llvm.ident = !{!19}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "status", scope: !2, file: !3, line: 5, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 6.0.0 (trunk 316001) (llvm/trunk 316171)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "x.c", directory: "/home/davide/work/llvm/build-release/bin")
!4 = !{}
!5 = !{!0}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 4096, elements: !14)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "status", file: !3, line: 2, size: 128, elements: !8)
!8 = !{!9, !11}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !7, file: !3, line: 3, baseType: !10, size: 32)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "p", scope: !7, file: !3, line: 4, baseType: !12, size: 64, offset: 64)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!14 = !{!15}
!15 = !DISubrange(count: 32)
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{i32 1, !"wchar_size", i32 4}
!19 = !{!"clang version 6.0.0 (trunk 316001) (llvm/trunk 316171)"}
!20 = distinct !DISubprogram(name: "f0", scope: !3, file: !3, line: 6, type: !21, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !22)
!21 = !DISubroutineType(types: !4)
!22 = !{!23}
!23 = !DILocalVariable(name: "i", scope: !20, file: !3, line: 8, type: !10)
!24 = !DILocation(line: 9, scope: !20)
