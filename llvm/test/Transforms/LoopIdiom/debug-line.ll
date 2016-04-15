; RUN: opt -loop-idiom < %s -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"


define void @foo(double* nocapture %a) nounwind ssp !dbg !0 {
entry:
  tail call void @llvm.dbg.value(metadata double* %a, i64 0, metadata !5, metadata !DIExpression()), !dbg !8
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !10, metadata !DIExpression()), !dbg !14
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.body ]
  %arrayidx = getelementptr double, double* %a, i64 %indvar
; CHECK: call void @llvm.memset{{.+}} !dbg 
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !15
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp ne i64 %indvar.next, 1000
  br i1 %exitcond, label %for.body, label %for.end, !dbg !14

for.end:                                          ; preds = %for.body
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !10, metadata !DIExpression()), !dbg !16
  ret void, !dbg !17
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.module.flags = !{!19}
!llvm.dbg.cu = !{!2}

!0 = distinct !DISubprogram(name: "foo", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !2, file: !18, scope: !1, type: !3)
!1 = !DIFile(filename: "li.c", directory: "/private/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 2.9 (trunk 127165:127174)", isOptimized: true, emissionKind: FullDebug, file: !18, enums: !9, retainedTypes: !9)
!3 = !DISubroutineType(types: !4)
!4 = !{null}
!5 = !DILocalVariable(name: "a", line: 2, arg: 1, scope: !0, file: !1, type: !6)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, scope: !2, baseType: !7)
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!8 = !DILocation(line: 2, column: 18, scope: !0)
!9 = !{}
!10 = !DILocalVariable(name: "i", line: 3, scope: !11, file: !1, type: !13)
!11 = distinct !DILexicalBlock(line: 3, column: 3, file: !18, scope: !12)
!12 = distinct !DILexicalBlock(line: 2, column: 21, file: !18, scope: !0)
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !DILocation(line: 3, column: 3, scope: !12)
!15 = !DILocation(line: 4, column: 5, scope: !11)
!16 = !DILocation(line: 3, column: 29, scope: !11)
!17 = !DILocation(line: 5, column: 1, scope: !12)
!18 = !DIFile(filename: "li.c", directory: "/private/tmp")
!19 = !{i32 1, !"Debug Info Version", i32 3}
