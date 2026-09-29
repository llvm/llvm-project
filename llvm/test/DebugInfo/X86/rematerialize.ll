; RUN: llc -O2 -filetype=obj -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN: | llvm-dwarfdump -debug-line - | FileCheck %s
;
; Generated with clang -O2 -g from
;
; typedef float __m128 __attribute__((__vector_size__(16)));
; 
; extern __m128 doSomething(__m128, __m128);
; 
; 
; __m128 foo(__m128 X) {     // line 6
;   const __m128 V = {0.5f, 0.5f, 0.5f, 0.5f};  // line 7
;   __m128 Sub = X - V;  // line 8
;   __m128 Add = X + V;  // line 9
; 
;   __m128 Result = doSomething(Add, Sub);  // line 11
; 
;   return V - Result;  // line 13
; }
;
;
; We want to see line 13 after line 11 without any other line in between.
; CHECK:       0x{{[0-9a-f]*}} 11
; CHECK-NOT:   0x{{[0-9a-f]*}}  8
; CHECK-NOT:   0x{{[0-9a-f]*}}  9
; CHECK:       0x{{[0-9a-f]*}} 13
; CHECK-NOT:   0x{{[0-9a-f]*}}  8
; CHECK-NOT:   0x{{[0-9a-f]*}}  9

; ModuleID = 'test.c'
source_filename = "test.c"
; target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
; target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define <4 x float> @foo(<4 x float> %X) local_unnamed_addr #0 !dbg !6 {
entry:
  tail call void @llvm.dbg.value(metadata <4 x float> %X, metadata !15, metadata !21), !dbg !22
  tail call void @llvm.dbg.value(metadata <4 x float> <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>, metadata !16, metadata !21), !dbg !23
  %sub = fadd <4 x float> %X, <float -5.000000e-01, float -5.000000e-01, float -5.000000e-01, float -5.000000e-01>, !dbg !24
  tail call void @llvm.dbg.value(metadata <4 x float> %sub, metadata !18, metadata !21), !dbg !25
  %add = fadd <4 x float> %X, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>, !dbg !26
  tail call void @llvm.dbg.value(metadata <4 x float> %add, metadata !19, metadata !21), !dbg !27
  %call = tail call <4 x float> @doSomething(<4 x float> %add, <4 x float> %sub) #3, !dbg !28
  tail call void @llvm.dbg.value(metadata <4 x float> %call, metadata !20, metadata !21), !dbg !29
  %sub1 = fsub <4 x float> <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>, %call, !dbg !30
  ret <4 x float> %sub1, !dbg !31
}

declare <4 x float> @doSomething(<4 x float>, <4 x float>) local_unnamed_addr #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind uwtable }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (trunk 278291)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/home/test")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 4.0.0 (trunk 278291)"}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 6, type: !7, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !14)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9}
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "__m128", file: !1, line: 1, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 128, align: 128, flags: DIFlagVector, elements: !12)
!11 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!12 = !{!13}
!13 = !DISubrange(count: 4)
!14 = !{!15, !16, !18, !19, !20}
!15 = !DILocalVariable(name: "X", arg: 1, scope: !6, file: !1, line: 6, type: !9)
!16 = !DILocalVariable(name: "V", scope: !6, file: !1, line: 7, type: !17)
!17 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !9)
!18 = !DILocalVariable(name: "Sub", scope: !6, file: !1, line: 8, type: !9)
!19 = !DILocalVariable(name: "Add", scope: !6, file: !1, line: 9, type: !9)
!20 = !DILocalVariable(name: "Result", scope: !6, file: !1, line: 11, type: !9)
!21 = !DIExpression()
!22 = !DILocation(line: 6, column: 19, scope: !6)
!23 = !DILocation(line: 7, column: 16, scope: !6)
!24 = !DILocation(line: 8, column: 18, scope: !6)
!25 = !DILocation(line: 8, column: 10, scope: !6)
!26 = !DILocation(line: 9, column: 18, scope: !6)
!27 = !DILocation(line: 9, column: 10, scope: !6)
!28 = !DILocation(line: 11, column: 19, scope: !6)
!29 = !DILocation(line: 11, column: 10, scope: !6)
!30 = !DILocation(line: 13, column: 12, scope: !6)
!31 = !DILocation(line: 13, column: 3, scope: !6)
