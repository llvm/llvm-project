; RUN: llc < %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; RUN: llc < %s -o %t -filetype=obj -regalloc=basic
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin8"

;CHECK: DW_AT_location{{.*}}(<0x1> 55 )

%0 = type { i64, i1 }

@__clz_tab = external constant [256 x i8]

define hidden i128 @__divti3(i128 %u, i128 %v) nounwind readnone !dbg !9 {
entry:
  tail call void @llvm.dbg.value(metadata i128 %u, i64 0, metadata !14, metadata !DIExpression()), !dbg !15
  tail call void @llvm.dbg.value(metadata i64 0, i64 0, metadata !17, metadata !DIExpression()), !dbg !21
  br i1 undef, label %bb2, label %bb4, !dbg !22

bb2:                                              ; preds = %entry
  br label %bb4, !dbg !23

bb4:                                              ; preds = %bb2, %entry
  br i1 undef, label %__udivmodti4.exit, label %bb82.i, !dbg !24

bb82.i:                                           ; preds = %bb4
  unreachable

__udivmodti4.exit:                                ; preds = %bb4
  ret i128 undef, !dbg !27
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

declare %0 @llvm.uadd.with.overflow.i64(i64, i64) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!32}

!0 = distinct !DISubprogram(name: "__udivmodti4", line: 879, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 879, file: !29, scope: !1, type: !3)
!1 = !DIFile(filename: "foobar.c", directory: "/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, emissionKind: 0, file: !29, enums: !31, retainedTypes: !31, subprograms: !28, imports:  null)
!3 = !DISubroutineType(types: !4)
!4 = !{!5, !5, !5, !8}
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "UTItype", line: 166, file: !30, scope: !6, baseType: !7)
!6 = !DIFile(filename: "foobar.h", directory: "/tmp")
!7 = !DIBasicType(tag: DW_TAG_base_type, size: 128, align: 128, encoding: DW_ATE_unsigned)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !29, scope: !1, baseType: !5)
!9 = distinct !DISubprogram(name: "__divti3", linkageName: "__divti3", line: 1094, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 1094, file: !29, scope: !1, type: !10)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12, !12}
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "TItype", line: 160, file: !30, scope: !6, baseType: !13)
!13 = !DIBasicType(tag: DW_TAG_base_type, size: 128, align: 128, encoding: DW_ATE_signed)
!14 = !DILocalVariable(name: "u", line: 1093, arg: 1, scope: !9, file: !1, type: !12)
!15 = !DILocation(line: 1093, scope: !9)
!16 = !{i64 0}
!17 = !DILocalVariable(name: "c", line: 1095, scope: !18, file: !1, type: !19)
!18 = distinct !DILexicalBlock(line: 1094, column: 0, file: !29, scope: !9)
!19 = !DIDerivedType(tag: DW_TAG_typedef, name: "word_type", line: 424, file: !30, scope: !6, baseType: !20)
!20 = !DIBasicType(tag: DW_TAG_base_type, name: "long int", size: 64, align: 64, encoding: DW_ATE_signed)
!21 = !DILocation(line: 1095, scope: !18)
!22 = !DILocation(line: 1103, scope: !18)
!23 = !DILocation(line: 1104, scope: !18)
!24 = !DILocation(line: 1003, scope: !25, inlinedAt: !26)
!25 = distinct !DILexicalBlock(line: 879, column: 0, file: !29, scope: !0)
!26 = !DILocation(line: 1107, scope: !18)
!27 = !DILocation(line: 1111, scope: !18)
!28 = !{!0, !9}
!29 = !DIFile(filename: "foobar.c", directory: "/tmp")
!30 = !DIFile(filename: "foobar.h", directory: "/tmp")
!31 = !{}
!32 = !{i32 1, !"Debug Info Version", i32 3}
