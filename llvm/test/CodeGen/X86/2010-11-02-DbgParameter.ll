; RUN: llc -O2 -asm-verbose < %s | FileCheck %s
; Radar 8616981

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin11.0.0"

%struct.bar = type { i32, i32 }

define i32 @foo(ptr nocapture %i) nounwind readnone optsize noinline ssp !dbg !0 {
; CHECK: TAG_formal_parameter
entry:
  tail call void @llvm.dbg.value(metadata ptr %i, i64 0, metadata !6, metadata !DIExpression()), !dbg !12
  ret i32 1, !dbg !13
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!19}

!0 = distinct !DISubprogram(name: "foo", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, scopeLine: 3, file: !17, scope: !1, type: !3, retainedNodes: !16)
!1 = !DIFile(filename: "one.c", directory: "/private/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 2.9 (trunk 117922)", isOptimized: true, emissionKind: FullDebug, file: !17, enums: !18, retainedTypes: !18, imports:  null)
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !DILocalVariable(name: "i", line: 3, arg: 1, scope: !0, file: !1, type: !7)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, file: !17, scope: !1, baseType: !8)
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "bar", line: 2, size: 64, align: 32, file: !17, scope: !1, elements: !9)
!9 = !{!10, !11}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "x", line: 2, size: 32, align: 32, file: !17, scope:  !1, baseType: !5)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "y", line: 2, size: 32, align: 32, offset: 32, file: !17, scope: !1, baseType: !5)
!12 = !DILocation(line: 3, column: 47, scope: !0)
!13 = !DILocation(line: 4, column: 2, scope: !14)
!14 = distinct !DILexicalBlock(line: 3, column: 50, file: !17, scope: !0)
!16 = !{!6}
!17 = !DIFile(filename: "one.c", directory: "/private/tmp")
!18 = !{}
!19 = !{i32 1, !"Debug Info Version", i32 3}
