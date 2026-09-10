; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -filetype=obj < %s \
; RUN:   | llvm-dwarfdump - | FileCheck %s

@g = global i64 0, align 8

; CHECK-LABEL: DW_AT_name ("globaladdr_tag_offset")
; CHECK:       DW_TAG_variable
; CHECK:       DW_AT_location (DW_OP_addrx {{0x[0-9a-f]+}}, DW_OP_stack_value)
; CHECK:       DW_AT_LLVM_tag_offset (0x07)
; CHECK:       DW_AT_name ("x")
define void @globaladdr_tag_offset() !dbg !6 {
entry:
  call void @llvm.dbg.value(metadata ptr @g, metadata !10, metadata !DIExpression(DW_OP_LLVM_tag_offset, 7)), !dbg !11
  tail call void @sink(ptr null), !dbg !11
  ret void, !dbg !11
}

declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @sink(ptr)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: true, emissionKind: FullDebug)
!3 = !DIFile(filename: "globaladdr-tag-offset.c", directory: "/")
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = distinct !DISubprogram(name: "globaladdr_tag_offset", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !12)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!10 = !DILocalVariable(name: "x", scope: !9, file: !3, line: 2, type: !7)
!9 = distinct !DILexicalBlock(scope: !6, file: !3, line: 2, column: 1)
!11 = !DILocation(line: 2, column: 1, scope: !9)
!12 = !{!10}
