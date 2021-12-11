; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o - %s \
; RUN:     | llvm-dwarfdump - \
; RUN:     | FileCheck %s

; CHECK:      DW_TAG_variable
; CHECK-NOT:    DW_AT_location
; CHECK-NEXT:   DW_AT_name  ("arr")
; CHECK-NOT:    DW_AT_location
; CHECK:      DW_TAG
define dso_local void @test() !dbg !4 {
entry:
  call void @llvm.dbg.value(metadata !DIArgList(i128 0), metadata !7, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_stack_value, DW_OP_LLVM_fragment, 0, 127)), !dbg !12
  ret void, !dbg !12
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !5, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DILocalVariable(name: "arr", scope: !4, file: !1, line: 1, type: !8)
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 128, elements: !10)
!9 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!10 = !{!11}
!11 = !DISubrange(count: 16)
!12 = !DILocation(line: 1, column: 1, scope: !4)
