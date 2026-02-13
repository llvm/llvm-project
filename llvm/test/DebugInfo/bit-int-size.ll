; RUN: %llc_dwarf %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
; REQUIRES: object-emission

;; Check base types with bit-sizes that don't fit fully fit within a byte
;; multiple get both a a byte_size and bit_size attribute.

; CHECK: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name      ("unsigned _BitInt")
; CHECK-NEXT: DW_AT_encoding  (DW_ATE_unsigned)
; CHECK-NEXT: DW_AT_byte_size (0x04)
; CHECK-NEXT: DW_AT_bit_size  (0x11)

; CHECK: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name      ("_BitInt")
; CHECK-NEXT: DW_AT_encoding  (DW_ATE_signed)
; CHECK-NEXT: DW_AT_byte_size (0x01)
; CHECK-NEXT: DW_AT_bit_size  (0x02)

@a = global i8 0, align 1, !dbg !0
@b = global i8 0, align 1, !dbg !5

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !7, line: 4, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 22.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "bit-int.c", directory: "/")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !7, line: 5, type: !8, isLocal: false, isDefinition: true)
!7 = !DIFile(filename: "bit-int.c", directory: "/")
!8 = !DIBasicType(name: "_BitInt", size: 8, dataSize: 2, encoding: DW_ATE_signed)
!9 = !DIBasicType(name: "unsigned _BitInt", size: 32, dataSize: 17, encoding: DW_ATE_unsigned)
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 22.0.0git"}
