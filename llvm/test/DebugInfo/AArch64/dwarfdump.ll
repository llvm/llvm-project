; RUN: llc -mtriple=aarch64-non-linux-gnu -dwarf-version=4 < %s -filetype=obj \
; RUN:    | llvm-dwarfdump -v - | FileCheck -check-prefix=CHECK -check-prefix=CHECK-4 %s
; RUN: llc -mtriple=aarch64-non-linux-gnu -dwarf-version=3 < %s -filetype=obj \
; RUN:    | llvm-dwarfdump -v - | FileCheck -check-prefix=CHECK -check-prefix=CHECK-3 %s

; We're mostly checking that relocations are applied correctly
; here. Currently R_AARCH64_ABS32 is used for references to debug data
; and R_AARCH64_ABS64 is used for program addresses.

; A couple of ABS32s, both at 0 and elsewhere, interpreted correctly:

; CHECK: DW_AT_producer [DW_FORM_strp] ( .debug_str[0x00000000] = "clang version 3.3 ")
; CHECK: DW_AT_name [DW_FORM_strp] ( .debug_str[0x00000013] = "tmp.c")

; A couple of ABS64s similarly:

; CHECK: DW_AT_low_pc [DW_FORM_addr] (0x0000000000000000 ".text")
; CHECK-4: DW_AT_high_pc [DW_FORM_data4] (0x00000008)
; CHECK-3: DW_AT_high_pc [DW_FORM_addr] (0x0000000000000008 ".text")

define i32 @main() nounwind !dbg !3 {
  ret i32 0, !dbg !8
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.3 ", isOptimized: false, emissionKind: FullDebug, file: !9, enums: !1, retainedTypes: !1, globals: !1, imports:  !1)
!1 = !{}
!3 = distinct !DISubprogram(name: "main", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !0, scopeLine: 1, file: !9, scope: !4, type: !5, retainedNodes: !1)
!4 = !DIFile(filename: "tmp.c", directory: "/home/tim/llvm/build")
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DILocation(line: 2, scope: !3)
!9 = !DIFile(filename: "tmp.c", directory: "/home/tim/llvm/build")
!10 = !{i32 1, !"Debug Info Version", i32 3}
