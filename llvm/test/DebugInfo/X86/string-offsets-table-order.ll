; RUN: llc -mtriple=x86_64-unknown-linux-gnu -split-dwarf-file=foo.dwo -filetype=obj < %s \
; RUN:   | llvm-dwarfdump -v - | FileCheck %s

; This triggers a situation where the order of entries in the .debug_str and
; .debug_str_offsets sections does not match and makes sure that all entries are
; still wired up correctly.

; Produced with "clang -S -emit-llvm -gdwarf-5" from source "int X;", copied
; three times and modified by hand. The modifications consisted of modifying the
; compilation directory and the variable names to trigger the insertion of names
; in different order.

; CHECK: .debug_info contents:
; CHECK:   DW_TAG_skeleton_unit
; CHECK:     DW_AT_comp_dir [DW_FORM_strx1] (indexed (00000000) string = "X3")
; CHECK:   DW_TAG_skeleton_unit
; CHECK:     DW_AT_comp_dir [DW_FORM_strx1] (indexed (00000001) string = "X2")
; CHECK:   DW_TAG_skeleton_unit
; CHECK:     DW_AT_comp_dir [DW_FORM_strx1] (indexed (00000002) string = "X1")
; CHECK: .debug_info.dwo contents:

; CHECK: .debug_str contents:
; CHECK: 0x[[X3:[0-9a-f]*]]: "X3"
; CHECK: 0x[[X1:[0-9a-f]*]]: "X1"
; CHECK: 0x[[X2:[0-9a-f]*]]: "X2"

; CHECK: .debug_str_offsets contents:
; CHECK: Format = DWARF32, Version = 5
; CHECK-NEXT: [[X3]] "X3"
; CHECK-NEXT: [[X2]] "X2"
; CHECK-NEXT: [[X1]] "X1"
; CHECK-NEXT: "foo.dwo"
; CHECK-EMPTY:



!llvm.dbg.cu = !{!10, !20, !30}
!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 2, !"Dwarf Version", i32 5}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{!"clang version 7.0.0 (trunk 337353) (llvm/trunk 337361)"}


@X1 = dso_local global i32 0, align 4, !dbg !11

!10 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !13, producer: "clang version 7.0.0 (trunk 337353) (llvm/trunk 337361)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !14, globals: !15)
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = distinct !DIGlobalVariable(name: "X1", scope: !10, file: !16, line: 1, type: !17, isLocal: false, isDefinition: true)
!13 = !DIFile(filename: "-", directory: "X3", checksumkind: CSK_MD5, checksum: "f2e6e10e303927a308f1645fbf6f710e")
!14 = !{}
!15 = !{!11}
!16 = !DIFile(filename: "<stdin>", directory: "X3", checksumkind: CSK_MD5, checksum: "f2e6e10e303927a308f1645fbf6f710e")
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)


@X2 = dso_local global i32 0, align 4, !dbg !21

!20 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !23, producer: "clang version 7.0.0 (trunk 337353) (llvm/trunk 337361)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !24, globals: !25)
!21 = !DIGlobalVariableExpression(var: !22, expr: !DIExpression())
!22 = distinct !DIGlobalVariable(name: "X2", scope: !20, file: !26, line: 1, type: !27, isLocal: false, isDefinition: true)
!23 = !DIFile(filename: "-", directory: "X2", checksumkind: CSK_MD5, checksum: "f2e6e10e303927a308f1645fbf6f710e")
!24 = !{}
!25 = !{!21}
!26 = !DIFile(filename: "<stdin>", directory: "X2", checksumkind: CSK_MD5, checksum: "f2e6e10e303927a308f1645fbf6f710e")
!27 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)


@X3 = dso_local global i32 0, align 4, !dbg !31

!30 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !33, producer: "clang version 7.0.0 (trunk 337353) (llvm/trunk 337361)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !34, globals: !35)
!31 = !DIGlobalVariableExpression(var: !32, expr: !DIExpression())
!32 = distinct !DIGlobalVariable(name: "X3", scope: !30, file: !36, line: 1, type: !37, isLocal: false, isDefinition: true)
!33 = !DIFile(filename: "-", directory: "X1", checksumkind: CSK_MD5, checksum: "f2e6e10e303927a308f1645fbf6f710e")
!34 = !{}
!35 = !{!31}
!36 = !DIFile(filename: "<stdin>", directory: "X1", checksumkind: CSK_MD5, checksum: "f2e6e10e303927a308f1645fbf6f710e")
!37 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
