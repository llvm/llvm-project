; FIXME: Missing DwarfAddrSection on AIX
; XFAIL: target={{.*}}-aix{{.*}}
; RUN: %llc_dwarf -filetype=obj %s -o - | llvm-dwarfdump -v -debug-info - | FileCheck %s

;; In DWARF v5, emit DW_AT_addr_base as DW_AT_addr_base is used for DW_AT_low_pc.
; CHECK: DW_AT_low_pc [DW_FORM_addrx]
; CHECK: DW_AT_addr_base

define i64 @foo() !dbg !7 {
entry:
  ret i64 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, emissionKind: LineTablesOnly, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "a.cc", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "a", scope: !1, file: !1, line: 22, type: !8, scopeLine: 22, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
