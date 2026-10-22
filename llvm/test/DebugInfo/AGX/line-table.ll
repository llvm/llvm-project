; RUN: llc -mtriple=agx -mcpu=agx-g13g -filetype=obj %s -o %t.o
; RUN: llvm-readelf -r %t.o | FileCheck %s --check-prefix=RELOCS
; RUN: llvm-dwarfdump --debug-line %t.o | FileCheck %s --check-prefix=LINES
; RUN: llvm-objdump -d -l %t.o | FileCheck %s --check-prefix=OBJDUMP
; RUN: llvm-addr2line -f -e %t.o 0x8 | FileCheck %s --check-prefix=ADDR

; The line table of an AGX object: its relocations (R_AGX_ABS32 against
; .debug_line_str, R_AGX_ABS64 against .text) are resolved by the tools.

; RELOCS: Relocation section '.rela.debug_line'
; RELOCS: R_AGX_ABS32 {{.*}} .debug_line_str + 0
; RELOCS: R_AGX_ABS64 {{.*}} .text + 0

; LINES: name: "k.cu"
; LINES: 0x0000000000000000 2 5 0 0 0 0 is_stmt prologue_end
; LINES-NEXT: 0x0000000000000008 3 7 0 0 0 0 is_stmt
; LINES-NEXT: 0x0000000000000010 4 1 0 0 0 0 is_stmt
; LINES-NEXT: end_sequence

; OBJDUMP: <k>:
; OBJDUMP: ; /src/k.cu:2
; OBJDUMP-NEXT: iadd r1, u2, 3
; OBJDUMP-NEXT: ; /src/k.cu:3
; OBJDUMP-NEXT: device_store
; OBJDUMP-NEXT: ; /src/k.cu:4
; OBJDUMP-NEXT: stop

; ADDR: k
; ADDR-NEXT: /src/k.cu:3

define agx_kernel void @k(ptr addrspace(1) %p, i32 %v) !dbg !5 {
  %a = add i32 %v, 3, !dbg !8
  store i32 %a, ptr addrspace(1) %p, !dbg !9
  ret void, !dbg !10
}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "t", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "k.cu", directory: "/src")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 2, !"Dwarf Version", i32 5}
!5 = distinct !DISubprogram(name: "k", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 2, column: 5, scope: !5)
!9 = !DILocation(line: 3, column: 7, scope: !5)
!10 = !DILocation(line: 4, column: 1, scope: !5)
