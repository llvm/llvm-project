; RUN: llc -filetype=obj -mtriple=riscv32 -mattr=+relax %s -o %t.o
; RUN: llvm-readobj -r %t.o | FileCheck -check-prefix=RELOC %s
; RUN: llvm-objdump --source %t.o | FileCheck --check-prefix=OBJDUMP-SOURCE %s
; RUN: llvm-dwarfdump --debug-info -debug-line -v %t.o | \
; RUN:     FileCheck -check-prefix=DWARF %s

; RELOC:       .rela.debug_info {
; RELOC-NEXT:    0x8 R_RISCV_32 .debug_abbrev 0x0
; RELOC-NEXT:    0x11 R_RISCV_32 .L0  0x0
; RELOC-NEXT:    0x15 R_RISCV_32 .Lline_table_start0 0x0
; RELOC-NEXT:    0x1B R_RISCV_ADD32 .L0  0x0
; RELOC-NEXT:    0x1B R_RISCV_SUB32 .L0  0x0
; RELOC-NEXT:    0x1F R_RISCV_32 .L0  0x0
; RELOC-NEXT:    0x25 R_RISCV_ADD32 .L0  0x0
; RELOC-NEXT:    0x25 R_RISCV_SUB32 .L0  0x0
; RELOC-NEXT:  }
; RELOC-NEXT:  .rela.debug_str_offsets {
; RELOC-NEXT:    0x8 R_RISCV_32 .L0  0x0
; RELOC-NEXT:    0xC R_RISCV_32 .L0  0x0
; RELOC-NEXT:    0x10 R_RISCV_32 .L0  0x0
; RELOC-NEXT:    0x14 R_RISCV_32 .L0  0x0
; RELOC-NEXT:    0x18 R_RISCV_32 .L0  0x0
; RELOC-NEXT:  }
; RELOC-NEXT:  .rela.debug_addr {
; RELOC-NEXT:    0x8 R_RISCV_32 .L0  0x0
; RELOC-NEXT:  }
; RELOC-NEXT:  .rela.debug_frame {
; RELOC-NEXT:    0x18 R_RISCV_32 .L0  0x0
; RELOC-NEXT:    0x1C R_RISCV_32 .L0  0x0
; RELOC-NEXT:    0x20 R_RISCV_ADD32 .L0  0x0
; RELOC-NEXT:    0x20 R_RISCV_SUB32 .L0  0x0
; RELOC-NEXT:    0x33 R_RISCV_SET6 .L0  0x0
; RELOC-NEXT:    0x33 R_RISCV_SUB6 .L0  0x0
; RELOC-NEXT:  }
; RELOC-NEXT:  .rela.debug_line {
; RELOC-NEXT:    0x22 R_RISCV_32 .debug_line_str 0x0
; RELOC-NEXT:    0x31 R_RISCV_32 .debug_line_str 0x2
; RELOC-NEXT:    0x46 R_RISCV_32 .debug_line_str 0x17
; RELOC-NEXT:    0x4F R_RISCV_32 .L0  0x0
; RELOC-NEXT:    0x5B R_RISCV_ADD16 .L0  0x0
; RELOC-NEXT:    0x5B R_RISCV_SUB16 .L0  0x0
; RELOC-NEXT:  }

; Check that we can print the source, even with relocations.
; OBJDUMP-SOURCE: Disassembly of section .text:
; OBJDUMP-SOURCE-EMPTY:
; OBJDUMP-SOURCE-NEXT: 00000000 <main>:
; OBJDUMP-SOURCE: ; {
; OBJDUMP-SOURCE: ; return 0;

; DWARF: .debug_line contents:
; DWARF-NEXT: debug_line[0x00000000]
; DWARF-NEXT: Line table prologue:
; DWARF-NEXT:     total_length: 0x00000062
; DWARF-NEXT:           format: DWARF32
; DWARF-NEXT:          version: 5
; DWARF-NEXT:     address_size: 4
; DWARF-NEXT:  seg_select_size: 0
; DWARF-NEXT:  prologue_length: 0x0000003e
; DWARF-NEXT:  min_inst_length: 1
; DWARF-NEXT: max_ops_per_inst: 1
; DWARF-NEXT:  default_is_stmt: 1
; DWARF-NEXT:        line_base: -5
; DWARF-NEXT:       line_range: 14
; DWARF-NEXT:      opcode_base: 13
; DWARF-NEXT: standard_opcode_lengths[DW_LNS_copy] = 0
; DWARF-NEXT: standard_opcode_lengths[DW_LNS_advance_pc] = 1
; DWARF-NEXT: standard_opcode_lengths[DW_LNS_advance_line] = 1
; DWARF-NEXT: standard_opcode_lengths[DW_LNS_set_file] = 1
; DWARF-NEXT: standard_opcode_lengths[DW_LNS_set_column] = 1
; DWARF-NEXT: standard_opcode_lengths[DW_LNS_negate_stmt] = 0
; DWARF-NEXT: standard_opcode_lengths[DW_LNS_set_basic_block] = 0
; DWARF-NEXT: standard_opcode_lengths[DW_LNS_const_add_pc] = 0
; DWARF-NEXT: standard_opcode_lengths[DW_LNS_fixed_advance_pc] = 1
; DWARF-NEXT: standard_opcode_lengths[DW_LNS_set_prologue_end] = 0
; DWARF-NEXT: standard_opcode_lengths[DW_LNS_set_epilogue_begin] = 0
; DWARF-NEXT: standard_opcode_lengths[DW_LNS_set_isa] = 1
; DWARF-NEXT: include_directories[  0] = .debug_line_str[0x00000000] = "."
; DWARF-NEXT: file_names[  0]:
; DWARF-NEXT:            name: .debug_line_str[0x00000002] = "dwarf-riscv-relocs.c"
; DWARF-NEXT:       dir_index: 0
; DWARF-NEXT:    md5_checksum: 05ab89f5481bc9f2d037e7886641e919
; DWARF-NEXT:          source: .debug_line_str[0x00000017] = "int main()\n{\n    return 0;\n}\n"
; DWARF-EMPTY:
; DWARF-NEXT:            Address            Line   Column File   ISA Discriminator OpIndex Flags
; DWARF-NEXT:            ------------------ ------ ------ ------ --- ------------- ------- -------------
; DWARF-NEXT:0x0000004a: 04 DW_LNS_set_file (0)
; DWARF-NEXT:0x0000004c: 00 DW_LNE_set_address (0x00000000)
; DWARF-NEXT:0x00000053: 13 address += 0,  line += 1,  op-index += 0
; DWARF-NEXT:            0x0000000000000000      2      0      0   0             0       0  is_stmt
; DWARF-NEXT:0x00000054: 05 DW_LNS_set_column (5)
; DWARF-NEXT:0x00000056: 0a DW_LNS_set_prologue_end
; DWARF-NEXT:0x00000057: f3 address += 16,  line += 1,  op-index += 0
; DWARF-NEXT:            0x0000000000000010      3      5      0   0             0       0  is_stmt prologue_end
; DWARF-NEXT:0x00000058: 03 DW_LNS_advance_line (4)
; DWARF-NEXT:0x0000005a: 09 DW_LNS_fixed_advance_pc (addr += 0x0010, op-index = 0)
; DWARF-NEXT:0x0000005d: 01 DW_LNS_copy
; DWARF-NEXT:            0x0000000000000020      4      5      0   0             0       0  is_stmt
; DWARF-NEXT:0x0000005e: 06 DW_LNS_negate_stmt
; DWARF-NEXT:0x0000005f: 0b DW_LNS_set_epilogue_begin
; DWARF-NEXT:0x00000060: 4a address += 4,  line += 0,  op-index += 0
; DWARF-NEXT:            0x0000000000000024      4      5      0   0             0       0  epilogue_begin
; DWARF-NEXT:0x00000061: 02 DW_LNS_advance_pc (addr += 16, op-index += 0)
; DWARF-NEXT:0x00000063: 00 DW_LNE_end_sequence
; DWARF-NEXT:            0x0000000000000034      4      5      0   0             0       0  end_sequence

; ModuleID = 'dwarf-riscv-relocs.c'
source_filename = "dwarf-riscv-relocs.c"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32"

; Function Attrs: noinline nounwind optnone
define dso_local i32 @main() #0 !dbg !7 {
entry:
  call void asm sideeffect ".cfi_remember_state\0A\09.cfi_adjust_cfa_offset 16\0A\09nop\0A\09call ext\0A\09nop\0A\09.cfi_restore_state\0A\09", ""() #1, !dbg !11
  ret i32 0, !dbg !12
}

declare void @ext()

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+relax" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "dwarf-riscv-relocs.c", directory: ".", checksumkind: CSK_MD5, checksum: "05ab89f5481bc9f2d037e7886641e919", source: "int main()\0A{\0A    return 0;\0A}\0A")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !8, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 3, column: 5, scope: !7)
!12 = !DILocation(line: 4, column: 5, scope: !7)
