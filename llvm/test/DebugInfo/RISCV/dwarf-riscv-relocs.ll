; RUN: llc -filetype=obj -mtriple=riscv32 -mattr=+relax %s -o %t.o
; RUN: llvm-readobj -r %t.o | FileCheck -check-prefix=READOBJ-RELOCS %s
; RUN: llvm-objdump --source %t.o | FileCheck --check-prefix=OBJDUMP-SOURCE %s
; RUN: llvm-dwarfdump --debug-info %t.o | \
; RUN:     FileCheck -check-prefix=DWARF-DUMP %s
; RUN: llvm-dwarfdump --debug-line -v %t.o | \
; RUN:     FileCheck -check-prefix=LINE-DUMP %s

; Check that we actually have relocations, otherwise this is kind of pointless.
; READOBJ-RELOCS:  Section ({{.*}}) .rela.debug_info {
; READOBJ-RELOCS:    0x1B R_RISCV_ADD32 .L0  0x0
; READOBJ-RELOCS-NEXT:    0x1B R_RISCV_SUB32 .L0  0x0
; READOBJ-RELOCS:  Section ({{.*}}) .rela.debug_frame {
; READOBJ-RELOCS:    0x20 R_RISCV_ADD32 .L0  0x0
; READOBJ-RELOCS-NEXT:    0x20 R_RISCV_SUB32 .L0  0x0
; READOBJ-RELOCS:  Section ({{.*}}) .rela.debug_line {
; READOBJ-RELOCS:    0x5A R_RISCV_ADD16 .L0  0x0
; READOBJ-RELOCS-NEXT:    0x5A R_RISCV_SUB16 .L0  0x0

; Check that we can print the source, even with relocations.
; OBJDUMP-SOURCE: Disassembly of section .text:
; OBJDUMP-SOURCE-EMPTY:
; OBJDUMP-SOURCE-NEXT: 00000000 <main>:
; OBJDUMP-SOURCE: ; {
; OBJDUMP-SOURCE: ; return 0;

; Check that we correctly dump the DWARF info, even with relocations.
; DWARF-DUMP: DW_AT_name        ("dwarf-riscv-relocs.c")
; DWARF-DUMP: DW_AT_comp_dir    (".")
; DWARF-DUMP: DW_AT_name      ("main")
; DWARF-DUMP: DW_AT_decl_file ("{{.*}}dwarf-riscv-relocs.c")
; DWARF-DUMP: DW_AT_decl_line (1)
; DWARF-DUMP: DW_AT_type      (0x00000032 "int")
; DWARF-DUMP: DW_AT_name      ("int")
; DWARF-DUMP: DW_AT_encoding  (DW_ATE_signed)
; DWARF-DUMP: DW_AT_byte_size (0x04)

; LINE-DUMP: .debug_line contents:
; LINE-DUMP-NEXT: debug_line[0x00000000]
; LINE-DUMP-NEXT: Line table prologue:
; LINE-DUMP-NEXT:     total_length: 0x00000061
; LINE-DUMP-NEXT:           format: DWARF32
; LINE-DUMP-NEXT:          version: 5
; LINE-DUMP-NEXT:     address_size: 4
; LINE-DUMP-NEXT:  seg_select_size: 0
; LINE-DUMP-NEXT:  prologue_length: 0x0000003e
; LINE-DUMP-NEXT:  min_inst_length: 1
; LINE-DUMP-NEXT: max_ops_per_inst: 1
; LINE-DUMP-NEXT:  default_is_stmt: 1
; LINE-DUMP-NEXT:        line_base: -5
; LINE-DUMP-NEXT:       line_range: 14
; LINE-DUMP-NEXT:      opcode_base: 13
; LINE-DUMP-NEXT: standard_opcode_lengths[DW_LNS_copy] = 0
; LINE-DUMP-NEXT: standard_opcode_lengths[DW_LNS_advance_pc] = 1
; LINE-DUMP-NEXT: standard_opcode_lengths[DW_LNS_advance_line] = 1
; LINE-DUMP-NEXT: standard_opcode_lengths[DW_LNS_set_file] = 1
; LINE-DUMP-NEXT: standard_opcode_lengths[DW_LNS_set_column] = 1
; LINE-DUMP-NEXT: standard_opcode_lengths[DW_LNS_negate_stmt] = 0
; LINE-DUMP-NEXT: standard_opcode_lengths[DW_LNS_set_basic_block] = 0
; LINE-DUMP-NEXT: standard_opcode_lengths[DW_LNS_const_add_pc] = 0
; LINE-DUMP-NEXT: standard_opcode_lengths[DW_LNS_fixed_advance_pc] = 1
; LINE-DUMP-NEXT: standard_opcode_lengths[DW_LNS_set_prologue_end] = 0
; LINE-DUMP-NEXT: standard_opcode_lengths[DW_LNS_set_epilogue_begin] = 0
; LINE-DUMP-NEXT: standard_opcode_lengths[DW_LNS_set_isa] = 1
; LINE-DUMP-NEXT: include_directories[  0] = .debug_line_str[0x00000000] = "."
; LINE-DUMP-NEXT: file_names[  0]:
; LINE-DUMP-NEXT:            name: .debug_line_str[0x00000002] = "dwarf-riscv-relocs.c"
; LINE-DUMP-NEXT:       dir_index: 0
; LINE-DUMP-NEXT:    md5_checksum: 05ab89f5481bc9f2d037e7886641e919
; LINE-DUMP-NEXT:          source: .debug_line_str[0x00000017] = "int main()\n{\n    return 0;\n}\n"
; LINE-DUMP-EMPTY:
; LINE-DUMP-NEXT:            Address            Line   Column File   ISA Discriminator OpIndex Flags
; LINE-DUMP-NEXT:            ------------------ ------ ------ ------ --- ------------- ------- -------------
; LINE-DUMP-NEXT:0x0000004a: 04 DW_LNS_set_file (0)
; LINE-DUMP-NEXT:0x0000004c: 00 DW_LNE_set_address (0x00000000)
; LINE-DUMP-NEXT:0x00000053: 13 address += 0,  line += 1,  op-index += 0
; LINE-DUMP-NEXT:            0x0000000000000000      2      0      0   0             0       0  is_stmt
; LINE-DUMP-NEXT:0x00000054: 05 DW_LNS_set_column (5)
; LINE-DUMP-NEXT:0x00000056: 0a DW_LNS_set_prologue_end
; LINE-DUMP-NEXT:0x00000057: 03 DW_LNS_advance_line (3)
; LINE-DUMP-NEXT:0x00000059: 09 DW_LNS_fixed_advance_pc (addr += 0x001c, op-index = 0)
; LINE-DUMP-NEXT:0x0000005c: 01 DW_LNS_copy
; LINE-DUMP-NEXT:            0x000000000000001c      3      5      0   0             0       0  is_stmt prologue_end
; LINE-DUMP-NEXT:0x0000005d: 06 DW_LNS_negate_stmt
; LINE-DUMP-NEXT:0x0000005e: 0b DW_LNS_set_epilogue_begin
; LINE-DUMP-NEXT:0x0000005f: 4a address += 4,  line += 0,  op-index += 0
; LINE-DUMP-NEXT:            0x0000000000000020      3      5      0   0             0       0  epilogue_begin
; LINE-DUMP-NEXT:0x00000060: 02 DW_LNS_advance_pc (addr += 16, op-index += 0)
; LINE-DUMP-NEXT:0x00000062: 00 DW_LNE_end_sequence
; LINE-DUMP-NEXT:            0x0000000000000030      3      5      0   0             0       0  end_sequence

; ModuleID = 'dwarf-riscv-relocs.c'
source_filename = "dwarf-riscv-relocs.c"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32"

; Function Attrs: noinline nounwind optnone
define dso_local i32 @main() #0 !dbg !7 {
entry:
  call void @ext()
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  ret i32 0, !dbg !11
}

declare void @ext()

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }

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
