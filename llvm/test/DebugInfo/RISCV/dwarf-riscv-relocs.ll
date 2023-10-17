; RUN: llc -filetype=obj -mtriple=riscv32 -mattr=+relax %s -o %t.o
; RUN: llvm-readobj -r %t.o | FileCheck -check-prefix=READOBJ-RELOCS %s
; RUN: llvm-objdump --source %t.o | FileCheck --check-prefix=OBJDUMP-SOURCE %s
; RUN: llvm-dwarfdump --debug-info --debug-line %t.o | \
; RUN:     FileCheck -check-prefix=DWARF-DUMP %s

; Check that we actually have relocations, otherwise this is kind of pointless.
; READOBJ-RELOCS:  Section ({{.*}}) .rela.debug_info {
; READOBJ-RELOCS:    0x1B R_RISCV_ADD32 <null> 0x0
; READOBJ-RELOCS-NEXT:    0x1B R_RISCV_SUB32 <null> 0x0
; READOBJ-RELOCS:  Section ({{.*}}) .rela.debug_frame {
; READOBJ-RELOCS:    0x20 R_RISCV_ADD32 <null> 0x0
; READOBJ-RELOCS-NEXT:    0x20 R_RISCV_SUB32 <null> 0x0
; READOBJ-RELOCS:  Section ({{.*}}) .rela.debug_line {
; READOBJ-RELOCS:    0x5A R_RISCV_ADD16 <null> 0x0
; READOBJ-RELOCS-NEXT:    0x5A R_RISCV_SUB16 <null> 0x0

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

; DWARF-DUMP: .debug_line contents:
; DWARF-DUMP-NEXT: debug_line[0x00000000]
; DWARF-DUMP-NEXT: Line table prologue:
; DWARF-DUMP-NEXT:     total_length: 0x00000067
; DWARF-DUMP-NEXT:           format: DWARF32
; DWARF-DUMP-NEXT:          version: 5
; DWARF-DUMP-NEXT:     address_size: 4
; DWARF-DUMP-NEXT:  seg_select_size: 0
; DWARF-DUMP-NEXT:  prologue_length: 0x0000003e
; DWARF-DUMP-NEXT:  min_inst_length: 1
; DWARF-DUMP-NEXT: max_ops_per_inst: 1
; DWARF-DUMP-NEXT:  default_is_stmt: 1
; DWARF-DUMP-NEXT:        line_base: -5
; DWARF-DUMP-NEXT:       line_range: 14
; DWARF-DUMP-NEXT:      opcode_base: 13
; DWARF-DUMP-NEXT: standard_opcode_lengths[DW_LNS_copy] = 0
; DWARF-DUMP-NEXT: standard_opcode_lengths[DW_LNS_advance_pc] = 1
; DWARF-DUMP-NEXT: standard_opcode_lengths[DW_LNS_advance_line] = 1
; DWARF-DUMP-NEXT: standard_opcode_lengths[DW_LNS_set_file] = 1
; DWARF-DUMP-NEXT: standard_opcode_lengths[DW_LNS_set_column] = 1
; DWARF-DUMP-NEXT: standard_opcode_lengths[DW_LNS_negate_stmt] = 0
; DWARF-DUMP-NEXT: standard_opcode_lengths[DW_LNS_set_basic_block] = 0
; DWARF-DUMP-NEXT: standard_opcode_lengths[DW_LNS_const_add_pc] = 0
; DWARF-DUMP-NEXT: standard_opcode_lengths[DW_LNS_fixed_advance_pc] = 1
; DWARF-DUMP-NEXT: standard_opcode_lengths[DW_LNS_set_prologue_end] = 0
; DWARF-DUMP-NEXT: standard_opcode_lengths[DW_LNS_set_epilogue_begin] = 0
; DWARF-DUMP-NEXT: standard_opcode_lengths[DW_LNS_set_isa] = 1
; DWARF-DUMP-NEXT: include_directories[  0] = "."
; DWARF-DUMP-NEXT: file_names[  0]:
; DWARF-DUMP-NEXT:            name: "dwarf-riscv-relocs.c"
; DWARF-DUMP-NEXT:       dir_index: 0
; DWARF-DUMP-NEXT:    md5_checksum: 05ab89f5481bc9f2d037e7886641e919
; DWARF-DUMP-NEXT:          source: "int main()\n{\n    return 0;\n}\n"
; DWARF-DUMP-EMPTY:
; DWARF-DUMP-NEXT: Address            Line   Column File   ISA Discriminator OpIndex Flags
; DWARF-DUMP-NEXT: ------------------ ------ ------ ------ --- ------------- ------- -------------
; DWARF-DUMP-NEXT: 0x0000000000000000      2      0      0   0             0       0  is_stmt
; DWARF-DUMP-NEXT: 0x000000000000001c      3      5      0   0             0       0  is_stmt prologue_end
; DWARF-DUMP-NEXT: 0x0000000000000028      3      5      0   0             0       0  epilogue_begin
; DWARF-DUMP-NEXT: 0x0000000000000030      3      5      0   0             0       0  end_sequence

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
