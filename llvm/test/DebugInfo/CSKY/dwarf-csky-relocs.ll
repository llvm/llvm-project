; RUN: llc -filetype=obj -mtriple=csky %s -o %t.o
; RUN: llvm-readobj -r %t.o | FileCheck -check-prefix=READOBJ-RELOCS %s
; RUN: llvm-objdump --source %t.o | FileCheck --check-prefix=OBJDUMP-SOURCE %s
; RUN: llvm-dwarfdump --debug-info --debug-line %t.o | \
; RUN:     FileCheck -check-prefix=DWARF-DUMP %s

; Check that we actually have relocations, otherwise this is kind of pointless.
; READOBJ-RELOCS:  Section ({{.*}}) .rela.debug_info {
; READOBJ-RELOCS:    0x8 R_CKCORE_ADDR32 .debug_abbrev 0x0
; READOBJ-RELOCS-NEXT:    0x11 R_CKCORE_ADDR32 .debug_str_offsets 0x8
; READOBJ-RELOCS-NEXT:    0x15 R_CKCORE_ADDR32 .debug_line 0x0
; READOBJ-RELOCS-NEXT:    0x1F R_CKCORE_ADDR32 .debug_addr 0x8
; READOBJ-RELOCS:  Section ({{.*}}) .rela.debug_frame {
; READOBJ-RELOCS:    0x18 R_CKCORE_ADDR32 .debug_frame 0x0
; READOBJ-RELOCS-NEXT:    0x1C R_CKCORE_ADDR32 .text 0x0
; READOBJ-RELOCS:  Section ({{.*}}) .rela.debug_line {
; READOBJ-RELOCS:    0x22 R_CKCORE_ADDR32 .debug_line_str 0x0
; READOBJ-RELOCS-NEXT: 0x31 R_CKCORE_ADDR32 .debug_line_str 0x2
; READOBJ-RELOCS-NEXT: 0x46 R_CKCORE_ADDR32 .debug_line_str 0x16
; READOBJ-RELOCS-NEXT: 0x4F R_CKCORE_ADDR32 .text 0x0

; Check that we can print the source, even with relocations.
; OBJDUMP-SOURCE: Disassembly of section .text:
; OBJDUMP-SOURCE-EMPTY:
; OBJDUMP-SOURCE-NEXT: 00000000 <main>:
; OBJDUMP-SOURCE: ; {
; OBJDUMP-SOURCE: ; return 0;

; Check that we correctly dump the DWARF info, even with relocations.
; DWARF-DUMP: DW_AT_name        ("dwarf-csky-relocs.c")
; DWARF-DUMP: DW_AT_comp_dir    (".")
; DWARF-DUMP: DW_AT_name      ("main")
; DWARF-DUMP: DW_AT_decl_file ("{{.*}}dwarf-csky-relocs.c")
; DWARF-DUMP: DW_AT_decl_line (2)
; DWARF-DUMP: DW_AT_type      (0x00000032 "int")
; DWARF-DUMP: DW_AT_name      ("int")
; DWARF-DUMP: DW_AT_encoding  (DW_ATE_signed)
; DWARF-DUMP: DW_AT_byte_size (0x04)

; DWARF-DUMP: .debug_line contents:
; DWARF-DUMP-NEXT: debug_line[0x00000000]
; DWARF-DUMP-NEXT: Line table prologue:
; DWARF-DUMP-NEXT:     total_length: 0x00000059
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
; DWARF-DUMP-NEXT:            name: "dwarf-csky-relocs.c"
; DWARF-DUMP-NEXT:       dir_index: 0
; DWARF-DUMP-NEXT:    md5_checksum: ba6dbc7dc09162edb18beacd8474bcd3
; DWARF-DUMP-NEXT:          source: "int main()\n{\n    return 0;\n}\n"
; DWARF-DUMP-EMPTY:
; DWARF-DUMP-NEXT: Address            Line   Column File   ISA Discriminator Flags
; DWARF-DUMP-NEXT: ------------------ ------ ------ ------ --- ------------- -------------
; DWARF-DUMP-NEXT: 0x0000000000000000      2      0      0   0             0  is_stmt
; DWARF-DUMP-NEXT: 0x000000000000000e      3      3      0   0             0  is_stmt prologue_end
; DWARF-DUMP-NEXT: 0x000000000000001a      3      3      0   0             0  is_stmt end_sequence

; ModuleID = 'dwarf-csky-relocs.c'
source_filename = "dwarf-csky-relocs.c"
target datalayout = "e-m:e-S32-p:32:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:32-v128:32:32-a:0:32-Fi32-n32"
target triple = "csky-unknown-linux"

; Function Attrs: noinline nounwind optnone
define dso_local i32 @main() #0 !dbg !9 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  ret i32 0, !dbg !14
}

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="ck810" "target-features"="+2e3,+3e7,+7e10,+cache,+dsp1e2,+dspe60,+e1,+e2,+edsp,+elrw,+hard-tp,+high-registers,+hwdiv,+mp,+mp1e2,+nvic,+trust" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "dwarf-csky-relocs.c", directory: ".", checksumkind: CSK_MD5, checksum: "ba6dbc7dc09162edb18beacd8474bcd3", source: "int main()\0A{\0A    return 0;\0A}\0A")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"frame-pointer", i32 2}
!8 = !{!"clang"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 2, type: !10, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{}
!14 = !DILocation(line: 3, column: 3, scope: !9)
