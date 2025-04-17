; RUN: llc --filetype=obj --mtriple=loongarch64 --mattr=-relax %s -o %t.o
; RUN: llvm-readobj -r %t.o | FileCheck --check-prefixes=RELOCS-BOTH,RELOCS-NORL %s
; RUN: llvm-objdump --source %t.o | FileCheck --check-prefix=SOURCE %s
; RUN: llvm-dwarfdump --debug-info --debug-line %t.o | FileCheck --check-prefix=DWARF %s

; RUN: llc --filetype=obj --mtriple=loongarch64 --mattr=+relax --align-all-functions=2 %s -o %t.r.o
; RUN: llvm-readobj -r %t.r.o | FileCheck --check-prefixes=RELOCS-BOTH,RELOCS-ENRL %s
; RUN: llvm-objdump --source %t.r.o | FileCheck --check-prefix=SOURCE %s
; RUN: llvm-dwarfdump --debug-info --debug-line %t.r.o | FileCheck --check-prefix=DWARF %s

; RELOCS-BOTH:       Relocations [
; RELOCS-BOTH-NEXT:    Section ({{.*}}) .rela.text {
; RELOCS-BOTH-NEXT:      0x14 R_LARCH_PCALA_HI20 sym 0x0
; RELOCS-ENRL-NEXT:      0x14 R_LARCH_RELAX - 0x0
; RELOCS-BOTH-NEXT:      0x18 R_LARCH_PCALA_LO12 sym 0x0
; RELOCS-ENRL-NEXT:      0x18 R_LARCH_RELAX - 0x0
; RELOCS-BOTH-NEXT:    }
; RELOCS-BOTH:         Section ({{.*}}) .rela.debug_frame {
; RELOCS-NORL-NEXT:      0x1C R_LARCH_32 .debug_frame 0x0
; RELOCS-NORL-NEXT:      0x20 R_LARCH_64 .text 0x0
; RELOCS-ENRL-NEXT:      0x1C R_LARCH_32 .L0  0x0
; RELOCS-ENRL-NEXT:      0x20 R_LARCH_64 .L0  0x0
; RELOCS-ENRL-NEXT:      0x28 R_LARCH_ADD64 .L0  0x0
; RELOCS-ENRL-NEXT:      0x28 R_LARCH_SUB64 .L0  0x0
; RELOCS-ENRL-NEXT:      0x3F R_LARCH_ADD6 .L0  0x0
; RELOCS-ENRL-NEXT:      0x3F R_LARCH_SUB6 .L0  0x0
; RELOCS-BOTH-NEXT:    }
; RELOCS-BOTH:         Section ({{.*}}) .rela.debug_line {
; RELOCS-BOTH-NEXT:      0x22 R_LARCH_32 .debug_line_str 0x0
; RELOCS-BOTH-NEXT:      0x31 R_LARCH_32 .debug_line_str 0x2
; RELOCS-BOTH-NEXT:      0x46 R_LARCH_32 .debug_line_str 0x1B
; RELOCS-NORL-NEXT:      0x4F R_LARCH_64 .text 0x0
; RELOCS-ENRL-NEXT:      0x4F R_LARCH_64 .L0  0x0
; RELOCS-ENRL-NEXT:      0x5F R_LARCH_ADD16 .L0  0x0
; RELOCS-ENRL-NEXT:      0x5F R_LARCH_SUB16 .L0  0x0
; RELOCS-BOTH-NEXT:    }
; RELOCS-BOTH-NEXT:  ]

; SOURCE:  0000000000000000 <foo>:
; SOURCE:  ; {
; SOURCE:  ;   asm volatile(
; SOURCE:  ;   return 0;

; DWARF:       DW_AT_producer ("clang")
; DWARF:       DW_AT_name ("dwarf-loongarch-relocs.c")
; DWARF:       DW_AT_comp_dir (".")
; DWARF:       DW_AT_name ("foo")
; DWARF-NEXT:  DW_AT_decl_file ("{{.*}}dwarf-loongarch-relocs.c")
; DWARF-NEXT:  DW_AT_decl_line (1)
; DWARF-NEXT:  DW_AT_type (0x00000032 "int")
; DWARF:       DW_AT_name ("int")
; DWARF-NEXT:  DW_AT_encoding (DW_ATE_signed)
; DWARF-NEXT:  DW_AT_byte_size (0x04)
; DWARF:       .debug_line contents:
; DWARF-NEXT:  debug_line[0x00000000]
; DWARF-NEXT:  Line table prologue:
; DWARF-NEXT:      total_length: {{.*}}
; DWARF-NEXT:            format: DWARF32
; DWARF-NEXT:           version: 5
; DWARF-NEXT:      address_size: 8
; DWARF-NEXT:   seg_select_size: 0
; DWARF-NEXT:   prologue_length: 0x0000003e
; DWARF-NEXT:   min_inst_length: 1
; DWARF-NEXT:  max_ops_per_inst: 1
; DWARF-NEXT:   default_is_stmt: 1
; DWARF-NEXT:         line_base: -5
; DWARF-NEXT:        line_range: 14
; DWARF-NEXT:       opcode_base: 13
; DWARF-NEXT:  standard_opcode_lengths[DW_LNS_copy] = 0
; DWARF-NEXT:  standard_opcode_lengths[DW_LNS_advance_pc] = 1
; DWARF-NEXT:  standard_opcode_lengths[DW_LNS_advance_line] = 1
; DWARF-NEXT:  standard_opcode_lengths[DW_LNS_set_file] = 1
; DWARF-NEXT:  standard_opcode_lengths[DW_LNS_set_column] = 1
; DWARF-NEXT:  standard_opcode_lengths[DW_LNS_negate_stmt] = 0
; DWARF-NEXT:  standard_opcode_lengths[DW_LNS_set_basic_block] = 0
; DWARF-NEXT:  standard_opcode_lengths[DW_LNS_const_add_pc] = 0
; DWARF-NEXT:  standard_opcode_lengths[DW_LNS_fixed_advance_pc] = 1
; DWARF-NEXT:  standard_opcode_lengths[DW_LNS_set_prologue_end] = 0
; DWARF-NEXT:  standard_opcode_lengths[DW_LNS_set_epilogue_begin] = 0
; DWARF-NEXT:  standard_opcode_lengths[DW_LNS_set_isa] = 1
; DWARF-NEXT:  include_directories[  0] = "."
; DWARF-NEXT:  file_names[  0]:
; DWARF-NEXT:             name: "dwarf-loongarch-relocs.c"
; DWARF-NEXT:        dir_index: 0
; DWARF-NEXT:     md5_checksum: f44d6d71bc4da58b4abe338ca507c007
; DWARF-NEXT:           source: "{{.*}}"
; DWARF-EMPTY:
; DWARF-NEXT:  Address            Line   Column File   ISA Discriminator OpIndex Flags
; DWARF-NEXT:  ------------------ ------ ------ ------ --- ------------- ------- -------------
; DWARF-NEXT:  0x0000000000000000      2      0      0   0             0       0  is_stmt
; DWARF-NEXT:  0x0000000000000010      3      3      0   0             0       0  is_stmt prologue_end
; DWARF-NEXT:  0x0000000000000020     10      3      0   0             0       0  is_stmt
; DWARF-NEXT:  0x000000000000002c     10      3      0   0             0       0  epilogue_begin
; DWARF-NEXT:  0x0000000000000034     10      3      0   0             0       0  end_sequence

; ModuleID = 'dwarf-loongarch-relocs.c'
source_filename = "dwarf-loongarch-relocs.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "loongarch64"

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @foo() #0 !dbg !8 {
  call void asm sideeffect ".cfi_remember_state\0A\09.cfi_adjust_cfa_offset 16\0A\09nop\0A\09la.pcrel $$t0, sym\0A\09nop\0A\09.cfi_restore_state\0A\09", ""() #1, !dbg !12, !srcloc !13
  ret i32 0, !dbg !14
}

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="loongarch64" "target-features"="+64bit,+d,+f,+ual" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "dwarf-loongarch-relocs.c", directory: ".", checksumkind: CSK_MD5, checksum: "f44d6d71bc4da58b4abe338ca507c007", source: "int foo()\0A{\0A  asm volatile(\0A    \22.cfi_remember_state\\n\\t\22\0A    \22.cfi_adjust_cfa_offset 16\\n\\t\22\0A    \22nop\\n\\t\22\0A    \22la.pcrel $t0, sym\\n\\t\22\0A    \22nop\\n\\t\22\0A    \22.cfi_restore_state\\n\\t\22);\0A  return 0;\0A}\0A")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"direct-access-external-data", i32 0}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{!"clang"}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !9, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocation(line: 3, column: 3, scope: !8)
!13 = !{i64 34, i64 56, i64 92, i64 106, i64 134, i64 148, i64 177}
!14 = !DILocation(line: 10, column: 3, scope: !8)
