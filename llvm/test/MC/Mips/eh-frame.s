// Test the bits of .eh_frame on mips that are already implemented correctly.

// RUN: llvm-mc -filetype=obj %s -o %t.o -triple mips-unknown-linux-gnu
// RUN: llvm-readobj -r %t.o | FileCheck --check-prefixes=RELOCS,ABS32 %s
// RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefixes=DWARF32,DWARF32_ABS %s

// RUN: llvm-mc -filetype=obj %s -o %t.o -triple mipsel-unknown-linux-gnu
// RUN: llvm-readobj -r %t.o | FileCheck --check-prefixes=RELOCS,ABS32 %s
// RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefixes=DWARF32,DWARF32_ABS %s

// RUN: llvm-mc -filetype=obj %s -o %t.o -triple mips64-unknown-linux-gnu
// RUN: llvm-readobj -r %t.o | FileCheck --check-prefixes=RELOCS,ABS64 %s
// RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefixes=DWARF64,DWARF64_ABS %s

// RUN: llvm-mc -filetype=obj %s -o %t.o -triple mips64el-unknown-linux-gnu
// RUN: llvm-readobj -r %t.o | FileCheck --check-prefixes=RELOCS,ABS64 %s
// RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefixes=DWARF64,DWARF64_ABS %s

/// Check that position-indenpendent code use PC-relative relocations:
// RUN: llvm-mc -filetype=obj %s -o %t.o -triple mips-unknown-linux-gnu --position-independent
// RUN: llvm-readobj -r %t.o | FileCheck --check-prefixes=RELOCS,PIC32 %s
// RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefixes=DWARF32,DWARF32_PIC %s

// RUN: llvm-mc -filetype=obj %s -o %t.o -triple mipsel-unknown-linux-gnu --position-independent
// RUN: llvm-readobj -r %t.o | FileCheck --check-prefixes=RELOCS,PIC32 %s
// RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefixes=DWARF32,DWARF32_PIC %s

// RUN: llvm-mc -filetype=obj %s -o %t.o -triple mips64-unknown-linux-gnu --position-independent
// RUN: llvm-readobj -r %t.o | FileCheck --check-prefixes=RELOCS,PIC64 %s
// RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefixes=DWARF64,DWARF64_PIC %s

// RUN: llvm-mc -filetype=obj %s -o %t.o -triple mips64el-unknown-linux-gnu --position-independent
// RUN: llvm-readobj -r %t.o | FileCheck --check-prefixes=RELOCS,PIC64 %s
// RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefixes=DWARF64,DWARF64_PIC %s

// RUN: llvm-mc -filetype=obj %s -o %t.o -triple mips64-unknown-linux-gnu --position-independent --large-code-model
// RUN: llvm-readobj -r %t.o | FileCheck --check-prefixes=RELOCS,PIC64 %s
// RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefixes=DWARF64,DWARF64_PIC %s

// RUN: llvm-mc -filetype=obj %s -o %t.o -triple mips64el-unknown-linux-gnu --position-independent  --large-code-model
// RUN: llvm-readobj -r %t.o | FileCheck --check-prefixes=RELOCS,PIC64 %s
// RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefixes=DWARF64,DWARF64_PIC %s

func:
	.cfi_startproc
	.cfi_endproc

// RELOCS:      Relocations [
// RELOCS:        Section ({{.+}}) .rel{{a?}}.eh_frame {
// ABS32-NEXT:      R_MIPS_32
// ABS64-NEXT:      R_MIPS_64/R_MIPS_NONE/R_MIPS_NONE
// PIC32-NEXT:      R_MIPS_PC32
// PIC64-NEXT:      R_MIPS_PC32/R_MIPS_NONE/R_MIPS_NONE
// RELOCS-NEXT:   }

// DWARF32: 00000000 00000010 00000000 CIE
// DWARF32-NEXT:     Format:                DWARF32
// DWARF32-NEXT:     Version:               1
// DWARF32-NEXT:     Augmentation:          "zR"
// DWARF32-NEXT:     Code alignment factor: 1
// DWARF32-NEXT:     Data alignment factor: -4
// DWARF32-NEXT:     Return address column: 31
// DWARF32_ABS-NEXT: Augmentation data: 0B
//                                      ^^ fde pointer encoding: DW_EH_PE_sdata4
// DWARF32_PIC-NEXT: Augmentation data: 1B
//                                      ^^ fde pointer encoding: DW_EH_PE_pcrel | DW_EH_PE_sdata4
// DWARF32-EMPTY:
// DWARF32-NEXT:     DW_CFA_def_cfa_register: SP_64
//
// DWARF32_ABS: 00000014 00000010 00000018 FDE cie=00000000 pc=00000000...00000000
// DWARF32_PIC: 00000014 00000010 00000018 FDE cie=00000000 pc=0000001c...0000001c
// DWARF32-NEXT:     Format:       DWARF32
// DWARF32-NEXT:     DW_CFA_nop:
// DWARF32-NEXT:     DW_CFA_nop:
// DWARF32-NEXT:     DW_CFA_nop:


// DWARF64: 00000000 00000010 00000000 CIE
// DWARF64-NEXT:     Format:                DWARF32
// DWARF64-NEXT:     Version:               1
// DWARF64-NEXT:     Augmentation:          "zR"
// DWARF64-NEXT:     Code alignment factor: 1
// DWARF64-NEXT:     Data alignment factor: -8
//                                          ^^ GAS uses -4. Should be ok as long as
//                                             all offsets we need are a multiple of 8.
// DWARF64-NEXT:     Return address column: 31
// DWARF64_ABS-NEXT: Augmentation data: 0C
//                                      ^^ fde pointer encoding: DW_EH_PE_sdata8
// DWARF64_PIC:      Augmentation data: 1B
//                                      ^^ fde pointer encoding: DW_EH_PE_pcrel | DW_EH_PE_sdata4
// DWARF64-EMPTY:
// DWARF64-NEXT:     DW_CFA_def_cfa_register: SP_64
// DWARF64_PIC-NEXT: DW_CFA_nop:
//
// DWARF64_ABS:      00000014 00000018 00000018 FDE cie=00000000 pc=00000000...00000000
// DWARF64_PIC:      00000014 00000010 00000018 FDE cie=00000000 pc=00000000...00000000
// DWARF64-NEXT:     Format:       DWARF32
// DWARF64-NEXT:     DW_CFA_nop:
// DWARF64-NEXT:     DW_CFA_nop:
// DWARF64-NEXT:     DW_CFA_nop:
