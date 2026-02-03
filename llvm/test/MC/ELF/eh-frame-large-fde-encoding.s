// Test that --large-fde-encoding uses 8-byte pointers in FDE CIE encoding
// for x86_64 ELF targets.

// Default encoding: DW_EH_PE_pcrel | DW_EH_PE_sdata4 = 0x1B
// RUN: llvm-mc -filetype=obj %s -o %t.o -triple x86_64-unknown-linux-gnu
// RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefix=SDATA4 %s

// With --large-fde-encoding: DW_EH_PE_pcrel | DW_EH_PE_sdata8 = 0x1C
// RUN: llvm-mc -filetype=obj %s -o %t.o -triple x86_64-unknown-linux-gnu --large-fde-encoding
// RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefix=SDATA8 %s

// Also test with --large-code-model which should also use sdata8
// RUN: llvm-mc -filetype=obj %s -o %t.o -triple x86_64-unknown-linux-gnu --large-code-model
// RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefix=SDATA8 %s

func:
	.cfi_startproc
	.cfi_endproc

// SDATA4:      {{[0-9a-f]+}} {{[0-9a-f]+}} 00000000 CIE
// SDATA4-NEXT:   Format:                DWARF32
// SDATA4-NEXT:   Version:               1
// SDATA4-NEXT:   Augmentation:          "zR"
// SDATA4:        Augmentation data:     1B
//                                       ^^ fde pointer encoding: DW_EH_PE_pcrel | DW_EH_PE_sdata4

// SDATA8:      {{[0-9a-f]+}} {{[0-9a-f]+}} 00000000 CIE
// SDATA8-NEXT:   Format:                DWARF32
// SDATA8-NEXT:   Version:               1
// SDATA8-NEXT:   Augmentation:          "zR"
// SDATA8:        Augmentation data:     1C
//                                       ^^ fde pointer encoding: DW_EH_PE_pcrel | DW_EH_PE_sdata8
