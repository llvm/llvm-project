## Test that --large-eh-encoding uses 8-byte pointers in the FDE CIE encoding
## for AArch64 ELF targets. The FDE CFI encoding is the only EH encoding that
## changes on AArch64: personality/LSDA/TType already default to sdata8 (see
## llvm/test/CodeGen/AArch64/large-eh-encoding.ll), so the FDE was the last
## sdata4 encoding that could overflow in large binaries.

## Default encoding: DW_EH_PE_pcrel | DW_EH_PE_sdata4 = 0x1B
# RUN: llvm-mc -filetype=obj %s -o %t.o -triple aarch64
# RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefix=SDATA4 %s

## With --large-eh-encoding: DW_EH_PE_pcrel | DW_EH_PE_sdata8 = 0x1C
# RUN: llvm-mc -filetype=obj %s -o %t.o -triple aarch64 --large-eh-encoding
# RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefix=SDATA8 %s

## --large-code-model should also use sdata8
# RUN: llvm-mc -filetype=obj %s -o %t.o -triple aarch64 --large-code-model
# RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefix=SDATA8 %s

func:
	.cfi_startproc
	.cfi_endproc

# SDATA4: {{[0-9a-f]+}} {{[0-9a-f]+}} 00000000 CIE
# SDATA4-NEXT:   Format: DWARF32
# SDATA4-NEXT:   Version: 1
# SDATA4-NEXT:   Augmentation: "zR"
# SDATA4: Augmentation data: 1B
## ^^ fde pointer encoding: DW_EH_PE_pcrel | DW_EH_PE_sdata4

# SDATA8: {{[0-9a-f]+}} {{[0-9a-f]+}} 00000000 CIE
# SDATA8-NEXT:   Format: DWARF32
# SDATA8-NEXT:   Version: 1
# SDATA8-NEXT:   Augmentation: "zR"
# SDATA8: Augmentation data: 1C
## ^^ fde pointer encoding: DW_EH_PE_pcrel | DW_EH_PE_sdata8
