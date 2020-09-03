// RUN: llvm-mc -triple=aarch64-darwin -filetype=obj %s -o - | \
// RUN:   llvm-objdump -r -d - | FileCheck --check-prefix=OBJ %s

// OBJ-LABEL: Disassembly of section __TEXT,__text:

  add x2, x3, _data@pageoff
// OBJ: [[addr:[0-9a-f]+]]: 62 00 00 91 add x2, x3, #0
// OBJ-NEXT: [[addr]]: ARM64_RELOC_PAGEOFF12	_data

  add x2, x3, #_data@pageoff, lsl #12
// OBJ: [[addr:[0-9a-f]+]]: 62 00 40 91 add x2, x3, #0, lsl #12
// OBJ-NEXT: [[addr]]: ARM64_RELOC_PAGEOFF12	_data
