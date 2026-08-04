# RUN: llvm-mc -filetype=obj -triple=hexagon %s -o %t.abs.o
# RUN: llvm-readobj -r %t.abs.o | FileCheck %s --check-prefix=ABS
# RUN: llvm-mc -filetype=obj -triple=hexagon -position-independent %s -o %t.pic.o
# RUN: llvm-readobj -r %t.pic.o | FileCheck %s --check-prefix=PIC

// Hexagon selects the FDE pointer encoding from whether the object is PIC.
func:
  .cfi_startproc
  .cfi_endproc

// ABS:      Relocations [
// ABS-NEXT:   Section ({{.*}}) .rela.eh_frame {
// ABS-NEXT:     R_HEX_32
// ABS-NEXT:   }
// ABS-NEXT: ]

// PIC:      Relocations [
// PIC-NEXT:   Section ({{.*}}) .rela.eh_frame {
// PIC-NEXT:     R_HEX_32_PCREL
// PIC-NEXT:   }
// PIC-NEXT: ]
