// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -r --expand-relocs | FileCheck %s

        .data
        .long undef
        .long (undef + 4)

        .globl local_a_ext
local_a_ext:
        .long local_a_ext

local_a:
        .long 0
local_a_elt:
        .long 0
local_b:
        .long local_b - local_c + 245
        .long 0
local_c:
        .long 0


        .long local_a_elt + 1
        .long local_a_elt + 10
        .short local_a_elt + 20
        .byte local_a_elt + 89

        .const

        .long
bar:
        .long local_a_elt - bar + 33

L0:
        .long L0
        .long L1

        .text
_f0:
L1:
        jmp	0xbabecafe
        jmp L0
        jmp L1
        ret

        .objc_class_name_A=0
	.globl .objc_class_name_A

        .text
        .globl _f1
        .weak_definition _f1
_f1:
        .data
        .long _f1
        .long _f1 + 4

// CHECK:     Relocations [
// CHECK-NEXT:  Section __text {
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x6
// CHECK-NEXT:      PCRel: 1
// CHECK-NEXT:      Length: 2
// CHECK-NEXT:      Type: GENERIC_RELOC_VANILLA (0)
// CHECK-NEXT:      Section: __const
// CHECK-NEXT:    }
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x1
// CHECK-NEXT:      PCRel: 1
// CHECK-NEXT:      Length: 2
// CHECK-NEXT:      Type: GENERIC_RELOC_VANILLA (0)
// CHECK-NEXT:      Section: - (0)
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  Section __data {
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x2F
// CHECK-NEXT:      PCRel: 0
// CHECK-NEXT:      Length: 2
// CHECK-NEXT:      Type: GENERIC_RELOC_VANILLA (0)
// CHECK-NEXT:      Symbol: _f1
// CHECK-NEXT:    }
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x2B
// CHECK-NEXT:      PCRel: 0
// CHECK-NEXT:      Length: 2
// CHECK-NEXT:      Type: GENERIC_RELOC_VANILLA (0)
// CHECK-NEXT:      Symbol: _f1
// CHECK-NEXT:    }
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x2A
// CHECK-NEXT:      PCRel: 0
// CHECK-NEXT:      Length: 0
// CHECK-NEXT:      Type: GENERIC_RELOC_VANILLA (0)
// CHECK-NEXT:      Value: 0x1D
// CHECK-NEXT:    }
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x28
// CHECK-NEXT:      PCRel: 0
// CHECK-NEXT:      Length: 1
// CHECK-NEXT:      Type: GENERIC_RELOC_VANILLA (0)
// CHECK-NEXT:      Value: 0x1D
// CHECK-NEXT:    }
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x24
// CHECK-NEXT:      PCRel: 0
// CHECK-NEXT:      Length: 2
// CHECK-NEXT:      Type: GENERIC_RELOC_VANILLA (0)
// CHECK-NEXT:      Value: 0x1D
// CHECK-NEXT:    }
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x20
// CHECK-NEXT:      PCRel: 0
// CHECK-NEXT:      Length: 2
// CHECK-NEXT:      Type: GENERIC_RELOC_VANILLA (0)
// CHECK-NEXT:      Value: 0x1D
// CHECK-NEXT:    }
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x14
// CHECK-NEXT:      PCRel: 0
// CHECK-NEXT:      Length: 2
// CHECK-NEXT:      Type: GENERIC_RELOC_LOCAL_SECTDIFF (4)
// CHECK-NEXT:      Value: 0x21
// CHECK-NEXT:    }
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x0
// CHECK-NEXT:      PCRel: 0
// CHECK-NEXT:      Length: 2
// CHECK-NEXT:      Type: GENERIC_RELOC_PAIR (1)
// CHECK-NEXT:      Value: 0x29
// CHECK-NEXT:    }
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x8
// CHECK-NEXT:      PCRel: 0
// CHECK-NEXT:      Length: 2
// CHECK-NEXT:      Type: GENERIC_RELOC_VANILLA (0)
// CHECK-NEXT:      Section: __data
// CHECK-NEXT:    }
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x4
// CHECK-NEXT:      PCRel: 0
// CHECK-NEXT:      Length: 2
// CHECK-NEXT:      Type: GENERIC_RELOC_VANILLA (0)
// CHECK-NEXT:      Symbol: undef
// CHECK-NEXT:    }
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x0
// CHECK-NEXT:      PCRel: 0
// CHECK-NEXT:      Length: 2
// CHECK-NEXT:      Type: GENERIC_RELOC_VANILLA (0)
// CHECK-NEXT:      Symbol: undef
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  Section __const {
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x8
// CHECK-NEXT:      PCRel: 0
// CHECK-NEXT:      Length: 2
// CHECK-NEXT:      Type: GENERIC_RELOC_VANILLA (0)
// CHECK-NEXT:      Section: __text
// CHECK-NEXT:    }
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x4
// CHECK-NEXT:      PCRel: 0
// CHECK-NEXT:      Length: 2
// CHECK-NEXT:      Type: GENERIC_RELOC_VANILLA (0)
// CHECK-NEXT:      Section: __const
// CHECK-NEXT:    }
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x0
// CHECK-NEXT:      PCRel: 0
// CHECK-NEXT:      Length: 2
// CHECK-NEXT:      Type: GENERIC_RELOC_LOCAL_SECTDIFF (4)
// CHECK-NEXT:      Value: 0x1D
// CHECK-NEXT:    }
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x0
// CHECK-NEXT:      PCRel: 0
// CHECK-NEXT:      Length: 2
// CHECK-NEXT:      Type: GENERIC_RELOC_PAIR (1)
// CHECK-NEXT:      Value: 0x40
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:]
