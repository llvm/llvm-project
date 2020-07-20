; RUN: llvm-mc -n -triple arm64-apple-darwin10 %s -filetype=obj -o - | llvm-readobj -r --expand-relocs - | FileCheck %s

	.text
_fred:
	bl	_func
	bl	_func + 20

	adrp	x3, _data@page
        ldr	w2, [x3, _data@pageoff]

        add	x3, x3, _data@pageoff + 4

	adrp	x3, _data@page+1
        ldr	w2, [x3, _data@pageoff + 4]

	adrp	x3, _data_ext@gotpage
        ldr	w2, [x3, _data_ext@gotpageoff]
        adrp    x0, L_.str@PAGE

	.data
_data:
        .quad _foo
        .quad _foo + 4
        .quad _foo - _bar
        .quad _foo - _bar + 4

        .long _foo - _bar

        .quad _foo@got
        .long _foo@got - .

        .section __TEXT,__cstring,cstring_literals
L_.str:
        .asciz "foo"


; CHECK:     Relocations [
; CHECK-NEXT:  Section __text {
; CHECK-NEXT:    Relocation {
; CHECK-NEXT:       Offset: 0x24
; CHECK-NEXT:       PCRel: 1
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_PAGE21 (3)
; CHECK-NEXT:       Symbol: L_.str
; CHECK-NEXT:     }
; CHECK-NEXT:    Relocation {
; CHECK-NEXT:       Offset: 0x20
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_GOT_LOAD_PAGEOFF12 (6)
; CHECK-NEXT:       Symbol: _data_ext
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x1C
; CHECK-NEXT:       PCRel: 1
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_GOT_LOAD_PAGE21 (5)
; CHECK-NEXT:       Symbol: _data_ext
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x18
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_ADDEND (10)
; CHECK-NEXT:       Section: - (4)
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x18
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_PAGEOFF12 (4)
; CHECK-NEXT:       Symbol: _data
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x14
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_ADDEND (10)
; CHECK-NEXT:       Section: __text
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x14
; CHECK-NEXT:       PCRel: 1
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_PAGE21 (3)
; CHECK-NEXT:       Symbol: _data
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x10
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_ADDEND (10)
; CHECK-NEXT:       Section: - (4)
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x10
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_PAGEOFF12 (4)
; CHECK-NEXT:       Symbol: _data
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0xC
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_PAGEOFF12 (4)
; CHECK-NEXT:       Symbol: _data
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x8
; CHECK-NEXT:       PCRel: 1
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_PAGE21 (3)
; CHECK-NEXT:       Symbol: _data
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x4
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_ADDEND (10)
; CHECK-NEXT:       Section: - (20)
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x4
; CHECK-NEXT:       PCRel: 1
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_BRANCH26 (2)
; CHECK-NEXT:       Symbol: _func
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x0
; CHECK-NEXT:       PCRel: 1
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_BRANCH26 (2)
; CHECK-NEXT:       Symbol: _func
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT:   Section __data {
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x2C
; CHECK-NEXT:       PCRel: 1
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_POINTER_TO_GOT (7)
; CHECK-NEXT:       Symbol: _foo
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x24
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 3
; CHECK-NEXT:       Type: ARM64_RELOC_POINTER_TO_GOT (7)
; CHECK-NEXT:       Symbol: _foo
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x20
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_SUBTRACTOR (1)
; CHECK-NEXT:       Symbol: _bar
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x20
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_UNSIGNED (0)
; CHECK-NEXT:       Symbol: _foo
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x18
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 3
; CHECK-NEXT:       Type: ARM64_RELOC_SUBTRACTOR (1)
; CHECK-NEXT:       Symbol: _bar
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x18
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 3
; CHECK-NEXT:       Type: ARM64_RELOC_UNSIGNED (0)
; CHECK-NEXT:       Symbol: _foo
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x10
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 3
; CHECK-NEXT:       Type: ARM64_RELOC_SUBTRACTOR (1)
; CHECK-NEXT:       Symbol: _bar
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x10
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 3
; CHECK-NEXT:       Type: ARM64_RELOC_UNSIGNED (0)
; CHECK-NEXT:       Symbol: _foo
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x8
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 3
; CHECK-NEXT:       Type: ARM64_RELOC_UNSIGNED (0)
; CHECK-NEXT:       Symbol: _foo
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x0
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 3
; CHECK-NEXT:       Type: ARM64_RELOC_UNSIGNED (0)
; CHECK-NEXT:       Symbol: _foo
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT: ]
