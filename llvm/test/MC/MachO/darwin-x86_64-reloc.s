// RUN: llvm-mc -n -triple x86_64-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -r --expand-relocs - | FileCheck %s

// These examples are taken from <mach-o/x86_64/reloc.h>.

        .data
        .long 0

        .text
_foo:
        ret

_baz:
        call _foo
 	call _foo+4
 	movq _foo@GOTPCREL(%rip), %rax
 	pushq _foo@GOTPCREL(%rip)
 	movl _foo(%rip), %eax
 	movl _foo+4(%rip), %eax
 	movb  $0x12, _foo(%rip)
 	movl  $0x12345678, _foo(%rip)
 	.quad _foo
_bar:
 	.quad _foo+4
 	.quad _foo - _bar
 	.quad _foo - _bar + 4
 	.long _foo - _bar
 	leaq L1(%rip), %rax
 	leaq L0(%rip), %rax
        addl $6,L0(%rip)
        addw $500,L0(%rip)
        addl $500,L0(%rip)

_prev:
        .space 12,0x90
 	.quad L1
L0:
        .quad L0
L_pc:
 	.quad _foo - L_pc
 	.quad _foo - L1
L1:
 	.quad L1 - _prev

        .data
.long	_foobar@GOTPCREL+4
.long	_foo@GOTPCREL+4

        .section	__DWARF,__debug_frame,regular,debug
        .quad L1
        .quad _ext_foo

// Make sure local label which overlaps with non-local one is assigned to the
// right atom.
        .text
_f2:
L2_0:
        addl $0, %eax
L2_1:        
_f3:
        addl L2_1 - L2_0, %eax
        
        .data
L4:     
        .long 0
        .text
        movl L4(%rip), %eax

        .section __TEXT,__literal8,8byte_literals
	.quad 0
L5:     
	.quad 0
f6:
        .quad 0
L6:
        .quad 0
        
        .text
	movl L5(%rip), %eax
	movl f6(%rip), %eax
	movl L6(%rip), %eax
        
        .data
        .quad L5
        .quad f6
	.quad L6

        .text
        cmpq $0, _foo@GOTPCREL(%rip)

// CHECK:      Relocations [
// CHECK-NEXT:   Section __data {
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x20
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: X86_64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Section: __literal8
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x18
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: X86_64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Symbol: f6
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x10
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: X86_64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Section: __literal8
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x8
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_GOT (4)
// CHECK-NEXT:       Symbol: _foo
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x4
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_GOT (4)
// CHECK-NEXT:       Symbol: _foobar
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   Section __text {
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0xDA
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_GOT (4)
// CHECK-NEXT:       Symbol: _foo
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0xD3
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_SIGNED (1)
// CHECK-NEXT:       Section: __literal8
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0xCD
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_SIGNED (1)
// CHECK-NEXT:       Symbol: f6
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0xC7
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_SIGNED (1)
// CHECK-NEXT:       Section: __literal8
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0xC1
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_SIGNED (1)
// CHECK-NEXT:       Section: __data
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0xA5
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: X86_64_RELOC_SUBTRACTOR (5)
// CHECK-NEXT:       Symbol: _prev
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0xA5
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: X86_64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Symbol: _foo
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x95
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: X86_64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Symbol: _prev
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x8D
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: X86_64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Symbol: _prev
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x79
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_SIGNED_4 (8)
// CHECK-NEXT:       Symbol: _prev
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x71
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_SIGNED_2 (7)
// CHECK-NEXT:       Symbol: _prev
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x69
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_SIGNED_1 (6)
// CHECK-NEXT:       Symbol: _prev
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x63
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_SIGNED (1)
// CHECK-NEXT:       Symbol: _prev
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x5C
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_SIGNED (1)
// CHECK-NEXT:       Symbol: _prev
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x3D
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: X86_64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Symbol: _foo
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x35
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: X86_64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Symbol: _foo
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x2D
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_SIGNED_4 (8)
// CHECK-NEXT:       Symbol: _foo
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x26
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_SIGNED_1 (6)
// CHECK-NEXT:       Symbol: _foo
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x20
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_SIGNED (1)
// CHECK-NEXT:       Symbol: _foo
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x1A
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_SIGNED (1)
// CHECK-NEXT:       Symbol: _foo
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x14
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_GOT (4)
// CHECK-NEXT:       Symbol: _foo
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0xE
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_GOT_LOAD (3)
// CHECK-NEXT:       Symbol: _foo
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x7
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_BRANCH (2)
// CHECK-NEXT:       Symbol: _foo
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x2
// CHECK-NEXT:       PCRel: 1
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_BRANCH (2)
// CHECK-NEXT:       Symbol: _foo
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   Section __debug_frame {
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x8
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: X86_64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Symbol: _ext_foo
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x0
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: X86_64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Section: __text
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]
