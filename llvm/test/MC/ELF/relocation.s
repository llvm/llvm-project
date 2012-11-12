// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

// Test that we produce the correct relocation.

bar:
        movl	$bar, %edx        # R_X86_64_32
        movq	$bar, %rdx        # R_X86_64_32S
        movq	$bar, bar(%rip)   # R_X86_64_32S
        movl	bar, %edx         # R_X86_64_32S
        movq	bar, %rdx         # R_X86_64_32S
.long bar                         # R_X86_64_32
        leaq	foo@GOTTPOFF(%rip), %rax # R_X86_64_GOTTPOFF
        leaq	foo@TLSGD(%rip), %rax    # R_X86_64_TLSGD
        leaq	foo@TPOFF(%rax), %rax    # R_X86_64_TPOFF32
        leaq	foo@TLSLD(%rip), %rdi    # R_X86_64_TLSLD
        leaq	foo@dtpoff(%rax), %rcx   # R_X86_64_DTPOFF32
        pushq    $bar
        movq	foo(%rip), %rdx
        leaq    foo-bar(%r14),%r14
        addq	$bar,%rax         # R_X86_64_32S


// CHECK:  # Section 1
// CHECK: (('sh_name', 0x00000006) # '.text'

// CHECK: # Relocation 0
// CHECK-NEXT:  (('r_offset', 0x0000000000000001)
// CHECK-NEXT:   ('r_sym', 0x00000002)
// CHECK-NEXT:   ('r_type', 0x0000000a)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 1
// CHECK-NEXT:  (('r_offset', 0x0000000000000008)
// CHECK-NEXT:   ('r_sym', 0x00000002)
// CHECK-NEXT:   ('r_type', 0x0000000b)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 2
// CHECK-NEXT:  (('r_offset', 0x0000000000000013)
// CHECK-NEXT:   ('r_sym', 0x00000002)
// CHECK-NEXT:   ('r_type', 0x0000000b)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 3
// CHECK-NEXT:  (('r_offset', 0x000000000000001a)
// CHECK-NEXT:   ('r_sym', 0x00000002)
// CHECK-NEXT:   ('r_type', 0x0000000b)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 4
// CHECK-NEXT:  (('r_offset', 0x0000000000000022)
// CHECK-NEXT:   ('r_sym', 0x00000002)
// CHECK-NEXT:   ('r_type', 0x0000000b)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 5
// CHECK-NEXT:  (('r_offset', 0x0000000000000026)
// CHECK-NEXT:   ('r_sym', 0x00000002)
// CHECK-NEXT:   ('r_type', 0x0000000a)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 6
// CHECK-NEXT:  (('r_offset', 0x000000000000002d)
// CHECK-NEXT:   ('r_sym', 0x00000006)
// CHECK-NEXT:   ('r_type', 0x00000016)
// CHECK-NEXT:   ('r_addend', 0xfffffffffffffffc)

// CHECK:  # Relocation 7
// CHECK-NEXT:  (('r_offset', 0x0000000000000034)
// CHECK-NEXT:   ('r_sym', 0x00000006)
// CHECK-NEXT:   ('r_type', 0x00000013)
// CHECK-NEXT:   ('r_addend', 0xfffffffffffffffc)

// CHECK:  # Relocation 8
// CHECK-NEXT:  (('r_offset', 0x000000000000003b)
// CHECK-NEXT:   ('r_sym', 0x00000006)
// CHECK-NEXT:   ('r_type', 0x00000017)
// CHECK-NEXT:   ('r_addend', 0x0000000000000000)

// CHECK:  # Relocation 9
// CHECK-NEXT:  (('r_offset', 0x0000000000000042)
// CHECK-NEXT:   ('r_sym', 0x00000006)
// CHECK-NEXT:   ('r_type', 0x00000014)
// CHECK-NEXT:   ('r_addend', 0xfffffffffffffffc)

// CHECK:  # Relocation 10
// CHECK-NEXT:  (('r_offset', 0x0000000000000049)
// CHECK-NEXT:   ('r_sym', 0x00000006)
// CHECK-NEXT:   ('r_type', 0x00000015)
// CHECK-NEXT:   ('r_addend', 0x0000000000000000)

// CHECK: # Relocation 11
// CHECK-NEXT:  (('r_offset', 0x000000000000004e)
// CHECK-NEXT:   ('r_sym', 0x00000002)
// CHECK-NEXT:   ('r_type', 0x0000000b)
// CHECK-NEXT:   ('r_addend', 0x0000000000000000)

// CHECK: # Relocation 12
// CHECK-NEXT: (('r_offset', 0x0000000000000055)
// CHECK-NEXT:  ('r_sym', 0x00000006)
// CHECK-NEXT:  ('r_type', 0x00000002)
// CHECK-NEXT:  ('r_addend', 0xfffffffffffffffc)

// CHECK: # Relocation 13
// CHECK-NEXT: (('r_offset', 0x000000000000005c)
// CHECK-NEXT:  ('r_sym', 0x00000006)
// CHECK-NEXT:  ('r_type', 0x00000002)
// CHECK-NEXT:  ('r_addend', 0x000000000000005c)

// CHECK: # Relocation 14
// CHECK-NEXT: (('r_offset', 0x0000000000000063)
// CHECK-NEXT:  ('r_sym', 0x00000002)
// CHECK-NEXT:  ('r_type', 0x0000000b)
// CHECK-NEXT:  ('r_addend', 0x0000000000000000)

// CHECK:   # Symbol 2
// CHECK: (('st_name', 0x00000000) # ''
// CHECK:  ('st_bind', 0x0)
// CHECK:  ('st_type', 0x3)
// CHECK:  ('st_other', 0x00)
// CHECK:  ('st_shndx', 0x0001)
