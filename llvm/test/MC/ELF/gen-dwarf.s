// RUN: llvm-mc -g -triple  i686-pc-linux-gnu %s -filetype=obj -o - | elf-dump | FileCheck %s


// Test that on ELF the debug info has a relocation to debug_abbrev and one to
// to debug_line.


    .text
    .globl foo
    .type foo, @function
    .align 4
foo:
    ret
    .size foo, .-foo

// Section 4 is .debug_line
// CHECK:       # Section 4
// CHECK-NEXT:  # '.debug_line'



// The two relocations, one to symbol 6 and one to 4
// CHECK:         # '.rel.debug_info'
// CHECK-NEXT:   ('sh_type',
// CHECK-NEXT:   ('sh_flags'
// CHECK-NEXT:   ('sh_addr',
// CHECK-NEXT:   ('sh_offset',
// CHECK-NEXT:   ('sh_size',
// CHECK-NEXT:   ('sh_link',
// CHECK-NEXT:   ('sh_info',
// CHECK-NEXT:   ('sh_addralign',
// CHECK-NEXT:   ('sh_entsize',
// CHECK-NEXT:   ('_relocations', [
// CHECK-NEXT:    # Relocation 0
// CHECK-NEXT:    (('r_offset', 0x00000006)
// CHECK-NEXT:     ('r_sym', 0x000006)
// CHECK-NEXT:     ('r_type', 0x01)
// CHECK-NEXT:    ),
// CHECK-NEXT:    # Relocation 1
// CHECK-NEXT:    (('r_offset', 0x0000000c)
// CHECK-NEXT:     ('r_sym', 0x000004)
// CHECK-NEXT:     ('r_type', 0x01)
// CHECK-NEXT:    ),


// Section 8 is .debug_abbrev
// CHECK:       # Section 8
// CHECK-NEXT:  (('sh_name', 0x00000001) # '.debug_abbrev'

// Symbol 4 is section 4 (.debug_line)
// CHECK:         # Symbol 4
// CHECK-NEXT:    (('st_name', 0x00000000) # ''
// CHECK-NEXT:     ('st_value', 0x00000000)
// CHECK-NEXT:     ('st_size', 0x00000000)
// CHECK-NEXT:     ('st_bind', 0x0)
// CHECK-NEXT:     ('st_type', 0x3)
// CHECK-NEXT:     ('st_other', 0x00)
// CHECK-NEXT:     ('st_shndx', 0x0004)
// CHECK-NEXT:    ),

// Symbol 6 is section 8 (.debug_abbrev)
// CHECK:         # Symbol 6
// CHECK-NEXT:    (('st_name', 0x00000000) # ''
// CHECK-NEXT:     ('st_value', 0x00000000)
// CHECK-NEXT:     ('st_size', 0x00000000)
// CHECK-NEXT:     ('st_bind', 0x0)
// CHECK-NEXT:     ('st_type', 0x3)
// CHECK-NEXT:     ('st_other', 0x00)
// CHECK-NEXT:     ('st_shndx', 0x0008)
// CHECK-NEXT:    ),
