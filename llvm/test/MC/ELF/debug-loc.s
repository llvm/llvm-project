// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that we don't regress on the size of the line info section. We used
// to handle negative line diffs incorrectly which manifested as very
// large integers being passed to DW_LNS_advance_line.

// FIXME: This size is the same as gnu as, but we can probably do a bit better.
// FIXME2: We need a debug_line dumper so that we can test the actual contents.

// CHECK:      # Section 4
// CHECK-NEXT: (('sh_name', 0x00000011) # '.debug_line'
// CHECK-NEXT:  ('sh_type', 0x00000001)
// CHECK-NEXT:  ('sh_flags', 0x0000000000000000)
// CHECK-NEXT:  ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:  ('sh_offset', 0x0000000000000044)
// CHECK-NEXT:  ('sh_size', 0x000000000000003d)
// CHECK-NEXT:  ('sh_link', 0x00000000)
// CHECK-NEXT:  ('sh_info', 0x00000000)
// CHECK-NEXT:  ('sh_addralign', 0x0000000000000001)
// CHECK-NEXT:  ('sh_entsize', 0x0000000000000000)
// CHECK-NEXT: ),

	.section	.debug_line,"",@progbits
	.text
foo:
	.file 1 "Driver.ii"
	.loc 1 2 0
        nop
	.loc 1 4 0
        nop
	.loc 1 3 0
        nop
