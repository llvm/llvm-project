# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o %t.o
# RUN: llvm-dwarfdump -debug-addr %t.o | FileCheck %s

# CHECK:          .debug_addr contents
# CHECK-NEXT:     length = 0x00000014, version = 0x0005, addr_size = 0x08, seg_size = 0x00
# CHECK-NEXT:     Addrs: [
# CHECK-NEXT:     0x0000000100000000
# CHECK-NEXT:     0x0000000100000001
# CHECK-NEXT:     ]
# CHECK-NOT:      {{.}}

	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long 8	                      # Length of Unit
	.short	5                     # DWARF version number
  .byte 1                       # DWARF unit type
	.byte	8                       # Address Size (in bytes)
	.long	.debug_abbrev           # Offset Into Abbrev. Section

  .section  .debug_addr,"",@progbits
.Ldebug_addr0:
  .long 20 # unit_length = .short + .byte + .byte + .quad + .quad
  .short 5 # version
  .byte 8  # address_size
  .byte 0  # segment_selector_size
  .quad 0x0000000100000000
  .quad 0x0000000100000001
