# RUN: llvm-mc %s -filetype obj -triple i386-pc-linux -o - | \
# RUN: not llvm-dwarfdump -debug-addr - 2> %t.err | FileCheck %s
# RUN: FileCheck %s -input-file %t.err -check-prefix=ERR

# ERR: address table at offset 0x0 has unsupported version 6
# ERR: address table at offset 0x20 has unsupported version 4
# ERR-NOT: {{.}}

# CHECK: .debug_addr contents
# CHECK-NEXT:     length = 0x0000000c, format = DWARF32, version = 0x0005, addr_size = 0x04, seg_size = 0x00
# CHECK-NEXT:     Addrs: [
# CHECK-NEXT:     0x00000002
# CHECK-NEXT:     0x00000003
# CHECK-NEXT:     ]
# CHECK-NOT:      {{.}}

	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long 8	                      # Length of Unit
	.short	5                     # DWARF version number
  .byte 1                       # DWARF unit type
	.byte	4                       # Address Size (in bytes)
	.long	.debug_abbrev           # Offset Into Abbrev. Section

	.section	.debug_addr,"",@progbits
.Ldebug_addr0:
  .long 12 # unit_length = .short + .byte + .byte + .long + .long
  .short 6 # version
  .byte 4  # address_size
  .byte 0  # segment_selector_size
  .long 0x00000000
  .long 0x00000001

	.section	.debug_addr,"",@progbits
.Ldebug_addr1:
  .long 12 # unit_length = .short + .byte + .byte + .long + .long
  .short 5 # version
  .byte 4  # address_size
  .byte 0  # segment_selector_size
  .long 0x00000002
  .long 0x00000003

	.section	.debug_addr,"",@progbits
.Ldebug_addr2:
  .long 12 # unit_length = .short + .byte + .byte + .long + .long
  .short 4 # version
  .byte 4  # address_size
  .byte 0  # segment_selector_size
  .long 0x00000000
  .long 0x00000001

