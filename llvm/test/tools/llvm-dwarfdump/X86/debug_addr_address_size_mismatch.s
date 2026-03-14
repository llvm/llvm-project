# RUN: llvm-mc %s -filetype obj -triple i386-pc-linux -o - | \
# RUN: llvm-dwarfdump -debug-addr - 2> %t.warn | FileCheck %s
# RUN: FileCheck %s -input-file %t.warn -check-prefix=WARN

# WARN: address table at offset 0x0 has address size 8 which is different from CU address size 4
# WARN-NOT: {{.}}

# CHECK: .debug_addr contents
# CHECK-NEXT:     length = 0x0000000c, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00
# CHECK-NEXT:     Addrs: [
# CHECK-NEXT:     0x0000000100000000
# CHECK-NEXT:     ]
# CHECK-NEXT:     length = 0x0000000c, format = DWARF32, version = 0x0005, addr_size = 0x04, seg_size = 0x00
# CHECK-NEXT:     Addrs: [
# CHECK-NEXT:     0x00000000
# CHECK-NEXT:     0x00000001
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
  .short 5 # version
  .byte 8  # address_size
  .byte 0  # segment_selector_size
  .long 0x00000000
  .long 0x00000001

	.section	.debug_addr,"",@progbits
.Ldebug_addr1:
  .long 12 # unit_length = .short + .byte + .byte + .long + .long
  .short 5 # version
  .byte 4  # address_size
  .byte 0  # segment_selector_size
  .long 0x00000000
  .long 0x00000001
