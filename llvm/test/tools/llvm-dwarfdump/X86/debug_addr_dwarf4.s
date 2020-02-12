# RUN: llvm-mc %s -filetype obj -triple i386-pc-linux -o %t.o
# RUN: llvm-dwarfdump --debug-addr %t.o | FileCheck %s

# CHECK: .debug_addr contents
# CHECK-NEXT:     Addrs: [
# CHECK-NEXT:     0x00000000
# CHECK-NEXT:     0x00000001

	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long 7	                      # Length of Unit
	.short	4                     # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	4                       # Address Size (in bytes)
	.section	.debug_addr,"",@progbits
  .long 0x00000000
  .long 0x00000001
