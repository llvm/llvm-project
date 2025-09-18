// RUN: llvm-mc -triple=aarch64 %s -o - | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o - | llvm-readelf --hex-dump=.ARM.attributes - | FileCheck %s --check-prefix=ELF

// ASM: .aeabi_subsection	private_subsection_1, optional, uleb128
// ASM: .aeabi_attribute 12, 257
// ASM: .aeabi_subsection	aeabi_2, required, uleb128
// ASM: .aeabi_attribute 76, 257
// ASM: .aeabi_subsection	aeabi_3, optional, ntbs
// ASM: .aeabi_attribute 34, hello_llvm
// ASM: .aeabi_subsection	private_subsection_4, required, ntbs
// ASM: .aeabi_attribute 777, "hello_llvm"
// ASM: .aeabi_subsection	private_subsection_1, optional, uleb128
// ASM: .aeabi_attribute 876, 257
// ASM: .aeabi_subsection	aeabi_2, required, uleb128
// ASM: .aeabi_attribute 876, 257
// ASM: .aeabi_subsection aeabi_3, optional, ntbs
// ASM: .aeabi_attribute 876, "hello_llvm"
// ASM: .aeabi_subsection	private_subsection_4, required, ntbs
// ASM: .aeabi_attribute 876, hello_llvm

// ELF: Hex dump of section '.ARM.attributes':
// ELF: 0x00000000 41220000 00707269 76617465 5f737562 A"...private_sub
// ELF: 0x00000010 73656374 696f6e5f 31000100 0c8102ec section_1.......
// ELF: 0x00000020 06810215 00000061 65616269 5f320000 .......aeabi_2..
// ELF: 0x00000030 004c8102 ec068102 29000000 61656162 .L......)...aeab
// ELF: 0x00000040 695f3300 01012268 656c6c6f 5f6c6c76 i_3..."hello_llv
// ELF: 0x00000050 6d00ec06 2268656c 6c6f5f6c 6c766d22 m..."hello_llvm"
// ELF: 0x00000060 00370000 00707269 76617465 5f737562 .7...private_sub
// ELF: 0x00000070 73656374 696f6e5f 34000001 89062268 section_4....."h
// ELF: 0x00000080 656c6c6f 5f6c6c76 6d2200ec 0668656c ello_llvm"...hel
// ELF: 0x00000090 6c6f5f6c 6c766d00                   lo_llvm.


.aeabi_subsection private_subsection_1, optional, uleb128
.aeabi_attribute 12, 257
.aeabi_subsection aeabi_2, required, uleb128
.aeabi_attribute 76, 257
.aeabi_subsection aeabi_3, optional, ntbs
.aeabi_attribute 34, hello_llvm
.aeabi_subsection private_subsection_4, required, ntbs
.aeabi_attribute 777, "hello_llvm"
.aeabi_subsection private_subsection_1, optional, uleb128
.aeabi_attribute 876, 257
.aeabi_subsection aeabi_2, required, uleb128
.aeabi_attribute 876, 257
.aeabi_subsection aeabi_3, optional, ntbs
.aeabi_attribute 876, "hello_llvm"
.aeabi_subsection private_subsection_4, required, ntbs
.aeabi_attribute 876, hello_llvm
