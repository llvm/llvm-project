// RUN: llvm-mc -triple=aarch64 %s -o - | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o - | llvm-readelf --hex-dump=.ARM.attributes - | FileCheck %s --check-prefix=ELF

// ASM: .aeabi_subsection	private_subsection_1, optional, uleb128
// ASM: .aeabi_attribute 12, 257
// ASM: .aeabi_subsection	private_subsection_2, required, uleb128
// ASM: .aeabi_attribute 76, 257
// ASM: .aeabi_subsection	private_subsection_3, optional, ntbs
// ASM: .aeabi_attribute 34, hello_llvm
// ASM: .aeabi_subsection	private_subsection_4, required, ntbs
// ASM: .aeabi_attribute 777, "hello_llvm"
// ASM: .aeabi_subsection	private_subsection_1, optional, uleb128
// ASM: .aeabi_attribute 876, 257
// ASM: .aeabi_subsection	private_subsection_2, required, uleb128
// ASM: .aeabi_attribute 876, 257
// ASM: .aeabi_subsection private_subsection_3, optional, ntbs
// ASM: .aeabi_attribute 876, "hello_llvm"
// ASM: .aeabi_subsection	private_subsection_4, required, ntbs
// ASM: .aeabi_attribute 876, hello_llvm

// ELF: Hex dump of section '.ARM.attributes':
// ELF-NEXT: 0x00000000 41220000 00707269 76617465 5f737562 A"...private_sub
// ELF-NEXT: 0x00000010 73656374 696f6e5f 31000100 0c8102ec section_1.......
// ELF-NEXT: 0x00000020 06810222 00000070 72697661 74655f73 ..."...private_s
// ELF-NEXT: 0x00000030 75627365 6374696f 6e5f3200 00004c81 ubsection_2...L.
// ELF-NEXT: 0x00000040 02ec0681 02360000 00707269 76617465 .....6...private
// ELF-NEXT: 0x00000050 5f737562 73656374 696f6e5f 33000101 _subsection_3...
// ELF-NEXT: 0x00000060 2268656c 6c6f5f6c 6c766d00 ec062268 "hello_llvm..."h
// ELF-NEXT: 0x00000070 656c6c6f 5f6c6c76 6d220037 00000070 ello_llvm".7...p
// ELF-NEXT: 0x00000080 72697661 74655f73 75627365 6374696f rivate_subsectio
// ELF-NEXT: 0x00000090 6e5f3400 00018906 2268656c 6c6f5f6c n_4....."hello_l
// ELF-NEXT: 0x000000a0 6c766d22 00ec0668 656c6c6f 5f6c6c76 lvm"...hello_llv
// ELF-NEXT: 0x000000b0 6d00                                m.


.aeabi_subsection private_subsection_1, optional, uleb128
.aeabi_attribute 12, 257
.aeabi_subsection private_subsection_2, required, uleb128
.aeabi_attribute 76, 257
.aeabi_subsection private_subsection_3, optional, ntbs
.aeabi_attribute 34, hello_llvm
.aeabi_subsection private_subsection_4, required, ntbs
.aeabi_attribute 777, "hello_llvm"
.aeabi_subsection private_subsection_1, optional, uleb128
.aeabi_attribute 876, 257
.aeabi_subsection private_subsection_2, required, uleb128
.aeabi_attribute 876, 257
.aeabi_subsection private_subsection_3, optional, ntbs
.aeabi_attribute 876, "hello_llvm"
.aeabi_subsection private_subsection_4, required, ntbs
.aeabi_attribute 876, hello_llvm
