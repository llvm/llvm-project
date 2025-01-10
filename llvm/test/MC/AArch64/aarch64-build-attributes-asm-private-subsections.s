// RUN: llvm-mc -triple=aarch64 %s -o - | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o - | llvm-readelf --hex-dump=.ARM.attributes - | FileCheck %s --check-prefix=ELF

// ASM: .aeabi_subsection	private_subsection_1, optional, uleb128
// ASM: .aeabi_attribute 12, 257
// ASM: .aeabi_subsection	private_subsection_2, required, uleb128
// ASM: .aeabi_attribute 76, 257
// ASM: .aeabi_subsection	private_subsection_3, optional, ntbs
// ASM: .aeabi_attribute 34, hello_llvm
// ASM: .aeabi_subsection	private_subsection_4, required, ntbs
// ASM: .aeabi_attribute 777, hello_llvm
// ASM: .aeabi_subsection	private_subsection_1, optional, uleb128
// ASM: .aeabi_attribute 876, 257
// ASM: .aeabi_subsection	private_subsection_2, required, uleb128
// ASM: .aeabi_attribute 876, 257
// ASM: .aeabi_subsection private_subsection_3, optional, ntbs
// ASM: .aeabi_attribute 876, hello_llvm
// ASM: .aeabi_subsection	private_subsection_4, required, ntbs
// ASM: .aeabi_attribute 876, hello_llvm

// ELF: Hex dump of section '.ARM.attributes':
// ELF-NEXT: 0x00000000 41220000 00707269 76617465 5f737562 A"...private_sub
// ELF-NEXT: 0x00000010 73656374 696f6e5f 31000100 0c8102ec section_1.......
// ELF-NEXT: 0x00000020 06810222 00000070 72697661 74655f73 ..."...private_s
// ELF-NEXT: 0x00000030 75627365 6374696f 6e5f3200 00004c81 ubsection_2...L.
// ELF-NEXT: 0x00000040 02ec0681 02340000 00707269 76617465 .....4...private
// ELF-NEXT: 0x00000050 5f737562 73656374 696f6e5f 33000101 _subsection_3...
// ELF-NEXT: 0x00000060 2268656c 6c6f5f6c 6c766d00 ec066865 "hello_llvm...he
// ELF-NEXT: 0x00000070 6c6c6f5f 6c6c766d 00350000 00707269 llo_llvm.5...pri
// ELF-NEXT: 0x00000080 76617465 5f737562 73656374 696f6e5f vate_subsection_
// ELF-NEXT: 0x00000090 34000001 89066865 6c6c6f5f 6c6c766d 4.....hello_llvm
// ELF-NEXT: 0x000000a0 00ec0668 656c6c6f 5f6c6c76 6d00     ...hello_llvm.


.aeabi_subsection private_subsection_1, optional, uleb128
.aeabi_attribute 12, 257
.aeabi_subsection private_subsection_2, required, uleb128
.aeabi_attribute 76, 257
.aeabi_subsection private_subsection_3, optional, ntbs
.aeabi_attribute 34, hello_llvm
.aeabi_subsection private_subsection_4, required, ntbs
.aeabi_attribute 777, hello_llvm
.aeabi_subsection private_subsection_1, optional, uleb128
.aeabi_attribute 876, 257
.aeabi_subsection private_subsection_2, required, uleb128
.aeabi_attribute 876, 257
.aeabi_subsection private_subsection_3, optional, ntbs
.aeabi_attribute 876, hello_llvm
.aeabi_subsection private_subsection_4, required, ntbs
.aeabi_attribute 876, hello_llvm
