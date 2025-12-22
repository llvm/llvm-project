# RUN: llvm-mc --filetype=obj --gsframe -triple x86_64 %s -o %t.o
# RUN: llvm-readelf --sframe %t.o | FileCheck %s

# Tests selection for the proper FDE AddrX encoding at the boundaries
# between uint8_t, uint16_t, and uint32_t. The first FRE always fits
# anywhere, because its address-offset is zero. The last FRE
# determines the smallest AddrX it is possible to use.  Align
# functions to 1024 to make it easier to interpet offsets.

	.cfi_sections .sframe

        .align 1024
fde0_uses_addr1:
# CHECK:        FuncDescEntry [0] {
# CHECK:          Start FRE Offset: 0x0
# CHECK-NEXT:          Num FREs: 2
# CHECK:            FRE Type: Addr1 (0x0)
	.cfi_startproc
# CHECK:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0x0
# CHECK-NEXT:          Return Address Signed: No
# CHECK-NEXT:          Offset Size: B1 (0x0)
# CHECK-NEXT:          Base Register: SP (0x1)
# CHECK-NEXT:          CFA Offset: 8
# CHECK-NEXT:          RA Offset: -8
# CHECK-NEXT:        }
        .fill 0xFF
	.cfi_def_cfa_offset 16
# CHECK-NEXT:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0xFF
# CHECK-NEXT:          Return Address Signed: No
# CHECK-NEXT:          Offset Size: B1 (0x0)
# CHECK-NEXT:          Base Register: SP (0x1)
# CHECK-NEXT:          CFA Offset: 16
# CHECK-NEXT:          RA Offset: -8
  	nop
        .cfi_endproc

        .align 1024	
fde1_uses_addr2:
# CHECK:        FuncDescEntry [1] {
# CHECK:          Start FRE Offset: 0x6
# CHECK-NEXT:          Num FREs: 2
# CHECK:            FRE Type: Addr2 (0x1)
	.cfi_startproc
# CHECK:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0x400
        .fill 0xFF + 1
	.cfi_def_cfa_offset 16
# CHECK:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0x500
        .cfi_endproc

.align 1024	
fde2_uses_addr2:
# CHECK:        FuncDescEntry [2] {
# CHECK:          Start FRE Offset: 0xE
# CHECK-NEXT:          Num FREs: 2
# CHECK:            FRE Type: Addr2 (0x1)
	.cfi_startproc
# CHECK:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0x800
        .fill 0xFFFF
	.cfi_def_cfa_offset 16
# CHECK:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0x107FF
	nop
        .cfi_endproc

        .align 1024
fde3_uses_addr4:
# CHECK:        FuncDescEntry [3] {
# CHECK:          Start FRE Offset: 0x16
# CHECK-NEXT:          Num FREs: 2
# CHECK:            FRE Type: Addr4 (0x2)
	.cfi_startproc
# CHECK:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0x10800
        .fill 0xFFFF + 1
# CHECK:        Frame Row Entry {
# CHECK-NEXT:          Start Address: 0x20800
	.cfi_def_cfa_offset 16
	nop
        .cfi_endproc

