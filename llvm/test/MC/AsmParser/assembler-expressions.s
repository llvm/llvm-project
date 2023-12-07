# RUN: not llvm-mc -triple x86_64-unknown-unknown %s 2>&1 | FileCheck %s --check-prefix=ASM-ERR --implicit-check-not=error:
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s | llvm-objdump -j .data  -s - | FileCheck %s --check-prefix=OBJDATA
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s | llvm-objdump -j .text -s - | FileCheck %s --check-prefix=OBJTEXT
.data

# OBJDATA: Contents of section .data
# OBJDATA-NEXT: 0000 aa01aa05 06000000 08ff

foo1:
# ASM-ERR: :[[#@LINE+1]]:5: error: expected absolute expression
.if . - foo1 == 0
    .byte 0xaa
.else
    .byte 0x00
.endif

foo2:
    .byte 1
# ASM-ERR: :[[#@LINE+1]]:5: error: expected absolute expression
.if foo2 - . == -1
    .byte 0xaa
.else
    .byte 0x00
.endif

foo3:
    .byte 5
# ASM-ERR: [[@LINE+1]]:5: error: expected absolute expression
.if . - foo3 == 1
   .byte 6
.else
   .byte 7
.endif

foo4:
.byte 0
.space 2
# ASM-ERR: :[[#@LINE+1]]:5: error: expected absolute expression
.if . - foo4 == 3
    .byte 8
.else
    .byte 9
.endif

.byte 0xff

# nop is a fixed size instruction so this should pass.

# OBJTEXT: Contents of section .text
# OBJTEXT-NEXT: 0000 909090ff 34250000 00009090 90785634
# OBJTEXT-NEXT: 0010 12785634 1290

.text
text1:
# ASM-ERR: [[@LINE+1]]:5: error: expected absolute expression
.if . - text1 == 0
	nop
.endif

text2:
        nop
# ASM-ERR: [[@LINE+1]]:5: error: expected absolute expression
.if . - text2 == 1
	nop
.else
	ret
.endif
	push gs
	nop
	nop
	nop
# No additional errors.
#
# ASM-ERR-NOT: {{[0-9]+}}:{{[0-9]+}}: error:


text3:
	.long 0x12345678
text4:
	.fill (text4-text3)/4, 4, 0x12345678
	nop





	
