# RUN: not llvm-mc -triple x86_64-unknown-unknown %s 2>&1 | FileCheck %s --check-prefix=ASM-ERR
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s | llvm-objdump -j .data  -s - | FileCheck %s --check-prefix=OBJDATA
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s | llvm-objdump -j .text -s - | FileCheck %s --check-prefix=OBJTEXT
.data

# OBJDATA: Contents of section .data
# OBJDATA-NEXT: 0000 aa0506ff
	
foo2:
# ASM-ERR: [[@LINE+1]]:5: error: expected absolute expression
.if . - foo2 == 0
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

.byte 0xff

# nop is a fixed size instruction so this should pass.
	
# OBJTEXT: Contents of section .text
# OBJTEXT-NEXT: 0000 9090ff34 25

.text
	
text1:
        nop
# ASM-ERR: [[@LINE+1]]:5: error: expected absolute expression
.if . - text1 == 1
	nop
.else
	ret
.endif
	push gs

# No additional errors.
#	
# ASM-ERR-NOT: {{[0-9]+}}:{{[0-9]+}}: error:
