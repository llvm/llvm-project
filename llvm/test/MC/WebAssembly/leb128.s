# RUN: llvm-mc -triple=wasm32-unknown-unknown %s | FileCheck --check-prefix=ASM %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s | llvm-readobj -r -S --sd - | FileCheck --check-prefix=OBJ %s

# RUN: not llvm-mc -filetype=obj -triple=wasm32-unknown-unknown --defsym ERR=1 %s -o /dev/null 2>&1 | FileCheck --check-prefix=ERR %s

# ASM: .section .custom_section.test,"",@
# ASM-NEXT: .int8 0
# ASM-NEXT: .int8 1
# ASM-NEXT: .int8 127
# ASM-NEXT: .ascii "\200\001"
# ASM-NEXT: .int8 23
# ASM-NEXT: .int8 42
# ASM-NEXT: .int8 127
# ASM-NEXT: .ascii "\200\177"

	.section .custom_section.test,"",@
	.uleb128	0
	.uleb128	1
	.uleb128	127
	.uleb128	128
	.uleb128	23, 42
	.sleb128	-1
	.sleb128	-128

# Forward and backward label differences within the same section
	.sleb128 .Lfoo - .Lbar
.Lfoo:
	.uleb128 .Lbar - .Lfoo
	.fill 16, 1, 0x90
.Lbar:

# OBJ:      Section {
# OBJ:        Name: test
# OBJ:        SectionData (
# OBJ-NEXT:     0000: 00017F80 01172A7F 807F6F11 90909090
# OBJ-NEXT:     0010: 90909090 90909090 90909090
# OBJ-NEXT:   )
# OBJ-NEXT: }

.ifdef ERR
# ERR: :[[#@LINE+1]]:22: error: .uleb128 expression is not absolute
	.uleb128 extern_sym - .Lfoo
# ERR: :[[#@LINE+1]]:17: error: .uleb128 expression is not absolute
	.uleb128 .Lfoo - extern_sym
# ERR: :[[#@LINE+1]]:13: error: .uleb128 expression is not absolute
	.uleb128 x - .Lfoo

	.section .custom_section.other,"",@
x:
# ERR: :[[#@LINE+1]]:17: error: .uleb128 expression is not absolute
	.uleb128 .Lfoo - x
.endif
