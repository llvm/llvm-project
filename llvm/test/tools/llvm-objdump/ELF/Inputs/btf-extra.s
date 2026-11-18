.section .BTF,"",@progbits
.short 0xeb9f
.byte 1, 0
.long 24, 0, strings-types, strings-types, end-strings
types:
// Integer encodings, including a nonzero offset to an empty string.
.long 1, 0x01000000, 4, 32
.long 0, 0x01000000, 4, 0x02000020
.long 0, 0x01000000, 4, 0x04000020
.long 0, 0x01000000, 4, 0x03000020
// All variable and function linkages.
.long 2, 0x0e000000, 1, 0
.long 2, 0x0e000000, 1, 1
.long 2, 0x0e000000, 1, 2
.long 2, 0x0e000000, 1, 3
.long 2, 0x0c000000, 1
.long 2, 0x0c000001, 1
.long 2, 0x0c000002, 1
.long 2, 0x0c000003, 1
// A data section referring to void, an INT and a VAR.
.long 0, 0x0f000003, 8
.long 0, 0, 0, 1, 0, 4, 5, 4, 4
// Unsigned high-bit enumeration values.
.long 0, 0x06000001, 4, 2, 0xffffffff
.long 0, 0x13000001, 8, 2, 0xffffffff, 0xffffffff
// A vararg sentinel and two members in disk order.
.long 0, 0x0d000001, 0, 0, 0
.long 0, 0x04000002, 8
.long 2, 1, 32, 4, 1, 0
strings:
.asciz ""
.asciz ""
.asciz "a"
.asciz "b"
end:
