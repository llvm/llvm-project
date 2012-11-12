# RUN: llvm-mc -triple mblaze-unknown-unknown -show-encoding %s | FileCheck %s

# Test to ensure that all register and immediate operands can be parsed by
# the assembly parser correctly. Testing the parsing of FSL immediate
# values is done in a different test.

# TYPE A:   OPCODE RD    RA    RB    FLAGS
# BINARY:   000000 00000 00000 00000 00000000000

# CHECK:    add
# BINARY:   000000 00000 00000 00000 00000000000
# CHECK:    encoding: [0x00,0x00,0x00,0x00]
            add     r0, r0, r0

# CHECK:    add
# BINARY:   000000 00001 00001 00001 00000000000
# CHECK:    encoding: [0x00,0x21,0x08,0x00]
            add     r1, r1, r1

# CHECK:    add
# BINARY:   000000 00010 00010 00010 00000000000
# CHECK:    encoding: [0x00,0x42,0x10,0x00]
            add     r2, r2, r2

# CHECK:    add
# BINARY:   000000 00011 00011 00011 00000000000
# CHECK:    encoding: [0x00,0x63,0x18,0x00]
            add     r3, r3, r3

# CHECK:    add
# BINARY:   000000 00100 00100 00100 00000000000
# CHECK:    encoding: [0x00,0x84,0x20,0x00]
            add     r4, r4, r4

# CHECK:    add
# BINARY:   000000 00101 00101 00101 00000000000
# CHECK:    encoding: [0x00,0xa5,0x28,0x00]
            add     r5, r5, r5

# CHECK:    add
# BINARY:   000000 00110 00110 00110 00000000000
# CHECK:    encoding: [0x00,0xc6,0x30,0x00]
            add     r6, r6, r6

# CHECK:    add
# BINARY:   000000 00111 00111 00111 00000000000
# CHECK:    encoding: [0x00,0xe7,0x38,0x00]
            add     r7, r7, r7

# CHECK:    add
# BINARY:   000000 01000 01000 01000 00000000000
# CHECK:    encoding: [0x01,0x08,0x40,0x00]
            add     r8, r8, r8

# CHECK:    add
# BINARY:   000000 01001 01001 01001 00000000000
# CHECK:    encoding: [0x01,0x29,0x48,0x00]
            add     r9, r9, r9

# CHECK:    add
# BINARY:   000000 01010 01010 01010 00000000000
# CHECK:    encoding: [0x01,0x4a,0x50,0x00]
            add     r10, r10, r10

# CHECK:    add
# BINARY:   000000 01011 01011 01011 00000000000
# CHECK:    encoding: [0x01,0x6b,0x58,0x00]
            add     r11, r11, r11

# CHECK:    add
# BINARY:   000000 01100 01100 01100 00000000000
# CHECK:    encoding: [0x01,0x8c,0x60,0x00]
            add     r12, r12, r12

# CHECK:    add
# BINARY:   000000 01101 01101 01101 00000000000
# CHECK:    encoding: [0x01,0xad,0x68,0x00]
            add     r13, r13, r13

# CHECK:    add
# BINARY:   000000 01110 01110 01110 00000000000
# CHECK:    encoding: [0x01,0xce,0x70,0x00]
            add     r14, r14, r14

# CHECK:    add
# BINARY:   000000 01111 01111 01111 00000000000
# CHECK:    encoding: [0x01,0xef,0x78,0x00]
            add     r15, r15, r15

# CHECK:    add
# BINARY:   000000 10000 10000 10000 00000000000
# CHECK:    encoding: [0x02,0x10,0x80,0x00]
            add     r16, r16, r16

# CHECK:    add
# BINARY:   000000 10001 10001 10001 00000000000
# CHECK:    encoding: [0x02,0x31,0x88,0x00]
            add     r17, r17, r17

# CHECK:    add
# BINARY:   000000 10010 10010 10010 00000000000
# CHECK:    encoding: [0x02,0x52,0x90,0x00]
            add     r18, r18, r18

# CHECK:    add
# BINARY:   000000 10011 10011 10011 00000000000
# CHECK:    encoding: [0x02,0x73,0x98,0x00]
            add     r19, r19, r19

# CHECK:    add
# BINARY:   000000 10100 10100 10100 00000000000
# CHECK:    encoding: [0x02,0x94,0xa0,0x00]
            add     r20, r20, r20

# CHECK:    add
# BINARY:   000000 10101 10101 10101 00000000000
# CHECK:    encoding: [0x02,0xb5,0xa8,0x00]
            add     r21, r21, r21

# CHECK:    add
# BINARY:   000000 10110 10110 10110 00000000000
# CHECK:    encoding: [0x02,0xd6,0xb0,0x00]
            add     r22, r22, r22

# CHECK:    add
# BINARY:   000000 10111 10111 10111 00000000000
# CHECK:    encoding: [0x02,0xf7,0xb8,0x00]
            add     r23, r23, r23

# CHECK:    add
# BINARY:   000000 11000 11000 11000 00000000000
# CHECK:    encoding: [0x03,0x18,0xc0,0x00]
            add     r24, r24, r24

# CHECK:    add
# BINARY:   000000 11001 11001 11001 00000000000
# CHECK:    encoding: [0x03,0x39,0xc8,0x00]
            add     r25, r25, r25

# CHECK:    add
# BINARY:   000000 11010 11010 11010 00000000000
# CHECK:    encoding: [0x03,0x5a,0xd0,0x00]
            add     r26, r26, r26

# CHECK:    add
# BINARY:   000000 11011 11011 11011 00000000000
# CHECK:    encoding: [0x03,0x7b,0xd8,0x00]
            add     r27, r27, r27

# CHECK:    add
# BINARY:   000000 11100 11100 11100 00000000000
# CHECK:    encoding: [0x03,0x9c,0xe0,0x00]
            add     r28, r28, r28

# CHECK:    add
# BINARY:   000000 11101 11101 11101 00000000000
# CHECK:    encoding: [0x03,0xbd,0xe8,0x00]
            add     r29, r29, r29

# CHECK:    add
# BINARY:   000000 11110 11110 11110 00000000000
# CHECK:    encoding: [0x03,0xde,0xf0,0x00]
            add     r30, r30, r30

# CHECK:    add
# BINARY:   000000 11111 11111 11111 00000000000
# CHECK:    encoding: [0x03,0xff,0xf8,0x00]
            add     r31, r31, r31

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000000000000
# CHECK:    encoding: [0x20,0x00,0x00,0x00]
            addi    r0, r0, 0

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000000000001
# CHECK:    encoding: [0x20,0x00,0x00,0x01]
            addi    r0, r0, 1

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000000000010
# CHECK:    encoding: [0x20,0x00,0x00,0x02]
            addi    r0, r0, 2

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000000000100
# CHECK:    encoding: [0x20,0x00,0x00,0x04]
            addi    r0, r0, 4

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000000001000
# CHECK:    encoding: [0x20,0x00,0x00,0x08]
            addi    r0, r0, 8

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000000010000
# CHECK:    encoding: [0x20,0x00,0x00,0x10]
            addi    r0, r0, 16

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000000100000
# CHECK:    encoding: [0x20,0x00,0x00,0x20]
            addi    r0, r0, 32

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000001000000
# CHECK:    encoding: [0x20,0x00,0x00,0x40]
            addi    r0, r0, 64

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000010000000
# CHECK:    encoding: [0x20,0x00,0x00,0x80]
            addi    r0, r0, 128

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000100000000
# CHECK:    encoding: [0x20,0x00,0x01,0x00]
            addi    r0, r0, 256

# CHECK:    addi
# BINARY:   001000 00000 00000 0000001000000000
# CHECK:    encoding: [0x20,0x00,0x02,0x00]
            addi    r0, r0, 512

# CHECK:    addi
# BINARY:   001000 00000 00000 0000010000000000
# CHECK:    encoding: [0x20,0x00,0x04,0x00]
            addi    r0, r0, 1024

# CHECK:    addi
# BINARY:   001000 00000 00000 0000100000000000
# CHECK:    encoding: [0x20,0x00,0x08,0x00]
            addi    r0, r0, 2048

# CHECK:    addi
# BINARY:   001000 00000 00000 0001000000000000
# CHECK:    encoding: [0x20,0x00,0x10,0x00]
            addi    r0, r0, 4096

# CHECK:    addi
# BINARY:   001000 00000 00000 0010000000000000
# CHECK:    encoding: [0x20,0x00,0x20,0x00]
            addi    r0, r0, 8192

# CHECK:    addi
# BINARY:   001000 00000 00000 0100000000000000
# CHECK:    encoding: [0x20,0x00,0x40,0x00]
            addi    r0, r0, 16384

# CHECK:    addi
# BINARY:   001000 00000 00000 1111111111111111
# CHECK:    encoding: [0x20,0x00,0xff,0xff]
            addi    r0, r0, -1

# CHECK:    addi
# BINARY:   001000 00000 00000 1111111111111110
# CHECK:    encoding: [0x20,0x00,0xff,0xfe]
            addi    r0, r0, -2

# CHECK:    addi
# BINARY:   001000 00000 00000 1111111111111100
# CHECK:    encoding: [0x20,0x00,0xff,0xfc]
            addi    r0, r0, -4

# CHECK:    addi
# BINARY:   001000 00000 00000 1111111111111000
# CHECK:    encoding: [0x20,0x00,0xff,0xf8]
            addi    r0, r0, -8

# CHECK:    addi
# BINARY:   001000 00000 00000 1111111111110000
# CHECK:    encoding: [0x20,0x00,0xff,0xf0]
            addi    r0, r0, -16

# CHECK:    addi
# BINARY:   001000 00000 00000 1111111111100000
# CHECK:    encoding: [0x20,0x00,0xff,0xe0]
            addi    r0, r0, -32

# CHECK:    addi
# BINARY:   001000 00000 00000 1111111111000000
# CHECK:    encoding: [0x20,0x00,0xff,0xc0]
            addi    r0, r0, -64

# CHECK:    addi
# BINARY:   001000 00000 00000 1111111110000000
# CHECK:    encoding: [0x20,0x00,0xff,0x80]
            addi    r0, r0, -128

# CHECK:    addi
# BINARY:   001000 00000 00000 1111111100000000
# CHECK:    encoding: [0x20,0x00,0xff,0x00]
            addi    r0, r0, -256

# CHECK:    addi
# BINARY:   001000 00000 00000 1111111000000000
# CHECK:    encoding: [0x20,0x00,0xfe,0x00]
            addi    r0, r0, -512

# CHECK:    addi
# BINARY:   001000 00000 00000 1111110000000000
# CHECK:    encoding: [0x20,0x00,0xfc,0x00]
            addi    r0, r0, -1024

# CHECK:    addi
# BINARY:   001000 00000 00000 1111100000000000
# CHECK:    encoding: [0x20,0x00,0xf8,0x00]
            addi    r0, r0, -2048

# CHECK:    addi
# BINARY:   001000 00000 00000 1111000000000000
# CHECK:    encoding: [0x20,0x00,0xf0,0x00]
            addi    r0, r0, -4096

# CHECK:    addi
# BINARY:   001000 00000 00000 1110000000000000
# CHECK:    encoding: [0x20,0x00,0xe0,0x00]
            addi    r0, r0, -8192

# CHECK:    addi
# BINARY:   001000 00000 00000 1100000000000000
# CHECK:    encoding: [0x20,0x00,0xc0,0x00]
            addi    r0, r0, -16384

# CHECK:    addi
# BINARY:   001000 00000 00000 1000000000000000
# CHECK:    encoding: [0x20,0x00,0x80,0x00]
            addi    r0, r0, -32768
