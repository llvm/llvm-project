@ RUN: llvm-mc -triple=thumbv7-apple-darwin -mcpu=cortex-a8 -show-encoding < %s | FileCheck %s

@------------------------------------------------------------------------------
@ unconditional branches accept narrow suffix and encode to short encodings
@------------------------------------------------------------------------------

         b.n    #-2048
         b.n    #2046

@ CHECK: b	#-2048                  @ encoding: [0x00,0xe4]
@ CHECK: b	#2046                   @ encoding: [0xff,0xe3]

@------------------------------------------------------------------------------
@ unconditional branches accept wide suffix and encode to wide encodings
@------------------------------------------------------------------------------

         b.w    #-2048
         b.w    #2046
         b.w    #-1677216
         b.w    #1677214

@ CHECK: b.w	#-2048                  @ encoding: [0xff,0xf7,0x00,0xbc]
@ CHECK: b.w	#2046                   @ encoding: [0x00,0xf0,0xff,0xbb]
@ CHECK: b.w	#-1677216               @ encoding: [0x66,0xf6,0x30,0xbc]
@ CHECK: b.w	#1677214                @ encoding: [0x99,0xf1,0xcf,0xbb]

@------------------------------------------------------------------------------
@ unconditional branches without width suffix encode depending of offset size
@------------------------------------------------------------------------------

         b      #-2048
         b      #2046
         b      #-2050
         b      #2048
         b      #-1677216
         b      #1677214

@ CHECK: b	#-2048                  @ encoding: [0x00,0xe4]
@ CHECK: b	#2046                   @ encoding: [0xff,0xe3]
@ CHECK: b.w	#-2050                  @ encoding: [0xff,0xf7,0xff,0xbb]
@ CHECK: b.w	#2048                   @ encoding: [0x00,0xf0,0x00,0xbc]
@ CHECK: b.w	#-1677216               @ encoding: [0x66,0xf6,0x30,0xbc]
@ CHECK: b.w	#1677214                @ encoding: [0x99,0xf1,0xcf,0xbb]

@------------------------------------------------------------------------------
@ unconditional branches with width narrow suffix in IT block 
@------------------------------------------------------------------------------

         it     eq
         beq.n  #-2048
         it     ne
         bne.n  #-2046

@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: beq	#-2048                  @ encoding: [0x00,0xe4] 
@ CHECK: it	ne                      @ encoding: [0x18,0xbf] 
@ CHECK: bne	#-2046                  @ encoding: [0x01,0xe4]

@------------------------------------------------------------------------------
@ unconditional branches with wide suffix in IT block
@------------------------------------------------------------------------------

         it     gt
         bgt.w  #-2048
         it     le
         ble.w  #2046
         it     ge
         bge.w  #-1677216
         it     lt
         blt.w  #1677214

@ CHECK: it	gt                      @ encoding: [0xc8,0xbf]
@ CHECK: bgt.w	#-2048                  @ encoding: [0xff,0xf7,0x00,0xbc]
@ CHECK: it	le                      @ encoding: [0xd8,0xbf]
@ CHECK: ble.w	#2046                   @ encoding: [0x00,0xf0,0xff,0xbb]
@ CHECK: it	ge                      @ encoding: [0xa8,0xbf]
@ CHECK: bge.w	#-1677216               @ encoding: [0x66,0xf6,0x30,0xbc]
@ CHECK: it	lt                      @ encoding: [0xb8,0xbf]
@ CHECK: blt.w	#1677214                @ encoding: [0x99,0xf1,0xcf,0xbb]

@------------------------------------------------------------------------------
@ conditional branches accept narrow suffix and encode to short encodings
@------------------------------------------------------------------------------

         beq.n    #-256
         bne.n    #254

@ CHECK: beq	#-256                   @ encoding: [0x80,0xd0]
@ CHECK: bne	#254                    @ encoding: [0x7f,0xd1]

@------------------------------------------------------------------------------
@ unconditional branches accept wide suffix and encode to wide encodings
@------------------------------------------------------------------------------

         bl.w     #256
         it ne
         blne.w   #256
         bmi.w    #-256
         bne.w    #254
         blt.w    #-1048576
         bge.w    #1048574

@ CHECK: bl	#256                    @ encoding: [0x00,0xf0,0x80,0xf8]
@ CHECK: it	ne                      @ encoding: [0x18,0xbf]
@ CHECK: blne	#256                    @ encoding: [0x00,0xf0,0x80,0xf8]
@ CHECK: bmi.w	#-256                   @ encoding: [0x3f,0xf5,0x80,0xaf]
@ CHECK: bne.w	#254                    @ encoding: [0x40,0xf0,0x7f,0x80]
@ CHECK: blt.w	#-1048576               @ encoding: [0xc0,0xf6,0x00,0x80]
@ CHECK: bge.w	#1048574                @ encoding: [0xbf,0xf2,0xff,0xaf]

@------------------------------------------------------------------------------
@ unconditional branches without width suffix encode depending of offset size
@------------------------------------------------------------------------------

         bne     #-256
         bgt     #254
         bne     #-258
         bgt     #256
         bne     #-1048576
         bgt     #1048574

@ CHECK: bne	#-256                   @ encoding: [0x80,0xd1]
@ CHECK: bgt	#254                    @ encoding: [0x7f,0xdc]
@ CHECK: bne.w	#-258                   @ encoding: [0x7f,0xf4,0x7f,0xaf]
@ CHECK: bgt.w	#256                    @ encoding: [0x00,0xf3,0x80,0x80]
@ CHECK: bne.w	#-1048576               @ encoding: [0x40,0xf4,0x00,0x80]
@ CHECK: bgt.w	#1048574                @ encoding: [0x3f,0xf3,0xff,0xaf]

@------------------------------------------------------------------------------
@ same branch insturction encoding to conditional or unconditional depending
@ on whether it is in an IT block or not
@------------------------------------------------------------------------------

         it     eq
         addeq  r0, r1
         bne    #128

@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: addeq	r0, r1                  @ encoding: [0x08,0x44]
@ CHECK: bne	#128                    @ encoding: [0x40,0xd1]

         ite    eq
         addeq  r0, r1
         bne    #128

@ CHECK: ite	eq                      @ encoding: [0x0c,0xbf]
@ CHECK: addeq	r0, r1                  @ encoding: [0x08,0x44]
@ CHECK: bne	#128                    @ encoding: [0x40,0xe0]

@ RUN: llvm-mc -triple=thumbv7-apple-darwin -mcpu=cortex-a8 -show-encoding < %s | FileCheck %s

@------------------------------------------------------------------------------
@ unconditional branches accept narrow suffix and encode to short encodings
@------------------------------------------------------------------------------

         b.n    #-2048
         b.n    #2046

@ CHECK: b	#-2048                  @ encoding: [0x00,0xe4]
@ CHECK: b	#2046                   @ encoding: [0xff,0xe3]

@------------------------------------------------------------------------------
@ unconditional branches accept wide suffix and encode to wide encodings
@------------------------------------------------------------------------------

         b.w    #-2048
         b.w    #2046
         b.w    #-1677216
         b.w    #1677214

@ CHECK: b.w	#-2048                  @ encoding: [0xff,0xf7,0x00,0xbc]
@ CHECK: b.w	#2046                   @ encoding: [0x00,0xf0,0xff,0xbb]
@ CHECK: b.w	#-1677216               @ encoding: [0x66,0xf6,0x30,0xbc]
@ CHECK: b.w	#1677214                @ encoding: [0x99,0xf1,0xcf,0xbb]

@------------------------------------------------------------------------------
@ unconditional branches without width suffix encode depending of offset size
@------------------------------------------------------------------------------

         b      #-2048
         b      #2046
         b      #-2050
         b      #2048
         b      #-1677216
         b      #1677214

@ CHECK: b	#-2048                  @ encoding: [0x00,0xe4]
@ CHECK: b	#2046                   @ encoding: [0xff,0xe3]
@ CHECK: b.w	#-2050                  @ encoding: [0xff,0xf7,0xff,0xbb]
@ CHECK: b.w	#2048                   @ encoding: [0x00,0xf0,0x00,0xbc]
@ CHECK: b.w	#-1677216               @ encoding: [0x66,0xf6,0x30,0xbc]
@ CHECK: b.w	#1677214                @ encoding: [0x99,0xf1,0xcf,0xbb]

@------------------------------------------------------------------------------
@ unconditional branches with width narrow suffix in IT block 
@------------------------------------------------------------------------------

         it     eq
         beq.n  #-2048
         it     ne
         bne.n  #-2046

@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: beq	#-2048                  @ encoding: [0x00,0xe4] 
@ CHECK: it	ne                      @ encoding: [0x18,0xbf] 
@ CHECK: bne	#-2046                  @ encoding: [0x01,0xe4]

@------------------------------------------------------------------------------
@ unconditional branches with wide suffix in IT block
@------------------------------------------------------------------------------

         it     gt
         bgt.w  #-2048
         it     le
         ble.w  #2046
         it     ge
         bge.w  #-1677216
         it     lt
         blt.w  #1677214

@ CHECK: it	gt                      @ encoding: [0xc8,0xbf]
@ CHECK: bgt.w	#-2048                  @ encoding: [0xff,0xf7,0x00,0xbc]
@ CHECK: it	le                      @ encoding: [0xd8,0xbf]
@ CHECK: ble.w	#2046                   @ encoding: [0x00,0xf0,0xff,0xbb]
@ CHECK: it	ge                      @ encoding: [0xa8,0xbf]
@ CHECK: bge.w	#-1677216               @ encoding: [0x66,0xf6,0x30,0xbc]
@ CHECK: it	lt                      @ encoding: [0xb8,0xbf]
@ CHECK: blt.w	#1677214                @ encoding: [0x99,0xf1,0xcf,0xbb]

@------------------------------------------------------------------------------
@ conditional branches accept narrow suffix and encode to short encodings
@------------------------------------------------------------------------------

         beq.n    #-256
         bne.n    #254

@ CHECK: beq	#-256                   @ encoding: [0x80,0xd0]
@ CHECK: bne	#254                    @ encoding: [0x7f,0xd1]

@------------------------------------------------------------------------------
@ unconditional branches accept wide suffix and encode to wide encodings
@------------------------------------------------------------------------------

         bmi.w    #-256
         bne.w    #254
         blt.w    #-1048576
         bge.w    #1048574

@ CHECK: bmi.w	#-256                   @ encoding: [0x3f,0xf5,0x80,0xaf]
@ CHECK: bne.w	#254                    @ encoding: [0x40,0xf0,0x7f,0x80]
@ CHECK: blt.w	#-1048576               @ encoding: [0xc0,0xf6,0x00,0x80]
@ CHECK: bge.w	#1048574                @ encoding: [0xbf,0xf2,0xff,0xaf]

@------------------------------------------------------------------------------
@ unconditional branches without width suffix encode depending of offset size
@------------------------------------------------------------------------------

         bne     #-256
         bgt     #254
         bne     #-258
         bgt     #256
         bne     #-1048576
         bgt     #1048574

@ CHECK: bne	#-256                   @ encoding: [0x80,0xd1]
@ CHECK: bgt	#254                    @ encoding: [0x7f,0xdc]
@ CHECK: bne.w	#-258                   @ encoding: [0x7f,0xf4,0x7f,0xaf]
@ CHECK: bgt.w	#256                    @ encoding: [0x00,0xf3,0x80,0x80]
@ CHECK: bne.w	#-1048576               @ encoding: [0x40,0xf4,0x00,0x80]
@ CHECK: bgt.w	#1048574                @ encoding: [0x3f,0xf3,0xff,0xaf]

@------------------------------------------------------------------------------
@ same branch insturction encoding to conditional or unconditional depending
@ on whether it is in an IT block or not
@------------------------------------------------------------------------------

         it     eq
         addeq  r0, r1
         bne    #128

@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: addeq	r0, r1                  @ encoding: [0x08,0x44]
@ CHECK: bne	#128                    @ encoding: [0x40,0xd1]

         ite    eq
         addeq  r0, r1
         bne    #128

@ CHECK: ite	eq                      @ encoding: [0x0c,0xbf]
@ CHECK: addeq	r0, r1                  @ encoding: [0x08,0x44]
@ CHECK: bne	#128                    @ encoding: [0x40,0xe0]


@------------------------------------------------------------------------------
@ Branch targets destined for ARM mode must == 0 (mod 4), otherwise (mod 2).
@------------------------------------------------------------------------------

        b #2
        bl #2
        beq #2
        cbz r0, #2
        @ N.b. destination is "align(PC, 4) + imm" so imm is still 4-byte
        @ aligned even though current PC may not and destination must be.
        blx #4

@ CHECK: b	#2                      @ encoding: [0x01,0xe0]
@ CHECK: bl	#2                      @ encoding: [0x00,0xf0,0x01,0xf8]
@ CHECK: beq	#2                      @ encoding: [0x01,0xd0]
@ CHECK: cbz	r0, #2                  @ encoding: [0x08,0xb1]
@ CHECK: blx	#4                      @ encoding: [0x00,0xf0,0x02,0xe8]
