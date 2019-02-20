; RUN: llc -march=mips -relocation-model=pic -mattr=+micromips \
; RUN:     -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s

; CHECK-LABEL: foo:
; CHECK-NEXT:     0:	41 a2 00 00 	lui	$2, 0
; CHECK-NEXT:     4:	30 42 00 00 	addiu	$2, $2, 0
; CHECK-NEXT:     8:	03 22 11 50 	addu	$2, $2, $25
; CHECK-NEXT:     c:	fc 42 00 00 	lw	$2, 0($2)
; CHECK-NEXT:    10:	69 20 	      lw16	$2, 0($2)
; CHECK-NEXT:    12:	40 c2 00 14 	bgtz	$2, 44 <foo+0x3e>
; CHECK-NEXT:    16:	00 00 00 00 	nop
; CHECK-NEXT:    1a:	33 bd ff f8 	addiu	$sp, $sp, -8
; CHECK-NEXT:    1e:	fb fd 00 00 	sw	$ra, 0($sp)
; CHECK-NEXT:    22:	41 a1 00 01 	lui	$1, 1
; CHECK-NEXT:    26:	40 60 00 02 	bal	8 <foo+0x2e>
; CHECK-NEXT:    2a:	30 21 04 68 	addiu	$1, $1, 1128
; CHECK-NEXT:    2e:	00 3f 09 50 	addu	$1, $ra, $1
; CHECK-NEXT:    32:	ff fd 00 00 	lw	$ra, 0($sp)
; CHECK-NEXT:    36:	00 01 0f 3c 	jr	$1
; CHECK-NEXT:    3a:	33 bd 00 08 	addiu	$sp, $sp, 8
; CHECK-NEXT:    3e:	94 00 00 02 	b	8 <foo+0x46>
; CHECK-NEXT:    42:	00 00 00 00 	nop
; CHECK-NEXT:    46:	30 20 4e 1f 	addiu	$1, $zero, 19999
; CHECK-NEXT:    4a:	b4 22 00 14 	bne	$2, $1, 44 <foo+0x76>
; CHECK-NEXT:    4e:	00 00 00 00 	nop
; CHECK-NEXT:    52:	33 bd ff f8 	addiu	$sp, $sp, -8
; CHECK-NEXT:    56:	fb fd 00 00 	sw	$ra, 0($sp)
; CHECK-NEXT:    5a:	41 a1 00 01 	lui	$1, 1
; CHECK-NEXT:    5e:	40 60 00 02 	bal	8 <foo+0x66>
; CHECK-NEXT:    62:	30 21 04 5c 	addiu	$1, $1, 1116
; CHECK-NEXT:    66:	00 3f 09 50 	addu	$1, $ra, $1
; CHECK-NEXT:    6a:	ff fd 00 00 	lw	$ra, 0($sp)
; CHECK-NEXT:    6e:	00 01 0f 3c 	jr	$1
; CHECK-NEXT:    72:	33 bd 00 08 	addiu	$sp, $sp, 8
; CHECK-NEXT:    76:	30 20 27 0f 	addiu	$1, $zero, 9999
; CHECK-NEXT:    7a:	94 22 00 14 	beq	$2, $1, 44 <foo+0xa6>
; CHECK-NEXT:    7e:	00 00 00 00 	nop
; CHECK-NEXT:    82:	33 bd ff f8 	addiu	$sp, $sp, -8
; CHECK-NEXT:    86:	fb fd 00 00 	sw	$ra, 0($sp)
; CHECK-NEXT:    8a:	41 a1 00 01 	lui	$1, 1
; CHECK-NEXT:    8e:	40 60 00 02 	bal	8 <foo+0x96>
; CHECK-NEXT:    92:	30 21 04 2c 	addiu	$1, $1, 1068
; CHECK-NEXT:    96:	00 3f 09 50 	addu	$1, $ra, $1
; CHECK-NEXT:    9a:	ff fd 00 00 	lw	$ra, 0($sp)
; CHECK-NEXT:    9e:	00 01 0f 3c 	jr	$1
; CHECK-NEXT:    a2:	33 bd 00 08 	addiu	$sp, $sp, 8

; CHECK:      10466:	00 00 00 00 	nop
; CHECK-NEXT: 1046a:	94 00 00 02 	b	8 <foo+0x10472>
; CHECK-NEXT: 1046e:	00 00 00 00 	nop
; CHECK-NEXT: 10472:	33 bd ff f8 	addiu	$sp, $sp, -8
; CHECK-NEXT: 10476:	fb fd 00 00 	sw	$ra, 0($sp)
; CHECK-NEXT: 1047a:	41 a1 00 01 	lui	$1, 1
; CHECK-NEXT: 1047e:	40 60 00 02 	bal	8 <foo+0x10486>
; CHECK-NEXT: 10482:	30 21 04 00 	addiu	$1, $1, 1024
; CHECK-NEXT: 10486:	00 3f 09 50 	addu	$1, $ra, $1
; CHECK-NEXT: 1048a:	ff fd 00 00 	lw	$ra, 0($sp)
; CHECK-NEXT: 1048e:	00 01 0f 3c 	jr	$1
; CHECK-NEXT: 10492:	33 bd 00 08 	addiu	$sp, $sp, 8
; CHECK-NEXT: 10496:	94 00 00 02 	b	8 <foo+0x1049e>

@x = external global i32, align 4

define void @foo() {
  %1 = load i32, i32* @x, align 4
  %2 = icmp sgt i32 %1, 0
  br i1 %2, label %la, label %lf

la:
  switch i32 %1, label %le [
    i32 9999, label %lb
    i32 19999, label %lc
  ]

lb:
  tail call void asm sideeffect ".space 0", ""()
  br label %le

lc:
  tail call void asm sideeffect ".space 0", ""()
  br label %le

le:
  tail call void asm sideeffect ".space 66500", ""()
  br label %lg

lf:
  tail call void asm sideeffect ".space 0", ""()
  br label %lg

lg:
  tail call void asm sideeffect ".space 0", ""()
  br label %li

li:
  tail call void asm sideeffect ".space 0", ""()
  ret void
}
