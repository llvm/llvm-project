! RUN: llvm-mc -triple=lanai -show-encoding -show-inst < %s | FileCheck %s

.text
   .align 4
   .global jump1

    bt %r5
! CHECK: encoding: [0xc1,0x00,0x2d,0x00]
! CHECK-NEXT: <MCInst #{{[0-9]+}} JR{{$}}
! CHECK-NEXT:  <MCOperand Reg:12>>

! BR classes
    bt 0x1234
! CHECK: encoding: [0xe0,0x00,0x12,0x34]
! CHECK-NEXT: <MCInst #{{[0-9]+}} BT{{$}}
! CHECK-NEXT: <MCOperand Imm:4660>

jump1:
    blt 2000
! CHECK: encoding: [0xec,0x00,0x07,0xd1]
! CHECK-NEXT: <MCInst #{{[0-9]+}} BRCC{{$}}
! CHECK-NEXT: <MCOperand Imm:2000>
! CHECK-NEXT: <MCOperand Imm:13>

jump2:
    blt jump1
! CHECK: encoding: [0b1110110A,A,A,0x01'A']
! CHECK-NEXT: fixup A - offset: 0, value: jump1, kind: FIXUP_LANAI_25
! CHECK-NEXT: <MCInst #{{[0-9]+}} BRCC{{$}}
! CHECK-NEXT: <MCOperand Expr:specifier(0,jump1)>
! CHECK-NEXT: <MCOperand Imm:13>

    bpl jump2
! CHECK: encoding: [0b1110101A,A,A,A]
! CHECK-NEXT: fixup A - offset: 0, value: jump2, kind: FIXUP_LANAI_25
! CHECK-NEXT: <MCInst #{{[0-9]+}} BRCC{{$}}
! CHECK-NEXT: <MCOperand Expr:specifier(0,jump2)>
! CHECK-NEXT: <MCOperand Imm:10>

    bt .
! CHECK:      .Ltmp{{[0-9]+}}
! CHECK-NEXT:   bt .Ltmp{{[0-9]+}}
! CHECK:      encoding: [0b1110000A,A,A,A]
! CHECK-NEXT:   fixup A - offset: 0, value: .Ltmp0, kind: FIXUP_LANAI_25
! CHECK-NEXT: <MCInst #{{[0-9]+}} BT{{$}}
! CHECK-NEXT:   <MCOperand Expr:.Ltmp0>

! SCC
    spl %r19
! CHECK: encoding: [0xea,0x4c,0x00,0x02]
! CHECK-NEXT: <MCInst #{{[0-9]+}} SCC{{$}}
! CHECK-NEXT: <MCOperand Reg:26>
! CHECK-NEXT: <MCOperand Imm:10>

! BRR
    bf.r 0x456
! CHECK: encoding: [0xe1,0x00,0x04,0x57]
! CHECK-NEXT: <MCInst #{{[0-9]+}} BRR{{$}}
! CHECK-NEXT: <MCOperand Imm:1110>
! CHECK-NEXT: <MCOperand Imm:1>

! Conditional ALU
  add.ge %r13, %r14, %r18
! CHECK: encoding: [0xc9,0x34,0x70,0x06]
! CHECK-NEXT: <MCInst #{{[0-9]+}} ADD_R
! CHECK-NEXT:  <MCOperand Reg:25>
! CHECK-NEXT:  <MCOperand Reg:20>
! CHECK-NEXT:  <MCOperand Reg:21>
! CHECK-NEXT:  <MCOperand Imm:12>>

  add.f %r13, %r14, %r18
! CHECK: encoding: [0xc9,0x36,0x70,0x00]
! CHECK-NEXT: <MCInst #{{[0-9]+}} ADD_F_R
! CHECK-NEXT:  <MCOperand Reg:25>
! CHECK-NEXT:  <MCOperand Reg:20>
! CHECK-NEXT:  <MCOperand Reg:21>
! CHECK-NEXT:  <MCOperand Imm:0>>
