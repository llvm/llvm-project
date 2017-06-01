! RUN: llvm-mc -arch=lanai -show-encoding -show-inst < %s | FileCheck %s

! Checking the machine instructions generated from ASM instructions for ALU
! operations.

! RM class
    ld [%r7], %r6
! CHECK: encoding: [0x83,0x1c,0x00,0x00]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDW_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:13>
! CHECK-NEXT: <MCOperand Reg:14>
! CHECK-NEXT: <MCOperand Imm:0>
! CHECK-NEXT: <MCOperand Imm:0>

    ld [%r6], %r6
! CHECK: encoding: [0x83,0x18,0x00,0x00]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDW_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:13>
! CHECK-NEXT: <MCOperand Reg:13>
! CHECK-NEXT: <MCOperand Imm:0>
! CHECK-NEXT: <MCOperand Imm:0>

     st %r6, [%r7]
! CHECK: encoding: [0x93,0x1c,0x00,0x00]
! CHECK-NEXT: <MCInst #{{[0-9]+}} SW_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:13>
! CHECK-NEXT: <MCOperand Reg:14>
! CHECK-NEXT: <MCOperand Imm:0>
! CHECK-NEXT: <MCOperand Imm:0>

    ld 0x123[%r7*], %r6
! CHECK: encoding: [0x83,0x1d,0x01,0x23]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDW_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:13>
! CHECK-NEXT: <MCOperand Reg:14>
! CHECK-NEXT: <MCOperand Imm:291>
! CHECK-NEXT: <MCOperand Imm:128>

    ld [%r7--], %r6
! CHECK: encoding: [0x83,0x1d,0xff,0xfc]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDW_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:13>
! CHECK-NEXT: <MCOperand Reg:14>
! CHECK-NEXT: <MCOperand Imm:-4>
! CHECK-NEXT: <MCOperand Imm:128>

    ld 0x123[%r7], %r6
! CHECK: encoding: [0x83,0x1e,0x01,0x23]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDW_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:13>
! CHECK-NEXT: <MCOperand Reg:14>
! CHECK-NEXT: <MCOperand Imm:291>
! CHECK-NEXT: <MCOperand Imm:0>

    ld 0x123[*%r7], %r6
! CHECK: encoding: [0x83,0x1f,0x01,0x23]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDW_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:13>
! CHECK-NEXT: <MCOperand Reg:14>
! CHECK-NEXT: <MCOperand Imm:291>
! CHECK-NEXT: <MCOperand Imm:64>

    ld [--%r7], %r6
! CHECK: encoding: [0x83,0x1f,0xff,0xfc]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDW_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:13>
! CHECK-NEXT: <MCOperand Reg:14>
! CHECK-NEXT: <MCOperand Imm:-4>
! CHECK-NEXT: <MCOperand Imm:64>

    st %r6, [%r7++]
! CHECK: encoding: [0x93,0x1d,0x00,0x04]
! CHECK-NEXT: <MCInst #{{[0-9]+}} SW_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:13>
! CHECK-NEXT: <MCOperand Reg:14>
! CHECK-NEXT: <MCOperand Imm:4>
! CHECK-NEXT: <MCOperand Imm:128>

    st.h %r6, [%r7++]
! CHECK: encoding: [0xf3,0x1f,0x24,0x02]
! CHECK-NEXT: <MCInst #{{[0-9]+}} STH_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:13>
! CHECK-NEXT: <MCOperand Reg:14>
! CHECK-NEXT: <MCOperand Imm:2>
! CHECK-NEXT: <MCOperand Imm:128>>

    ld.b [--%r7], %r6
! CHECK: encoding: [0xf3,0x1f,0x4f,0xff]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDBs_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:13>
! CHECK-NEXT: <MCOperand Reg:14>
! CHECK-NEXT: <MCOperand Imm:-1>
! CHECK-NEXT: <MCOperand Imm:64>>

! Largest RM value before SLS encoding is used
    ld [0x7fff], %r7
! CHECK: encoding: [0x83,0x82,0x7f,0xff]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDW_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:14>
! CHECK-NEXT: <MCOperand Reg:7>
! CHECK-NEXT: <MCOperand Imm:32767>
! CHECK-NEXT: <MCOperand Imm:0>

    ld [0x8000], %r7
! CHECK: encoding: [0xf3,0x80,0x80,0x00]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDADDR{{$}}
! CHECK-NEXT: <MCOperand Reg:14>
! CHECK-NEXT: <MCOperand Imm:32768>

! Negative RM value
    ld [0xfffffe8c], %pc
! CHECK: encoding: [0x81,0x02,0xfe,0x8c]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDW_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:2>
! CHECK-NEXT: <MCOperand Reg:7>
! CHECK-NEXT: <MCOperand Imm:-372>
! CHECK-NEXT: <MCOperand Imm:0>

    ld [-372], %pc
! CHECK: encoding: [0x81,0x02,0xfe,0x8c]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDW_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:2>
! CHECK-NEXT: <MCOperand Reg:7>
! CHECK-NEXT: <MCOperand Imm:-372>
! CHECK-NEXT: <MCOperand Imm:0>

! RRM class
    ld %r9[%r12*], %r20
! CHECK: encoding: [0xaa,0x31,0x48,0x02]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDW_RR{{$}}
! CHECK-NEXT: <MCOperand Reg:27>
! CHECK-NEXT: <MCOperand Reg:19>
! CHECK-NEXT: <MCOperand Reg:16>
! CHECK-NEXT: <MCOperand Imm:128>

    ld %r9[%r12], %r20
! CHECK: encoding: [0xaa,0x32,0x48,0x02]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDW_RR{{$}}
! CHECK-NEXT: <MCOperand Reg:27>
! CHECK-NEXT: <MCOperand Reg:19>
! CHECK-NEXT: <MCOperand Reg:16>
! CHECK-NEXT: <MCOperand Imm:0>

    ld [%r12 sub %r9], %r20
! CHECK: encoding: [0xaa,0x32,0x4a,0x02]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDW_RR{{$}}
! CHECK-NEXT: <MCOperand Reg:27>
! CHECK-NEXT: <MCOperand Reg:19>
! CHECK-NEXT: <MCOperand Reg:16>
! CHECK-NEXT: <MCOperand Imm:2>

    ld %r9[*%r12], %r20
! CHECK: encoding: [0xaa,0x33,0x48,0x02]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDW_RR{{$}}
! CHECK-NEXT: <MCOperand Reg:27>
! CHECK-NEXT: <MCOperand Reg:19>
! CHECK-NEXT: <MCOperand Reg:16>
! CHECK-NEXT: <MCOperand Imm:64>

    st %r20, %r9[*%r12]
! CHECK: encoding: [0xba,0x33,0x48,0x02]
! CHECK-NEXT: <MCInst #{{[0-9]+}} SW_RR{{$}}
! CHECK-NEXT: <MCOperand Reg:27>
! CHECK-NEXT: <MCOperand Reg:19>
! CHECK-NEXT: <MCOperand Reg:16>
! CHECK-NEXT: <MCOperand Imm:64>

    ld.b [%r12 sub %r9], %r20
! CHECK: encoding: [0xaa,0x32,0x4a,0x04]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDBs_RR{{$}}
! CHECK-NEXT: <MCOperand Reg:27>
! CHECK-NEXT: <MCOperand Reg:19>
! CHECK-NEXT: <MCOperand Reg:16>
! CHECK-NEXT: <MCOperand Imm:2>

    uld.h [%r12 sub %r9], %r20
! CHECK: encoding: [0xaa,0x32,0x4a,0x01]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDHz_RR{{$}}
! CHECK-NEXT: <MCOperand Reg:27>
! CHECK-NEXT: <MCOperand Reg:19>
! CHECK-NEXT: <MCOperand Reg:16>
! CHECK-NEXT: <MCOperand Imm:2>


! SPLS class
    st.b %r3, [%r6]
! CHECK: encoding: [0xf1,0x9b,0x60,0x00]
! CHECK-NEXT: <MCInst #{{[0-9]+}} STB_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:10>
! CHECK-NEXT: <MCOperand Reg:13>
! CHECK-NEXT: <MCOperand Imm:0>
! CHECK-NEXT: <MCOperand Imm:0>

    st.b %r3, 1[%r6*]
! CHECK: encoding: [0xf1,0x9b,0x64,0x01]
! CHECK-NEXT: <MCInst #{{[0-9]+}} STB_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:10>
! CHECK-NEXT: <MCOperand Reg:13>
! CHECK-NEXT: <MCOperand Imm:1>
! CHECK-NEXT: <MCOperand Imm:128>

    st.b %r3, 1[%r6]
! CHECK: encoding: [0xf1,0x9b,0x68,0x01]
! CHECK-NEXT: <MCInst #{{[0-9]+}} STB_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:10>
! CHECK-NEXT: <MCOperand Reg:13>
! CHECK-NEXT: <MCOperand Imm:1>
! CHECK-NEXT: <MCOperand Imm:0>

    st.b %r3, 1[*%r6]
! CHECK: encoding: [0xf1,0x9b,0x6c,0x01]
! CHECK-NEXT: <MCInst #{{[0-9]+}} STB_RI{{$}}
! CHECK-NEXT: <MCOperand Reg:10>
! CHECK-NEXT: <MCOperand Reg:13>
! CHECK-NEXT: <MCOperand Imm:1>
! CHECK-NEXT: <MCOperand Imm:64>

! SLS class
    st %r30, [0x1234]
! CHECK: encoding: [0xff,0x01,0x12,0x34]
! CHECK-NEXT: <MCInst #{{[0-9]+}} STADDR{{$}}
! CHECK-NEXT: <MCOperand Reg:37>
! CHECK-NEXT: <MCOperand Imm:4660>

    ld [0xfe8c], %pc
! CHECK: encoding: [0xf1,0x00,0xfe,0x8c]
! CHECK-NEXT: <MCInst #{{[0-9]+}} LDADDR{{$}}
! CHECK-NEXT: <MCOperand Reg:2>
! CHECK-NEXT: <MCOperand Imm:65164>

! SLI class
    mov hi(x), %r4
! CHECK: encoding: [0x02,0x01,A,A]
! CHECK-NEXT: fixup A - offset: 0, value: hi(x), kind: FIXUP_LANAI_HI16{{$}}
! CHECK-NEXT: <MCInst #{{[0-9]+}} ADD_I_HI
! CHECK-NEXT: <MCOperand Reg:11>
! CHECK-NEXT: <MCOperand Reg:7>
! CHECK-NEXT: <MCOperand Expr:(hi(x))>

    mov hi(l+4), %r7
! CHECK: encoding: [0x03,0x81,A,A]
! CHECK-NEXT: fixup A - offset: 0, value: (hi(l))+4, kind: FIXUP_LANAI_HI16{{$}}
! CHECK-NEXT: <MCInst #{{[0-9]+}} ADD_I_HI
! CHECK-NEXT: <MCOperand Reg:14>
! CHECK-NEXT: <MCOperand Reg:7>
! CHECK-NEXT: <MCOperand Expr:((hi(l))+4)>

