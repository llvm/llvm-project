# RUN: llvm-mc -triple mblaze-unknown-unknown -show-encoding %s | FileCheck %s

# Test to ensure that all FSL immediate operands and FSL instructions
# can be parsed by the assembly parser correctly.

# TYPE F:   OPCODE RD           NCTAE        FSL
# BINARY:   011011 00000 000000 00000 000000 0000

# TYPE FD:  OPCODE RD          RB      NCTAE
# BINARY:   011011 00000 00000 00000 0 00000 00000

# TYPE FP:  OPCODE       RA      NCTA         FSL
#           000000 00000 00000 1 0000 0000000 0000

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x00,0x00]
            get         r0, rfsl0

# CHECK:    nget
# BINARY:   011011 00000 000000 10000 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x40,0x00]
            nget        r0, rfsl0

# CHECK:    cget
# BINARY:   011011 00000 000000 01000 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x20,0x00]
            cget        r0, rfsl0

# CHECK:    ncget
# BINARY:   011011 00000 000000 11000 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x60,0x00]
            ncget       r0, rfsl0

# CHECK:    tget
# BINARY:   011011 00000 000000 00100 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x10,0x00]
            tget        r0, rfsl0

# CHECK:    tnget
# BINARY:   011011 00000 000000 10100 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x50,0x00]
            tnget       r0, rfsl0

# CHECK:    tcget
# BINARY:   011011 00000 000000 01100 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x30,0x00]
            tcget       r0, rfsl0

# CHECK:    tncget
# BINARY:   011011 00000 000000 11100 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x70,0x00]
            tncget      r0, rfsl0

# CHECK:    aget
# BINARY:   011011 00000 000000 00010 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x08,0x00]
            aget        r0, rfsl0

# CHECK:    naget
# BINARY:   011011 00000 000000 10010 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x48,0x00]
            naget       r0, rfsl0

# CHECK:    caget
# BINARY:   011011 00000 000000 01010 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x28,0x00]
            caget       r0, rfsl0

# CHECK:    ncaget
# BINARY:   011011 00000 000000 11010 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x68,0x00]
            ncaget      r0, rfsl0

# CHECK:    taget
# BINARY:   011011 00000 000000 00110 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x18,0x00]
            taget       r0, rfsl0

# CHECK:    tnaget
# BINARY:   011011 00000 000000 10110 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x58,0x00]
            tnaget      r0, rfsl0

# CHECK:    tcaget
# BINARY:   011011 00000 000000 01110 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x38,0x00]
            tcaget      r0, rfsl0

# CHECK:    tncaget
# BINARY:   011011 00000 000000 11110 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x78,0x00]
            tncaget     r0, rfsl0

# CHECK:    eget
# BINARY:   011011 00000 000000 00001 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x04,0x00]
            eget        r0, rfsl0

# CHECK:    neget
# BINARY:   011011 00000 000000 10001 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x44,0x00]
            neget       r0, rfsl0

# CHECK:    ecget
# BINARY:   011011 00000 000000 01001 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x24,0x00]
            ecget       r0, rfsl0

# CHECK:    necget
# BINARY:   011011 00000 000000 11001 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x64,0x00]
            necget      r0, rfsl0

# CHECK:    teget
# BINARY:   011011 00000 000000 00101 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x14,0x00]
            teget       r0, rfsl0

# CHECK:    tneget
# BINARY:   011011 00000 000000 10101 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x54,0x00]
            tneget      r0, rfsl0

# CHECK:    tecget
# BINARY:   011011 00000 000000 01101 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x34,0x00]
            tecget      r0, rfsl0

# CHECK:    tnecget
# BINARY:   011011 00000 000000 11101 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x74,0x00]
            tnecget     r0, rfsl0

# CHECK:    eaget
# BINARY:   011011 00000 000000 00011 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x0c,0x00]
            eaget       r0, rfsl0

# CHECK:    neaget
# BINARY:   011011 00000 000000 10011 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x4c,0x00]
            neaget      r0, rfsl0

# CHECK:    ecaget
# BINARY:   011011 00000 000000 01011 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x2c,0x00]
            ecaget      r0, rfsl0

# CHECK:    necaget
# BINARY:   011011 00000 000000 11011 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x6c,0x00]
            necaget     r0, rfsl0

# CHECK:    teaget
# BINARY:   011011 00000 000000 00111 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x1c,0x00]
            teaget      r0, rfsl0

# CHECK:    tneaget
# BINARY:   011011 00000 000000 10111 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x5c,0x00]
            tneaget     r0, rfsl0

# CHECK:    tecaget
# BINARY:   011011 00000 000000 01111 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x3c,0x00]
            tecaget     r0, rfsl0

# CHECK:    tnecaget
# BINARY:   011011 00000 000000 11111 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x7c,0x00]
            tnecaget    r0, rfsl0

# CHECK:    getd
# BINARY:   010011 00000 00000 00001 0 00000 00000
# CHECK:    encoding: [0x4c,0x00,0x08,0x00]
            getd        r0, r1

# CHECK:    ngetd
# BINARY:   010011 00000 00000 00001 0 10000 00000
# CHECK:    encoding: [0x4c,0x00,0x0a,0x00]
            ngetd       r0, r1

# CHECK:    cgetd
# BINARY:   010011 00000 00000 00001 0 01000 00000
# CHECK:    encoding: [0x4c,0x00,0x09,0x00]
            cgetd       r0, r1

# CHECK:    ncgetd
# BINARY:   010011 00000 00000 00001 0 11000 00000
# CHECK:    encoding: [0x4c,0x00,0x0b,0x00]
            ncgetd      r0, r1

# CHECK:    tgetd
# BINARY:   010011 00000 00000 00001 0 00100 00000
# CHECK:    encoding: [0x4c,0x00,0x08,0x80]
            tgetd       r0, r1

# CHECK:    tngetd
# BINARY:   010011 00000 00000 00001 0 10100 00000
# CHECK:    encoding: [0x4c,0x00,0x0a,0x80]
            tngetd      r0, r1

# CHECK:    tcgetd
# BINARY:   010011 00000 00000 00001 0 01100 00000
# CHECK:    encoding: [0x4c,0x00,0x09,0x80]
            tcgetd      r0, r1

# CHECK:    tncgetd
# BINARY:   010011 00000 00000 00001 0 11100 00000
# CHECK:    encoding: [0x4c,0x00,0x0b,0x80]
            tncgetd     r0, r1

# CHECK:    agetd
# BINARY:   010011 00000 00000 00001 0 00010 00000
# CHECK:    encoding: [0x4c,0x00,0x08,0x40]
            agetd       r0, r1

# CHECK:    nagetd
# BINARY:   010011 00000 00000 00001 0 10010 00000
# CHECK:    encoding: [0x4c,0x00,0x0a,0x40]
            nagetd      r0, r1

# CHECK:    cagetd
# BINARY:   010011 00000 00000 00001 0 01010 00000
# CHECK:    encoding: [0x4c,0x00,0x09,0x40]
            cagetd     r0, r1

# CHECK:    ncagetd
# BINARY:   010011 00000 00000 00001 0 11010 00000
# CHECK:    encoding: [0x4c,0x00,0x0b,0x40]
            ncagetd     r0, r1

# CHECK:    tagetd
# BINARY:   010011 00000 00000 00001 0 00110 00000
# CHECK:    encoding: [0x4c,0x00,0x08,0xc0]
            tagetd      r0, r1

# CHECK:    tnagetd
# BINARY:   010011 00000 00000 00001 0 10110 00000
# CHECK:    encoding: [0x4c,0x00,0x0a,0xc0]
            tnagetd     r0, r1

# CHECK:    tcagetd
# BINARY:   010011 00000 00000 00001 0 01110 00000
# CHECK:    encoding: [0x4c,0x00,0x09,0xc0]
            tcagetd     r0, r1

# CHECK:    tncagetd
# BINARY:   010011 00000 00000 00001 0 11110 00000
# CHECK:    encoding: [0x4c,0x00,0x0b,0xc0]
            tncagetd    r0, r1

# CHECK:    egetd
# BINARY:   010011 00000 00000 00001 0 00001 00000
# CHECK:    encoding: [0x4c,0x00,0x08,0x20]
            egetd       r0, r1

# CHECK:    negetd
# BINARY:   010011 00000 00000 00001 0 10001 00000
# CHECK:    encoding: [0x4c,0x00,0x0a,0x20]
            negetd      r0, r1

# CHECK:    ecgetd
# BINARY:   010011 00000 00000 00001 0 01001 00000
# CHECK:    encoding: [0x4c,0x00,0x09,0x20]
            ecgetd      r0, r1

# CHECK:    necgetd
# BINARY:   010011 00000 00000 00001 0 11001 00000
# CHECK:    encoding: [0x4c,0x00,0x0b,0x20]
            necgetd     r0, r1

# CHECK:    tegetd
# BINARY:   010011 00000 00000 00001 0 00101 00000
# CHECK:    encoding: [0x4c,0x00,0x08,0xa0]
            tegetd      r0, r1

# CHECK:    tnegetd
# BINARY:   010011 00000 00000 00001 0 10101 00000
# CHECK:    encoding: [0x4c,0x00,0x0a,0xa0]
            tnegetd     r0, r1

# CHECK:    tecgetd
# BINARY:   010011 00000 00000 00001 0 01101 00000
# CHECK:    encoding: [0x4c,0x00,0x09,0xa0]
            tecgetd     r0, r1

# CHECK:    tnecgetd
# BINARY:   010011 00000 00000 00001 0 11101 00000
# CHECK:    encoding: [0x4c,0x00,0x0b,0xa0]
            tnecgetd    r0, r1

# CHECK:    eagetd
# BINARY:   010011 00000 00000 00001 0 00011 00000
# CHECK:    encoding: [0x4c,0x00,0x08,0x60]
            eagetd      r0, r1

# CHECK:    neagetd
# BINARY:   010011 00000 00000 00001 0 10011 00000
# CHECK:    encoding: [0x4c,0x00,0x0a,0x60]
            neagetd     r0, r1

# CHECK:    ecagetd
# BINARY:   010011 00000 00000 00001 0 01011 00000
# CHECK:    encoding: [0x4c,0x00,0x09,0x60]
            ecagetd     r0, r1

# CHECK:    necagetd
# BINARY:   010011 00000 00000 00001 0 11011 00000
# CHECK:    encoding: [0x4c,0x00,0x0b,0x60]
            necagetd    r0, r1

# CHECK:    teagetd
# BINARY:   010011 00000 00000 00001 0 00111 00000
# CHECK:    encoding: [0x4c,0x00,0x08,0xe0]
            teagetd     r0, r1

# CHECK:    tneagetd
# BINARY:   010011 00000 00000 00001 0 10111 00000
# CHECK:    encoding: [0x4c,0x00,0x0a,0xe0]
            tneagetd    r0, r1

# CHECK:    tecagetd
# BINARY:   010011 00000 00000 00001 0 01111 00000
# CHECK:    encoding: [0x4c,0x00,0x09,0xe0]
            tecagetd    r0, r1

# CHECK:    tnecagetd
# BINARY:   010011 00000 00000 00001 0 11111 00000
# CHECK:    encoding: [0x4c,0x00,0x0b,0xe0]
            tnecagetd   r0, r1

# CHECK:    put
# BINARY:   011011 00000 00000 1 0000 0000000 0000
# CHECK:    encoding: [0x6c,0x00,0x80,0x00]
            put         r0, rfsl0

# CHECK:    aput
# BINARY:   011011 00000 00000 1 0001 0000000 0000
# CHECK:    encoding: [0x6c,0x00,0x88,0x00]
            aput        r0, rfsl0

# CHECK:    cput
# BINARY:   011011 00000 00000 1 0100 0000000 0000
# CHECK:    encoding: [0x6c,0x00,0xa0,0x00]
            cput        r0, rfsl0

# CHECK:    caput
# BINARY:   011011 00000 00000 1 0101 0000000 0000
# CHECK:    encoding: [0x6c,0x00,0xa8,0x00]
            caput       r0, rfsl0

# CHECK:    nput
# BINARY:   011011 00000 00000 1 1000 0000000 0000
# CHECK:    encoding: [0x6c,0x00,0xc0,0x00]
            nput        r0, rfsl0

# CHECK:    naput
# BINARY:   011011 00000 00000 1 1001 0000000 0000
# CHECK:    encoding: [0x6c,0x00,0xc8,0x00]
            naput       r0, rfsl0

# CHECK:    ncput
# BINARY:   011011 00000 00000 1 1100 0000000 0000
# CHECK:    encoding: [0x6c,0x00,0xe0,0x00]
            ncput       r0, rfsl0

# CHECK:    ncaput
# BINARY:   011011 00000 00000 1 1101 0000000 0000
# CHECK:    encoding: [0x6c,0x00,0xe8,0x00]
            ncaput      r0, rfsl0

# CHECK:    tput
# BINARY:   011011 00000 00000 1 0010 0000000 0000
# CHECK:    encoding: [0x6c,0x00,0x90,0x00]
            tput        rfsl0

# CHECK:    taput
# BINARY:   011011 00000 00000 1 0011 0000000 0000
# CHECK:    encoding: [0x6c,0x00,0x98,0x00]
            taput       rfsl0

# CHECK:    tcput
# BINARY:   011011 00000 00000 1 0110 0000000 0000
# CHECK:    encoding: [0x6c,0x00,0xb0,0x00]
            tcput       rfsl0

# CHECK:    tcaput
# BINARY:   011011 00000 00000 1 0111 0000000 0000
# CHECK:    encoding: [0x6c,0x00,0xb8,0x00]
            tcaput      rfsl0

# CHECK:    tnput
# BINARY:   011011 00000 00000 1 1010 0000000 0000
# CHECK:    encoding: [0x6c,0x00,0xd0,0x00]
            tnput       rfsl0

# CHECK:    tnaput
# BINARY:   011011 00000 00000 1 1011 0000000 0000
# CHECK:    encoding: [0x6c,0x00,0xd8,0x00]
            tnaput      rfsl0

# CHECK:    tncput
# BINARY:   011011 00000 00000 1 1110 0000000 0000
# CHECK:    encoding: [0x6c,0x00,0xf0,0x00]
            tncput      rfsl0

# CHECK:    tncaput
# BINARY:   011011 00000 00000 1 1111 0000000 0000
# CHECK:    encoding: [0x6c,0x00,0xf8,0x00]
            tncaput     rfsl0

# CHECK:    putd
# BINARY:   010011 00000 00000 00001 1 0000 000000
# CHECK:    encoding: [0x4c,0x00,0x0c,0x00]
            putd        r0, r1

# CHECK:    aputd
# BINARY:   010011 00000 00000 00001 1 0001 000000
# CHECK:    encoding: [0x4c,0x00,0x0c,0x40]
            aputd       r0, r1

# CHECK:    cputd
# BINARY:   010011 00000 00000 00001 1 0100 000000
# CHECK:    encoding: [0x4c,0x00,0x0d,0x00]
            cputd       r0, r1

# CHECK:    caputd
# BINARY:   010011 00000 00000 00001 1 0101 000000
# CHECK:    encoding: [0x4c,0x00,0x0d,0x40]
            caputd      r0, r1

# CHECK:    nputd
# BINARY:   010011 00000 00000 00001 1 1000 000000
# CHECK:    encoding: [0x4c,0x00,0x0e,0x00]
            nputd       r0, r1

# CHECK:    naputd
# BINARY:   010011 00000 00000 00001 1 1001 000000
# CHECK:    encoding: [0x4c,0x00,0x0e,0x40]
            naputd      r0, r1

# CHECK:    ncputd
# BINARY:   010011 00000 00000 00001 1 1100 000000
# CHECK:    encoding: [0x4c,0x00,0x0f,0x00]
            ncputd      r0, r1

# CHECK:    ncaputd
# BINARY:   010011 00000 00000 00001 1 1101 000000
# CHECK:    encoding: [0x4c,0x00,0x0f,0x40]
            ncaputd     r0, r1

# CHECK:    tputd
# BINARY:   010011 00000 00000 00001 1 0010 000000
# CHECK:    encoding: [0x4c,0x00,0x0c,0x80]
            tputd       r1

# CHECK:    taputd
# BINARY:   010011 00000 00000 00001 1 0011 000000
# CHECK:    encoding: [0x4c,0x00,0x0c,0xc0]
            taputd      r1

# CHECK:    tcputd
# BINARY:   010011 00000 00000 00001 1 0110 000000
# CHECK:    encoding: [0x4c,0x00,0x0d,0x80]
            tcputd      r1

# CHECK:    tcaputd
# BINARY:   010011 00000 00000 00001 1 0111 000000
# CHECK:    encoding: [0x4c,0x00,0x0d,0xc0]
            tcaputd     r1

# CHECK:    tnputd
# BINARY:   010011 00000 00000 00001 1 1010 000000
# CHECK:    encoding: [0x4c,0x00,0x0e,0x80]
            tnputd      r1

# CHECK:    tnaputd
# BINARY:   010011 00000 00000 00001 1 1011 000000
# CHECK:    encoding: [0x4c,0x00,0x0e,0xc0]
            tnaputd     r1

# CHECK:    tncputd
# BINARY:   010011 00000 00000 00001 1 1110 000000
# CHECK:    encoding: [0x4c,0x00,0x0f,0x80]
            tncputd     r1

# CHECK:    tncaputd
# BINARY:   010011 00000 00000 00001 1 1111 000000
# CHECK:    encoding: [0x4c,0x00,0x0f,0xc0]
            tncaputd    r1

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 0001
# CHECK:    encoding: [0x6c,0x00,0x00,0x01]
            get     r0, rfsl1

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 0010
# CHECK:    encoding: [0x6c,0x00,0x00,0x02]
            get     r0, rfsl2

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 0011
# CHECK:    encoding: [0x6c,0x00,0x00,0x03]
            get     r0, rfsl3

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 0100
# CHECK:    encoding: [0x6c,0x00,0x00,0x04]
            get     r0, rfsl4

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 0101
# CHECK:    encoding: [0x6c,0x00,0x00,0x05]
            get     r0, rfsl5

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 0110
# CHECK:    encoding: [0x6c,0x00,0x00,0x06]
            get     r0, rfsl6

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 0111
# CHECK:    encoding: [0x6c,0x00,0x00,0x07]
            get     r0, rfsl7

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 1000
# CHECK:    encoding: [0x6c,0x00,0x00,0x08]
            get     r0, rfsl8

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 1001
# CHECK:    encoding: [0x6c,0x00,0x00,0x09]
            get     r0, rfsl9

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 1010
# CHECK:    encoding: [0x6c,0x00,0x00,0x0a]
            get     r0, rfsl10

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 1011
# CHECK:    encoding: [0x6c,0x00,0x00,0x0b]
            get     r0, rfsl11

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 1100
# CHECK:    encoding: [0x6c,0x00,0x00,0x0c]
            get     r0, rfsl12

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 1101
# CHECK:    encoding: [0x6c,0x00,0x00,0x0d]
            get     r0, rfsl13

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 1110
# CHECK:    encoding: [0x6c,0x00,0x00,0x0e]
            get     r0, rfsl14

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 1111
# CHECK:    encoding: [0x6c,0x00,0x00,0x0f]
            get     r0, rfsl15
