* RUN: llvm-mc <%s --triple s390x-ibm-zos --filetype=obj -o - | \
* RUN:   od -An -tx1 -v | tr -d ' \n' > %t.hex
* RUN: FileCheck --ignore-case %s < %t.hex
* REQUIRES: systemz-registered-target
*
* Test the DS (define storage) alignment forms: DS 0B, 0H, 0F, 0D, 0Q.
* Each DS pads the current location counter with zero bytes up to the
* requested boundary.  Every case below starts at a misaligned offset, so
* the padding is visible in the TXT record.
*
* Layout of the C_CODE64 section (offset: byte):
*    0: 01                 DC XL1'01'
*    1: 00                 DS 0H pads to offset 2
*    2: 02                 DC XL1'02'
*    3: 00                 DS 0F pads to offset 4
*    4: 03                 DC XL1'03'
*  5-7: 00 00 00           DS 0D pads to offset 8
*    8: 04                 DC XL1'04'
* 9-15: 00 x 7             DS 0Q pads to offset 16
*   16: 05                 DC XL1'05'
*   17: 06                 DS 0B is a no-op; DC XL1'06'
*18-19: 00 00              Lbl DS 0F pads to offset 20
*   20: 07                 DC XL1'07'
*21-24: 00 00 00 00        DC A(Lbl) placeholder (resolved by relocation)
* Total length is 25 bytes (0x19).
*
* CHECK: 03100000000000{{[0-9a-f][0-9a-f]}}000000000000000000000000000000190100020003000000040000000000000005060000070000000000

DSTEST   CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
         DC XL1'01'
         DS 0H
         DC XL1'02'
         DS 0F
         DC XL1'03'
         DS 0D
         DC XL1'04'
         DS 0Q
         DC XL1'05'
         DS 0B
         DC XL1'06'
Lbl      DS 0F
         DC XL1'07'
         DC A(Lbl)
         END
