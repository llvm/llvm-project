* RUN: llvm-mc <%s --triple s390x-ibm-zos --filetype=obj -o - | \
* RUN:   od -An -tx1 -v | tr -d ' \n' > %t.hex
* RUN: FileCheck --ignore-case --check-prefix=RLD %s < %t.hex
* REQUIRES: systemz-registered-target
*
* DC RD(sym) produces an RLD item with reference type 7
* (RLD_RT_RTypeConstant, an R-type constant) and referent type 0
* (RLD_RO_Label), so f1 == 0x70. The item is a Store, so f2 == 0x01.
* The symbol passed to DC RD must be a relocatable symbol that references a
* program section, control section, entry point or function descriptor. Dummy
* sections (pseudo registers) are not valid operands and are not tested here.
* This test uses every kind of valid symbol as the operand of DC RD, in a
* module that defines the function "main". It also covers the R form (4 bytes)
* and an RD nested in AD(...) with an addend.
*
* RLD item layout:
*   byte 0 (f0): 0x80=same-R  0x40=same-P  0x20=same-Offset
*   byte 1 (f1): (ReferenceType<<4) | ReferentType  -> 70 for RD/R, 00 for VD
*   byte 2 (f2): (Action<<1) | FetchStore           -> 01 = Add/Store
*   byte 4 (f4): target field length in bytes       -> 08 for RD, 04 for R
* followed by the optional R-ESDID, P-ESDID and P-Offset.
*
* ESDIDs, in order of definition:
*    1 SD #C        2 ED C_CODE64     (root SD that owns the ERs)
*    3 SD main#C    4 ED C_CODE64     5 ED C_DATA
*    6 PR main#S    7 SD Other#C      8 ED C_CODE64     9 ED C_OTHER
*   10 PR LocPart  11 LD main        12 ER ExtElem     13 PR ExtPart
*   14 ER Other#C
*
* The P pointer is always ESDID 6 (the main#S pointer section). The items are
* sorted by R-ESDID and then by P-offset, so the P-offsets are not in ascending
* order within the whole list.
*
* Kinds of symbols (offsets are in main#S):
*   A. external element        EXTRN ExtElem          (ER, R=12, offset 0)
*   B. external part           EXTRN PART(ExtPart)    (PR, R=13, offset 8)
*   C. local part              LocPart                (PR, R=10, offset 16)
*   D. entry point             main                   (LD, R=11, offset 24)
*   E. R(sym), 4 bytes, of ExtElem at offset 32
*   F. AD(RD(sym)+n) for each of A-D at offsets 36, 44, 52, 60 (f4=8)
*   G. control section         Other#C                (ER, R=14, offset 68)
*   H. function descriptors: RD(sym) followed by VD(sym), for main (RD at
*      offset 76, VD at offset 84) and for ExtElem (RD at offset 92, VD at
*      offset 100). The VD items are V-type constants (reference type 0), so
*      their f1 == 0x00; they are checked because they form the second half
*      of the descriptor and are in the same RLD list.
*
* The checks start at the first RLD record (0321). Items that straddle two
* physical records contain the 3 byte continuation header (0323/0322) of the
* next record in the middle of the item.
*
* R=10 (LocPart): case C at 16, case F at 52 (f4=8).
* RLD: 0321
* RLD-SAME: 00700100080000000000000a0000000600000010
* RLD-SAME: c07001000800000000000034
* R=11 (main): case D at 24, case F at 60, case H RD at 76 and VD at 84. The
* VD item continues in the next physical record.
* RLD-SAME: 40700100080000000000000b00000018
* RLD-SAME: c0700100080000000000003c
* RLD-SAME: c0700100080000000000004c
* RLD-SAME: c000
* RLD-SAME: 03230001000800000000000054
* R=12 (ExtElem): case A at 0, case E (R form, f4=4) at 32, case F at 36, case
* H RD at 92 and VD at 100. The last item continues in the next physical record.
* RLD-SAME: 40700100080000000000000c00000000
* RLD-SAME: c07001000400000000000020
* RLD-SAME: c07001000800000000000024
* RLD-SAME: c0700100080000000000005c
* RLD-SAME: c00001000800000000000064
* R=13 (ExtPart): case B at 8, case F at 44.
* RLD-SAME: 407001
* RLD-SAME: 03220000080000000000000d00000008
* RLD-SAME: c0700100080000000000002c
* R=14 (Other#C): case G at 68.
* RLD-SAME: 40700100080000000000000e00000044

main#C   CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
         ENTRY main
main     XATTR LINKAGE(XPLINK),REFERENCE(CODE),PSECT(main#S),SCOPE(EXPOR
               RT)
main     DS 0H
         stmg 6,7,1840(4)
         aghi 4,-224
         lg 1,0(5)
         lg 6,16(5)
         lg 5,8(5)
         basr 7,6
         bcr 0,0
         lg 7,2072(4)
         aghi 4,224
         b 2(7)

main#C   CSECT
C_DATA   CATTR ALIGN(4),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(mai
               in#S)
main#S   XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
* Case A: external element.
         DC RD(ExtElem)
* Case B: external part.
         DC RD(ExtPart)
* Case C: local part.
         DC RD(LocPart)
* Case D: entry point (label in code).
         DC RD(main)
* Case E: R form, 4 bytes.
         DC R(ExtElem)
* Case F: RD nested in AD with an addend.
         DC AD(RD(ExtElem)+16)
         DC AD(RD(ExtPart)+8)
         DC AD(RD(LocPart)+4)
         DC AD(RD(main)+2)
* Case G: control section.
         DC RD(Other#C)
* Case H: function descriptors (RD of the function followed by VD of it).
         DC RD(main)
         DC VD(main)
         DC RD(ExtElem)
         DC VD(ExtElem)

Other#C  CSECT
C_OTHER  CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
         bcr 0,0

main#C   CSECT
C_DATA   CATTR ALIGN(3),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(Loc
               cPart)
LocPart  XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(EXPORT)
LocPart  DS 0H
         DC XL4'00000000'

main#C   CSECT
C_DATA   CATTR PART(main#S)
         EXTRN PART(ExtPart)

main#C   CSECT
C_CODE64 CATTR
         EXTRN ExtElem
ExtElem  XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
         END
