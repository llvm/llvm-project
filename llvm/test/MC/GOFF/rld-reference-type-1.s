* RUN: llvm-mc <%s --triple s390x-ibm-zos --filetype=obj -o - | \
* RUN:   od -An -tx1 -v | tr -d ' \n' > %t.hex
* RUN: FileCheck --ignore-case --check-prefix=RLD %s < %t.hex
* REQUIRES: systemz-registered-target
*
* DC QD(sym) produces an RLD item with reference type 1 (RLD_RT_ROffset, a
* class-relative offset) and referent type 2 (RLD_RO_Class), so f1 == 0x12.
* This test uses every kind of symbol that can be the operand of DC QD, in a
* module that defines the function "main". It also covers the QL<n> length
* forms and a QD nested in AD(...) with an addend.
*
* RLD item layout:
*   byte 0 (f0): 0x80=same-R  0x40=same-P  0x20=same-Offset
*   byte 1 (f1): (ReferenceType<<4) | ReferentType  -> always 12 here
*   byte 2 (f2): (Action<<1) | FetchStore           -> 00 = Add/Fetch
*   byte 4 (f4): target field length in bytes       -> 08 for QD, n for QL<n>
* followed by the optional R-ESDID, P-ESDID and P-Offset.
*
* ESDIDs, in order of definition:
*    1 SD #C        2 ED C_CODE64     (root SD that owns the ERs)
*    3 SD main#C    4 ED C_CODE64     5 ED C_DATA
*    6 PR main#S    7 PR LocPart      8 ED C_PREG64 (NAMESPACE(PSEUDOREG))
*    9 PR DumLoc   10 LD main        11 ER ExtElem
*   12 PR ExtPart  13 PR ExtDummy
*
* The P pointer is always ESDID 6 (the main#S pointer section). The items are
* sorted by R-ESDID and then by P-offset, so the P-offsets are not in ascending
* order within the whole list.
*
* Kinds of symbols (offsets are in main#S):
*   A. external element        EXTRN ExtElem          (ER, R=11, offset 0)
*   B. external part           EXTRN PART(ExtPart)    (PR, R=12, offset 8)
*   C. external dummy section  EXTRN PART(ExtDummy)   (PR in the pseudo
*                                                      register class, R=13,
*                                                      offset 16)
*   D. local part              LocPart                (PR, R=7, offset 24)
*   E. local dummy section     DumLoc                 (PR in the pseudo
*                                                      register class, R=9,
*                                                      offset 32)
*   F. label in code           main                   (LD, R=10, offset 40)
*   G. QL1, QL2, QL4, QL8 of ExtElem at offsets 48, 49, 51, 55 (f4 = 1,2,4,8)
*   H. AD(QD(sym)+n) for each of A-F at offsets 63, 71, 79, 87, 95, 103 (f4=8)
*
* The checks start at the first RLD record (0321). Items that straddle two
* physical records contain the 3 byte continuation header (0323/0322) of the
* next record in the middle of the item.
*
* RLD: 0321
* R=7 (LocPart): case D at 24, case H at 87.
* RLD-SAME: 0012000008000000000000070000000600
* RLD-SAME: 000018
* RLD-SAME: c01200000800000000000057
* R=9 (DumLoc): case E at 32, case H at 95.
* RLD-SAME: 40120000080000000000000900000020
* RLD-SAME: c0120000080000000000005f
* R=10 (main): case F at 40, case H at 103. The second item continues in the
* next physical record.
* RLD-SAME: 40120000080000000000000a0000
* RLD-SAME: 0323000028
* RLD-SAME: c01200000800000000000067
* R=11 (ExtElem): case A at 0, case G at 48, 49, 51, 55, case H at 63.
* RLD-SAME: 40120000080000000000000b00000000
* RLD-SAME: c01200000100000000000030
* RLD-SAME: c01200000200000000000031
* RLD-SAME: c01200000400000000000033
* RLD-SAME: c012000008000000000000
* RLD-SAME: 03220037
* RLD-SAME: c0120000080000000000003f
* R=12 (ExtPart): case B at 8, case H at 71.
* RLD-SAME: 40120000080000000000000c00000008
* RLD-SAME: c01200000800000000000047
* R=13 (ExtDummy): case C at 16, case H at 79.
* RLD-SAME: 40120000080000000000000d00000010
* RLD-SAME: c0120000080000000000004f

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
         DC QD(ExtElem)
* Case B: external part.
         DC QD(ExtPart)
* Case C: external dummy section.
         DC QD(ExtDummy)
* Case D: local part.
         DC QD(LocPart)
* Case E: local dummy section.
         DC QD(DumLoc)
* Case F: label in code.
         DC QD(main)
* Case G: QL<n> with an explicit length, all on the external element.
* Offsets 48 (1 byte), 49 (2 bytes), 51 (4 bytes), 55 (8 bytes).
         DC QL1(ExtElem)
         DC QL2(ExtElem)
         DC QL4(ExtElem)
         DC QL8(ExtElem)
* Case H: QD nested in AD with an addend. The addend is kept in the TXT bytes
* and every symbol kind gets the same f1 == 0x12 item with f4 == 08.
* Offsets 63, 71, 79, 87, 95, 103.
         DC AD(QD(ExtElem)+16)
         DC AD(QD(ExtPart)+8)
         DC AD(QD(ExtDummy)-4)
         DC AD(QD(LocPart)+4)
         DC AD(QD(DumLoc)+XL8'10')
         DC AD(QD(main)+2)

main#C   CSECT
C_DATA   CATTR ALIGN(3),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(Loc
               cPart)
LocPart  XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(EXPORT)
LocPart  DS 0H
         DC XL4'00000000'

main#C   CSECT
C_PREG64 CATTR ALIGN(3),FILL(0),NOTEXECUTABLE,RMODE(64),NAMESPACE(PSEUDO
               OREG),PART(DumLoc)
DumLoc   XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(EXPORT)
DumLoc   DS 0H
         DC XL4'00000000'
         EXTRN PART(ExtDummy)

main#C   CSECT
C_DATA   CATTR PART(main#S)
         EXTRN PART(ExtPart)

main#C   CSECT
C_CODE64 CATTR
         EXTRN ExtElem
ExtElem  XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
         END
