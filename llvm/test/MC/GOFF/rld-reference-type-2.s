* RUN: llvm-mc <%s --triple s390x-ibm-zos --filetype=obj -o - | \
* RUN:   od -An -tx1 -v | tr -d ' \n' > %t.hex
* RUN: FileCheck --ignore-case --check-prefix=RLD %s < %t.hex
* REQUIRES: systemz-registered-target
*
* DC J(sym) / DC JD(sym) produces an RLD item with reference type 2
* (RLD_RT_RLength, a J-type length constant) and referent type 0
* (RLD_RO_Label), so f1 == 0x20.
* The symbol sym inside JD(sym) must be a symbol that names an external dummy
* section (DSECT), an external control section, or a class (such as one defined
* with a DXD, DSECT, or class attribute).
* This test covers each such kind of symbol in a module defining the function
* "main".
*
* RLD item layout:
*   byte 0 (f0): 0x80=same-R  0x40=same-P  0x20=same-Offset
*   byte 1 (f1): (ReferenceType<<4) | ReferentType  -> always 20 here
*   byte 2 (f2): (Action<<1) | FetchStore           -> 00 = Add/Fetch
*   byte 4 (f4): target field length in bytes       -> 08 for JD
* followed by the optional R-ESDID, P-ESDID and P-Offset.
*
* ESDIDs, in order of definition:
*    1 SD #C        2 ED C_CODE64     (root SD that owns the ERs)
*    3 SD main#C    4 ED C_CODE64     5 ED C_DATA
*    6 PR main#S    7 ED C_PREG64 (NAMESPACE(PSEUDOREG))
*    8 PR DumLoc    9 LD main        10 ER C_CODE64
*   11 ER C_DATA  12 ER C_PREG64    13 PR ExtDummy
*   14 ER ExtCsect
*
* The P pointer is always ESDID 6 (the main#S pointer section). The items are
* sorted by R-ESDID and then by P-offset, so the P-offsets are not in ascending
* order within the whole list.
*
* Kinds of symbols (offsets are in main#S):
*   A. class (code class)                 C_CODE64             (ER, R=10, offset 0)
*   B. class (WSA class)                  C_DATA              (ER, R=11, offset 8)
*   C. class (pseudo-register class)      C_PREG64             (ER, R=12, offset 16)
*   D. external dummy section             EXTRN PART(ExtDummy) (PR in the pseudo
*                                                               register class,
*                                                               R=13, offset 24)
*   E. local dummy section (DXD/DSECT)    DumLoc               (PR in the pseudo
*                                                               register class,
*                                                               R=8, offset 32)
*   F. external control section           EXTRN ExtCsect       (ER, R=14, offset 40)
*
* The checks start at the first RLD record (0321). Items that straddle two
* physical records contain the 3 byte continuation header (0323/0322) of the
* next record in the middle of the item.
*
* RLD: 0321
* R=8 (DumLoc): case E at 32.
* RLD-SAME: 0020000008000000000000080000000600000020
* R=10 (C_CODE64): case A at 0.
* RLD-SAME: 40200000080000000000000a00000000
* R=11 (C_DATA): case B at 8.
* RLD-SAME: 40200000080000000000000b00000008
* R=12 (C_PREG64): case C at 16.
* RLD-SAME: 40200000080000000000000c00000010
* R=13 (ExtDummy): case D at 24. The item continues in the next physical
* record.
* RLD-SAME: 402000000800
* RLD-SAME: 03220000000000000d00000018
* R=14 (ExtCsect): case F at 40.
* RLD-SAME: 40200000080000000000000e00000028

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
* Case A: class (code class).
         DC JD(C_CODE64)
* Case B: class (WSA class).
         DC JD(C_DATA)
* Case C: class (pseudo-register class).
         DC JD(C_PREG64)
* Case D: external dummy section.
         DC JD(ExtDummy)
* Case E: local dummy section (DXD/DSECT).
         DC JD(DumLoc)
* Case F: external control section.
         DC JD(ExtCsect)

main#C   CSECT
C_PREG64 CATTR ALIGN(3),FILL(0),NOTEXECUTABLE,RMODE(64),NAMESPACE(PSEUDO
               OREG),PART(DumLoc)
DumLoc   XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(EXPORT)
DumLoc   DS 0H
         DC XL4'00000000'
         EXTRN PART(ExtDummy)

main#C   CSECT
C_CODE64 CATTR
         EXTRN ExtCsect
ExtCsect XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
         END
