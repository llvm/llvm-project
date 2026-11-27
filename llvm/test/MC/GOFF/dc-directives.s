* RUN: llvm-mc <%s --triple s390x-ibm-zos --filetype=obj -o - | \
* RUN:   od -An -tx1 -v | tr -d ' \n' > %t.hex
* RUN: FileCheck --ignore-case --check-prefix=TXT %s < %t.hex
* RUN: FileCheck --ignore-case --check-prefix=RLD %s < %t.hex
* REQUIRES: systemz-registered-target
*
* Comprehensive DC directive tests: XL<n>, X, AL<n>, AD, A, VD, V, RD, QD,
* and multi-operand combinations.
*
* Two separate FileCheck passes share the same assembled object:
*   TXT pass - checks data bytes emitted into TXT (text) records, in order.
*   RLD pass - checks relocation flag bytes in the combined RLD record.
*
* Each physical GOFF record is exactly 80 bytes.  TXT records start with
* 0310 (byte[1]=0x10 = record-type 1).  Bytes [2:4]=0x0000, bytes [4:8]
* contain the ESDID, bytes [8:22] are zero, bytes [22:24] hold the data
* length, and bytes [24:] hold the data.  The TXT patterns below embed the
* ESDID so each line uniquely identifies its record.
*
* RLD items are packed into one logical RLD record spanning three physical
* records.  The first physical record starts with 0321 (RecContinued bit set).
* Each item has an 8-byte fixed header:
*   byte 0 (f0): 0x80=same-R  0x40=same-P  0x20=same-Offset
*   byte 1 (f1): (ReferenceType<<4) | ReferentType
*   byte 2 (f2): (Action<<1) | FetchStore  (VCon/RCon set bit 0 = Store = 1)
*   byte 4 (f4): target field length in bytes
* followed by optional 4-byte R-ESDID, 4-byte P-ESDID, 4-byte P-Offset.

* TXT: 03f0

*──────────────────────────────────────────────────────────────────────────────
* TEST 1 - DC XL explicit-length hex byte literals
*   XL1'AB'               -> AB
*   XL2'CAFE'             -> CA FE
*   XL4'DEADBEEF'         -> DE AD BE EF
*   XL8'0102030405060708' -> 01 02 03 04 05 06 07 08
*   X'FF'                 -> FF
* ESDID=0x0004  datalen=0x0010 (16 bytes)
*──────────────────────────────────────────────────────────────────────────────
XL1TEST  CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
         DC XL1'AB'
         DC XL2'CAFE'
         DC XL4'DEADBEEF'
         DC XL8'0102030405060708'
         DC X'FF'
* TXT: 031000000000000400000000000000000000000000000010abcafedeadbeef0102030405060708ff

*──────────────────────────────────────────────────────────────────────────────
* TEST 2 - DC XL left-zero-padding
*   XL4'FF'  -> 00 00 00 FF  (padded left to 4 bytes)
*   XL3'1'   -> 00 00 01     (padded left to 3 bytes)
* ESDID=0x0006  datalen=0x0007 (7 bytes)
*──────────────────────────────────────────────────────────────────────────────
XL2TEST  CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
         DC XL4'FF'
         DC XL3'1'
* TXT: 031000000000000600000000000000000000000000000007000000ff000001

*──────────────────────────────────────────────────────────────────────────────
* TEST 3 - DC AL absolute literals (no relocation)
*   AL1(42)   -> 2A
*   AL2(1000) -> 03 E8
*   AL4(256)  -> 00 00 01 00
* ESDID=0x000a  datalen=0x0007 (7 bytes)
*──────────────────────────────────────────────────────────────────────────────
AL1TEST  CSECT
C_WSA64  CATTR ALIGN(3),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(AL1
               1TEST#S)
AL1TEST#S XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
         DC AL1(42)
         DC AL2(1000)
         DC AL4(256)
* TXT: 031000000000000a000000000000000000000000000000072a03e800000100

*──────────────────────────────────────────────────────────────────────────────
* TEST 4 - DC A(sym) - 4-byte A-type address constant
* A(SymAD) emits 4 zero placeholder bytes replaced by relocation.
* ESDID=0x000e  datalen=0x0004
*──────────────────────────────────────────────────────────────────────────────
ADTEST   CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
SymAD    DS 0H
ADTEST   CSECT
C_DATA   CATTR ALIGN(3),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(ADT
               TEST#S)
ADTEST#S XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
         DC A(SymAD)
* TXT: 031000000000000e0000000000000000000000000000000400000000

*──────────────────────────────────────────────────────────────────────────────
* TEST 5 - DC A(sym) - short form of AD (identical 4-byte output)
* ESDID=0x0012  datalen=0x0004
*──────────────────────────────────────────────────────────────────────────────
ATEST    CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
SymA     DS 0H
ATEST    CSECT
C_DATA   CATTR ALIGN(3),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(ATE
               EST#S)
ATEST#S  XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
         DC A(SymA)
* TXT: 03100000000000120000000000000000000000000000000400000000

*──────────────────────────────────────────────────────────────────────────────
* TEST 6 - DC A(sym+n) and DC A(sym-n): addend baked into TXT
*   AD(SymAE+4) -> TXT word = 00000004
*   AD(SymAE-1) -> TXT word = FFFFFFFF
* The C_CODE64 sub-section also emits DC XL4'00000000' (4 zero bytes) at
* ESDID=0x0014 before the C_WSA64 pointer section at ESDID=0x0016.
*──────────────────────────────────────────────────────────────────────────────
ADEXPR   CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
SymAE    DS 0H
         DC XL4'00000000'
ADEXPR   CSECT
C_DATA   CATTR ALIGN(3),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(ADE
               EXPR#S)
ADEXPR#S XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
         DC A(SymAE+4)
         DC A(SymAE-1)
* TXT: 03100000000000140000000000000000000000000000000400000000
* TXT: 03100000000000160000000000000000000000000000000800000004ffffffff

*──────────────────────────────────────────────────────────────────────────────
* TEST 7 - DC VD(sym) - 8-byte V-type external address
* VD(ExtVD) emits 8 zero placeholder bytes replaced by relocation.
* ESDID=0x001a  datalen=0x0008
*──────────────────────────────────────────────────────────────────────────────
VDTEST   CSECT
C_WSA64  CATTR ALIGN(3),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(VDT
               TEST#S)
VDTEST#S XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
         EXTRN ExtVD
         DC VD(ExtVD)
* TXT: 031000000000001a000000000000000000000000000000080000000000000000

*──────────────────────────────────────────────────────────────────────────────
* TEST 8 - DC V(sym) - short 4-byte form of VD
* ESDID=0x001e  datalen=0x0004
*──────────────────────────────────────────────────────────────────────────────
VTEST    CSECT
C_DATA   CATTR ALIGN(3),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(VTE
               EST#S)
VTEST#S  XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
         EXTRN ExtV
         DC V(ExtV)
* TXT: 031000000000001e0000000000000000000000000000000400000000

*──────────────────────────────────────────────────────────────────────────────
* TEST 9 - DC RD(sym) - 8-byte R-type PSECT address
* ESDID=0x0022  datalen=0x0008
*──────────────────────────────────────────────────────────────────────────────
RDTEST   CSECT
C_WSA64  CATTR ALIGN(3),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(RDT
               TEST#S)
RDTEST#S XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
         EXTRN ExtRD
         DC RD(ExtRD)
* TXT: 0310000000000022000000000000000000000000000000080000000000000000

*──────────────────────────────────────────────────────────────────────────────
* TEST 10 - DC QD(sym) - 8-byte Q-type class-relative offset
* ESDID=0x0026  datalen=0x0008
*──────────────────────────────────────────────────────────────────────────────
QDTEST   CSECT
C_WSA64  CATTR ALIGN(3),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(QDT
               TEST#S)
QDTEST#S XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
         EXTRN ExtQD
         DC QD(ExtQD)
* TXT: 0310000000000026000000000000000000000000000000080000000000000000

*──────────────────────────────────────────────────────────────────────────────
* TEST 11 - Multi-operand: DC XL1'AB',A(sym),XL1'CD'
*   TXT layout: AB 00000000 CD  (6 bytes)
* ESDID=0x002a  datalen=0x0006
*──────────────────────────────────────────────────────────────────────────────
COMBO1   CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
SymCB1   DS 0H
COMBO1   CSECT
C_DATA   CATTR ALIGN(3),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(COM
               MBO1#S)
COMBO1#S XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
         DC XL1'AB',A(SymCB1),XL1'CD'
* TXT: 031000000000002a00000000000000000000000000000006ab00000000cd

*──────────────────────────────────────────────────────────────────────────────
* TEST 12 - Multi-operand: DC A(ExtA),VD(ExtB)
*   TXT layout: 00000000 0000000000000000  (12 bytes)
* ESDID=0x002e  datalen=0x000c
*──────────────────────────────────────────────────────────────────────────────
COMBO2   CSECT
C_DATA   CATTR ALIGN(3),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(COM
               MBO2#S)
COMBO2#S XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
         EXTRN ExtA
         EXTRN ExtB
         DC A(ExtA),VD(ExtB)
* TXT: 031000000000002e0000000000000000000000000000000c000000000000000000000000

*──────────────────────────────────────────────────────────────────────────────
* TEST 13 - DC A(Off8) where Off8 is defined via EQU
*   Off8 EQU SymEQ+8  -> A(Off8) emits the addend 8 in the TXT word; the
*   relocation refers to SymEQ.
*   The C_CODE64 sub-section emits DC XL8'0000000000000000' at ESDID=0x0030.
*   The C_WSA64 pointer section emits A(Off8) at ESDID=0x0032.
*──────────────────────────────────────────────────────────────────────────────
EQUTEST  CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
SymEQ    DS 0H
         DC XL8'0000000000000000'
Off8     EQU SymEQ+8
EQUTEST  CSECT
C_DATA   CATTR ALIGN(3),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(EQU
               UTEST#S)
EQUTEST#S XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
         DC A(Off8)
* TXT: 0310000000000030000000000000000000000000000000080000000000000000
* TXT: 03100000000000320000000000000000000000000000000400000008

*──────────────────────────────────────────────────────────────────────────────
* TEST 14 - Comma-separated DC XL1 on one line
*   DC XL1'01',XL1'02',XL1'03' -> 01 02 03  (3 bytes)
* ESDID=0x0034  datalen=0x0003
*──────────────────────────────────────────────────────────────────────────────
MULTIDC  CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
         DC XL1'01',XL1'02',XL1'03'
* TXT: 031000000000003400000000000000000000000000000003010203

*──────────────────────────────────────────────────────────────────────────────
* TEST 15 - Duplication factor: DC 3XL1'00' -> 00 00 00 (3 bytes)
* ESDID=0x0036  datalen=0x0003
*──────────────────────────────────────────────────────────────────────────────
DUPDC    CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
         DC 3XL1'00'
* TXT: 031000000000003600000000000000000000000000000003000000

         END

*==============================================================================
* RLD checks
*
* All relocations are packed into one logical RLD record spanning three
* physical GOFF records with continuation flags.  The raw hex stream includes
* the physical record prefix bytes 0321/0323/0322 interspersed within the
* logical payload.  The patterns below are written against the raw hex and
* therefore include those prefix bytes when an item spans a record boundary.
*
* Record boundaries in the logical payload:
*   Record 1 (0321): logical payload bytes 0-73   (74 bytes, raw chars 12-159)
*   Record 2 (0323): logical payload bytes 74-150 (77 bytes, raw chars 166-319)
*   Record 3 (0322): logical payload bytes 151+   (raw chars 326+)
*
* Item header (8 bytes): f0(flags) f1(reftype/refttype) f2(action/fetchstore)
*   f3(reserved) f4(target_len) f5(reserved) f6-f7(reserved uint16=0).
* Then: optional R-ESDID(4), P-ESDID(4), P-Offset(4).
* FetchStore bit (f2 bit 0): 0=Fetch(ACon/QCon), 1=Store(VCon/RCon).
*
* Item 1: A(SymAD)  f0=0x00  f2=0x00  f4=0x04
*   R=0x37  P=0x0e  off=0x00000000
* RLD: 0321
* RLD: 0000000004000000000000370000000e00000000
*
* Item 2: A(SymA)  f0=0x20(same-off)  f2=0x00  f4=0x04
*   R=0x38  P=0x12
* RLD: 20000000040000000000003800000012
*
* Item 3: A(SymAE+4)  f0=0x20  f2=0x00  f4=0x04
*   R=0x39  P=0x16
* RLD: 20000000040000000000003900000016
*
* Item 4: A(SymAE-1)  f0=0xc0(same-R+same-P)  f2=0x00  f4=0x04
*   off=0x00000004
* RLD: c00000000400000000000004
*
* Item 5: VD(ExtVD)  f0=0x00  f2=0x01(Store)  f4=0x08
*   R=0x3a  P=0x1a  off=0x00000000
*   (spans rec1/rec2 boundary; rec2 header 032300 appears between bytes 9-10)
* RLD: 00000100080000000000032300003a0000001a00000000
*
* Item 6: V(ExtV)  f0=0x20  f2=0x01  f4=0x04
*   R=0x3b  P=0x1e
* RLD: 20000100040000000000003b0000001e
*
* Item 7: RD(ExtRD)  f0=0x20  f1=0x70(RCon)  f2=0x01  f4=0x08
*   R=0x3c  P=0x22
* RLD: 20700100080000000000003c00000022
*
* Item 8: QD(ExtQD)  f0=0x20  f1=0x12(QCon+Class)  f2=0x00  f4=0x08
*   R=0x3d  P=0x26
* RLD: 20120000080000000000003d00000026
*
* Item 9: A(SymCB1)  f0=0x00  f2=0x00  f4=0x04
*   R=0x3e  P=0x2a  off=0x00000001
*   (spans rec2/rec3 boundary; rec3 header 032200 appears between bytes 18-19)
* RLD: 00000000040000000000003e0000002a00000003220001
*
* Item 10: A(ExtA)  f0=0x00  f2=0x00  f4=0x04
*   R=0x3f  P=0x2e  off=0x00000000
* RLD: 00000000040000000000003f0000002e00000000
*
* Item 11: VD(ExtB)  f0=0x40(same-P)  f2=0x01  f4=0x08
*   R=0x40  off=0x00000004
* RLD: 40000100080000000000004000000004
