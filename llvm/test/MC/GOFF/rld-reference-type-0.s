* RUN: llvm-mc <%s --triple s390x-ibm-zos --filetype=obj -o - | \
* RUN:   od -An -tx1 -v | tr -d ' \n' > %t.hex
* RUN: FileCheck --ignore-case --check-prefix=RLD %s < %t.hex
* REQUIRES: systemz-registered-target
*
* Every kind of relocation that produces an RLD item with reference type 0
* (RLD_RT_RAddress, an A-type constant) and referent type 0 (RLD_RO_Label),
* in a module that defines the function "main".
*
* RLD item layout (all items below have f1 == 00, i.e. ReferenceType == 0 and
* ReferentType == 0; f4 == 04 is the field length in bytes):
*   byte 0 (f0): 0x80=same-R  0x40=same-P  0x20=same-Offset
*   byte 1 (f1): (ReferenceType<<4) | ReferentType
*   byte 2 (f2): (Action<<1) | FetchStore   00 = Add/Fetch, 02 = Subtract/Fetch
*   byte 4 (f4): target field length
* followed by the optional R-ESDID, P-ESDID and P-Offset.
*
* Cases (P = pointer section ESDID, R = referenced ESDID):
*   A. AD(sym1-sym2) in code      : Subtract sym2 + Add sym1 (two items)
*   B. AD(sym2-sym1) in a PPA2 pointer section (same, other direction)
*   C. A(ext)                     : external symbol
*   D. AD(ext)                    : external symbol
*   E. AD(sym)                    : label in another section
*   F. AD(sym+4), AD(sym-1)       : addend is kept in the TXT bytes
*   G. AD(ext1-ext2)              : Add ext1 + Subtract ext2
*   H. AD(EquSym), EQU sym+8
*   I. XL1,A(ext),XL1            : unaligned ACon in a multi operand DC
*   J. AD(ext1),A(ext2)          : consecutive ACons
*   K. AD(data) for ADA slots     : pointers to external data symbols
* A same-section difference (AD(L#PPA1_main_0-L#EPM_main_0), ...) is folded
* by the assembler and must not produce an item.
*
* The checks below start at the first RLD record (0321) and cover every item.
* Items that straddle two physical records contain the 3 byte continuation
* header (0323/0322) of the next record in the middle of the item.

* Case A: Subtract L#PPA2, Add CELQSTRT in the code section.
* RLD: 0321000001000000020004000000000000040000000400000004
* RLD-SAME: 60000000040000000000000b
* Case B: Add L#PPA2, Subtract CELQSTRT in the .&ppa2 pointer section.
* RLD-SAME: 0000000004000000000000040000000600000000
* RLD-SAME: 60000200040000000000000b
* Cases E, F, H: AD(L#PPA2), AD(L#PPA2+4), AD(L#PPA2-1), AD(Off8) at offsets
* 8, 12, 16, 24.  Off8 is an EQU (L#PPA2+8), so it resolves to the ESD of
* L#PPA2 (R == 4) and the addend 8 is kept in the TXT bytes.  The first item
* continues in the next physical record (0323).
* RLD-SAME: 00000000040000000000
* RLD-SAME: 03230000040000000800000008
* RLD-SAME: c0000000040000000000000c
* RLD-SAME: c00000000400000000000010
* RLD-SAME: c00000000400000000000018
* Case K: AD(i2) at 42, AD(i) at 46.  i2 and i are defined PR-section symbols
* (ESD IDs 9 and 10 respectively) so they sort before the external symbols.
* The i item straddles two physical records (0323).
* RLD-SAME: 4000000004000000000000090000002a
* RLD-SAME: 40000000040000000000000a000000
* RLD-SAME: 0323002e
* Cases C, D, G: A(ExtA) at 0, AD(ExtAD) at 4, AD(ExtA-ExtAD) at 20.
* RLD-SAME: 40000000040000000000000d00000000
* RLD-SAME: c0000000040000000000001440000000040000000000000e00000004
* RLD-SAME: c00002000400000000000014
* Case I: AD(ExtM) at 29.
* RLD-SAME: 40000000040000000000000f0000001d40000000
* Case J: AD(ExtC1) at 34, AD(ExtC2) at 38.
* RLD-SAME: 03220004000000000000100000002240000000040000000000001100000026

main#C   CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
main#C   XATTR LINKAGE(XPLINK),PSECT(main#S),SCOPE(SECTION)
L#PPA2   DS 0H
         DC XL1'03'
         DC XL1'E7'
         DC XL1'22'
         DC XL1'04'
* Case A: two RLD items, Add CELQSTRT and Subtract L#PPA2.
         DC A(CELQSTRT-L#PPA2)
         DC XL4'00000000'
         DC XL1'85'
         DC XL1'00'
         DC XL2'0000'

main#C   CSECT
C_@@QPPA2 CATTR ALIGN(3),FILL(0),NOTEXECUTABLE,READONLY,RMODE(64),PART(.
               .&ppa2)
.&ppa2   XATTR LINKAGE(OS),REFERENCE(DATA),SCOPE(SECTION)
* Case B: PPA2 minus an external symbol.
         DC A(L#PPA2-CELQSTRT)

main#C   CSECT
C_CODE64 CATTR
* -- Begin function main
         DS 0B
L#EPM_main_0 DS 0H
         DC XL7'00C300C500C500'
         DC XL1'F1'
* Same section: forward reference to PPA1, same section, folded by the assembler
* (no RLD item is expected for this one).
         DC A(L#PPA1_main_0-L#EPM_main_0)
         DC XL4'00000080'
         ENTRY main
main     XATTR LINKAGE(XPLINK),REFERENCE(CODE),PSECT(main#S),SCOPE(EXPOR
               RT)
main     DS 0H
         stmg 6,7,1840(4)
L#stack_update0 DS 0H
         aghi 4,-224
L#end_of_prologue0 DS 0H
         lg 1,0(5)
         lg 6,16(5)
         lg 5,8(5)
         stg 1,2264(4)
         basr 7,6
         bcr 0,0
         lg 7,2072(4)
         aghi 4,224
         b 2(7)
L#main_end_0 DS 0H
* -- End function
         DS 0B
L#PPA1_main_0 DS 0H
         DC XL1'02'
         DC XL1'CE'
         DC XL2'0300'
         DC A(L#PPA2-L#PPA1_main_0)
         DC XL1'80'
         DC XL1'80'
         DC XL1'00'
         DC XL1'81'
         DC XL2'0000'
L#tmp0   EQU L#end_of_prologue0-main
         DC A(L#tmp0/2)
L#tmp1   EQU L#stack_update0-main
         DC A(L#tmp1/2),XL1'0'
         DC A(L#main_end_0-L#EPM_main_0)
         DC XL2'0004'
         DC XL4'94819589'
         DC A(L#EPM_main_0-L#PPA1_main_0)

* Cases C-K: data constants in the writable static area.
main#C   CSECT
C_DATA   CATTR ALIGN(4),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(mai
               in#S)
main#S   XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
* Case C: A(ext)
         DC A(ExtA)
* Case D: AD(ext)
         DC A(ExtAD)
* Case E: AD of a symbol in another section
         DC A(L#PPA2)
* Case F: AD(sym+4), addend 4 in TXT
         DC A(L#PPA2+4)
* Case F: AD(sym-1), addend -1 in TXT
         DC A(L#PPA2-1)
* Case G: AD(ext1-ext2), Add + Subtract
         DC A(ExtA-ExtAD)
* Case H: AD of a symbol defined by EQU
Off8     EQU L#PPA2+8
         DC A(Off8)
* Case I: multi operand DC
         DC XL1'AB',A(ExtM),XL1'CD'
* Case J: two consecutive ACons
         DC A(ExtC1),A(ExtC2)
* Case K: pointers to data symbols (ADA slots)
         DC A(i2)
         DC A(i)

main#C   CSECT
C_CODE64 CATTR
         EXTRN CELQSTRT
CELQSTRT XATTR LINKAGE(OS),SCOPE(EXPORT)
         EXTRN ExtA
ExtA     XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
         EXTRN ExtAD
ExtAD    XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
         EXTRN ExtM
ExtM     XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
         EXTRN ExtC1
ExtC1    XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
         EXTRN ExtC2
ExtC2    XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
C_DATA   CATTR PART(i2)
i2       XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(EXPORT)
C_DATA   CATTR PART(i)
i        XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(EXPORT)
         END
