* RUN: llvm-mc <%s --triple s390x-ibm-zos --filetype=obj -o - | \
* RUN:   od -An -tx1 -v | tr -d ' \n' > %t.hex
* RUN: FileCheck --ignore-case --check-prefix=RLD %s < %t.hex
* REQUIRES: systemz-registered-target
*
* Relative immediate instructions (e.g. brasl, larl, brcl, bras, brc) referencing
* external or cross-section symbols produce an RLD item with reference type 6
* (RLD_RT_RRelativeImmediate) and referent type 0 (RLD_RO_Label), so f1 == 0x60.
* Action is Add (f2 == 0x00).
* Field length (f4) is 4 for 32-bit relative immediate and 2 for 16-bit relative
* immediate.
* Intra-section relative immediate instructions are resolved directly by the
* assembler and do not emit an RLD item.
*
* RLD item layout (all items below have f1 == 60, i.e. ReferenceType == 6 and
* ReferentType == 0):
*   byte 0 (f0): 0x80=same-R  0x40=same-P  0x20=same-Offset
*   byte 1 (f1): (ReferenceType<<4) | ReferentType  -> 60 here
*   byte 2 (f2): (Action<<1) | FetchStore           -> 00 = Add/Fetch
*   byte 4 (f4): target field length in bytes       -> 04 or 02
* followed by the optional R-ESDID, P-ESDID and P-Offset.
*
* ESDIDs, in order of definition:
*    1 SD #C        2 ED C_CODE64     (root SD that owns the ERs)
*    3 SD main#C    4 ED C_CODE64     5 ED C_DATA
*    6 PR main#S    7 ED C_OTHER      8 SD Other#C
*   10 LD main     11 ER ExtFunc     12 ER ExtSym
*   13 LD OtherSecLabel
*
* Cases (P = pointer section ESDID 4 (C_CODE64)):
*   A. brasl 7, ExtFunc        : R=ExtFunc (11), offset 2, f4=4
*   B. larl 1, ExtSym          : R=ExtSym (12), offset 8, f4=4
*   C. brcl 15, ExtFunc        : R=ExtFunc (11), offset 14, f4=4
*   D. brasl 7, OtherSecLabel  : R=OtherSecLabel (13), offset 20, f4=4
*   E. brasl 7, ExtFunc+4      : R=ExtFunc (11), offset 26, f4=4
*   F. bras 7, ExtFunc         : R=ExtFunc (11), offset 32, f4=2
*   G. brc 15, OtherSecLabel   : R=OtherSecLabel (13), offset 36, f4=2
*   H. Intra-section branches to LocalLabel (offsets 38-54) -> folded, no RLD item
*
* The RLD items are sorted by PEsdId, REsdId, POffset.
* Items with R=11 (ExtFunc) at offsets 2, 14, 26 (f4=4) and 32 (f4=2):
* RLD: 0321
* RLD-SAME: 00600000040000000000000b0000000400000002
* RLD-SAME: c0600000040000000000000e
* RLD-SAME: c0600000040000000000001a
* RLD-SAME: c06000000200000000000020
* Item with R=12 (ExtSym) at offset 8 (f4=4):
* RLD-SAME: 40600000040000000000000c00000008
* Items with R=13 (OtherSecLabel) at offset 20 (f4=4) and 36 (f4=2); the first
* item continues in the next physical record (0322).
* RLD-SAME: 4060
* RLD-SAME: 0322000000040000000000000d00000014
* RLD-SAME: c06000000200000000000024

main#C   CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
         ENTRY main
main     XATTR LINKAGE(XPLINK),REFERENCE(CODE),PSECT(main#S),SCOPE(EXPOR
               RT)
main     DS 0H
* Case A: External function call (brasl) - 4-byte relocation at offset 2
         brasl 7,ExtFunc
* Case B: External symbol address (larl) - 4-byte relocation at offset 8
         larl 1,ExtSym
* Case C: External branch long (brcl) - 4-byte relocation at offset 14
         brcl 15,ExtFunc
* Case D: Cross-section reference (brasl) - 4-byte relocation at offset 20
         brasl 7,OtherSecLabel
* Case E: External call with addend (brasl) - 4-byte relocation at offset 26
         brasl 7,ExtFunc+4
* Case F: External call short (bras) - 2-byte relocation at offset 32 (instruction is 4 bytes at offset 30)
         bras 7,ExtFunc
* Case G: Cross-section branch short (brc) - 2-byte relocation at offset 36 (instruction is 4 bytes at offset 34)
         brc 15,OtherSecLabel
* Case H: Intra-section branches - resolved directly, no RLD entry
         brasl 7,LocalLabel
         larl 1,LocalLabel
         bras 7,LocalLabel
         brc 15,LocalLabel
LocalLabel DS 0H
         bcr 0,0

* Other section
Other#C  CSECT
C_OTHER  CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
OtherSecLabel DS 0H
         bcr 0,0

main#C   CSECT
C_CODE64 CATTR
         EXTRN ExtFunc
ExtFunc  XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
         EXTRN ExtSym
ExtSym   XATTR LINKAGE(XPLINK),SCOPE(EXPORT)

main#C   CSECT
C_DATA   CATTR ALIGN(4),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(mai
               in#S)
main#S   XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
         DS 0D
         END
