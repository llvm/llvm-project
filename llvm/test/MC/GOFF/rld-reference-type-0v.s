* RUN: llvm-mc <%s --triple s390x-ibm-zos --filetype=obj -o - | \
* RUN:   od -An -tx1 -v | tr -d ' \n' > %t.hex
* RUN: FileCheck --ignore-case --check-prefix=RLD %s < %t.hex
* REQUIRES: systemz-registered-target
*
* DC V(sym) / DC VD(sym) in a module that defines the function "main".
*
* A V-type constant is recorded as an RLD item with reference type 0
* (RLD_RT_RAddress) and referent type 0 (RLD_RO_Label), like an A-type
* constant, but with the Store flag set:
*   byte 0 (f0): 0x80=same-R  0x40=same-P  0x20=same-Offset
*   byte 1 (f1): 00  ReferenceType 0, ReferentType 0
*   byte 2 (f2): 01  Action 0 (Add), FetchStore 1 (Store)
*   byte 4 (f4): field length
* Compare with an A-type constant, where f2 is 00 (Fetch).  Only R-type
* constants (RD) get reference type 7.
* The symbol passed to V/VD must be an external symbol: an external label,
* an entry point or a CSECT/section name. A local symbol or a local part
* (a PR defined in this module) is not a valid operand and is not tested here.
*
* Cases:
*   A. DC V(ext)               4 byte field
*   B. DC VD(ext)              8 byte field
*   C. DC V(ext),VD(ext2)      consecutive V-cons, mixed length
*   D. DC XL1,VD(ext),XL1      unaligned VD in a multi operand DC
*   E. DC VD(ext)              same symbol referenced twice
*   F. DC A(ext)               control: A-type constant, Fetch, no Store flag
*   G. DC VD(main)             entry point
*   H. DC VD(Other#C)          control section name
*   I. DC VD(ExtPart)          external part (EXTRN PART)
*   J. DC VD(WeakExt)          weak external (WXTRN)
*
* ESDIDs referenced (R pointer): 10 LD main, 11 ER VExt4, 12 ER VExt8,
* 13 ER VExt4b, 14 ER VExt8b, 15 ER VExt8c, 16 ER AExt, 17 ER Other#C,
* 18 PR ExtPart, 19 ER WeakExt.  The P pointer is ESDID 6 (the main#S
* pointer section).
*
* Pointer section offsets (items are emitted ordered by referenced ESDID):
*   A: V(VExt4)     off  0, f4=04
*   B: VD(VExt8)    off  4, f4=08
*   C: V(VExt4b)    off 12, f4=04;  VD(VExt8b) off 16, f4=08
*   D: VD(VExt8c)   off 25, f4=08
*   E: VD(VExt8)    off 34, f4=08   (same R and P as item B: flags c0)
*   F: A(AExt)      off 42, f4=04   (f2=00, i.e. no Store flag)
*   G: VD(main)     off 46, f4=08
*   H: VD(Other#C)  off 54, f4=08
*   I: VD(ExtPart)  off 62, f4=08
*   J: VD(WeakExt)  off 70, f4=08
* The physical RLD record is split twice: the 3 byte continuation header
* 0323 appears in the middle of the V(VExt4b) item and 0322 in the middle of
* the VD(ExtPart) item.

* Case G: VD(main), R=10, offset 46.
* RLD: 0321000000b0
* RLD-SAME: 00000100080000000000000a000000060000002e
* Case A: V(VExt4), Add/Store, 4 bytes, offset 0.
* RLD-SAME: 40000100040000000000000b00000000
* Case B: VD(VExt8), Add/Store, 8 bytes, offset 4.
* RLD-SAME: 40000100080000000000000c00000004
* Case E: VD(VExt8) again, same R and P, offset 34.
* RLD-SAME: c00001000800000000000022
* Case C: V(VExt4b) at 12 (continued in 0323).
* RLD-SAME: 40000100040000000000
* RLD-SAME: 032300000d0000000c
* Case C: VD(VExt8b) at 16.
* RLD-SAME: 40000100080000000000000e00000010
* Case D: VD(VExt8c) at 25.
* RLD-SAME: 40000100080000000000000f00000019
* Case F: control, A(AExt) at 42: reference type 0 but Fetch (f2=00).
* RLD-SAME: 4000000004000000000000100000002a
* Case H: VD(Other#C) at 54.
* RLD-SAME: 40000100080000000000001100000036
* Case I: VD(ExtPart) at 62 (continued in 0322).
* RLD-SAME: 40000100080000
* RLD-SAME: 03220000000000120000003e
* Case J: VD(WeakExt) at 70.
* RLD-SAME: 40000100080000000000001300000046


main#C   CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
main#C   XATTR LINKAGE(XPLINK),PSECT(main#S),SCOPE(SECTION)
L#PPA2   DS 0H
         DC XL1'03'
         DC XL1'E7'
         DC XL1'22'
         DC XL1'04'
         DC XL4'00000000'
         DC XL4'00000000'
         DC XL1'85'
         DC XL1'00'
         DC XL2'0000'

main#C   CSECT
C_CODE64 CATTR
* -- Begin function main
         DS 0B
L#EPM_main_0 DS 0H
         DC XL7'00C300C500C500'
         DC XL1'F1'
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
         DC XL4'00000000'
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

main#C   CSECT
C_DATA   CATTR ALIGN(4),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(mai
               in#S)
main#S   XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
* Case A: V(ext)
         DC V(VExt4)
* Case B: VD(ext)
         DC VD(VExt8)
* Case C: consecutive V and VD
         DC V(VExt4b),VD(VExt8b)
* Case D: unaligned VD in a multi operand DC
         DC XL1'AB',VD(VExt8c),XL1'CD'
* Case E: same symbol referenced twice
         DC VD(VExt8)
* Case F: control, A-type constant
         DC A(AExt)
* Case G: entry point
         DC VD(main)
* Case H: control section name
         DC VD(Other#C)
* Case I: external part
         DC VD(ExtPart)
* Case J: weak external
         DC VD(WeakExt)

main#C   CSECT
C_CODE64 CATTR
         EXTRN VExt4
VExt4    XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
         EXTRN VExt8
VExt8    XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
         EXTRN VExt4b
VExt4b   XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
         EXTRN VExt8b
VExt8b   XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
         EXTRN VExt8c
VExt8c   XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
         EXTRN AExt
AExt     XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
         WXTRN WeakExt
WeakExt  XATTR LINKAGE(XPLINK),SCOPE(EXPORT)

Other#C  CSECT
C_OTHER  CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
         bcr 0,0

main#C   CSECT
C_DATA   CATTR PART(main#S)
         EXTRN PART(ExtPart)

         END
