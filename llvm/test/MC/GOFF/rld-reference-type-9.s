* RUN: llvm-mc <%s --triple s390x-ibm-zos --filetype=obj -o - | \
* RUN:   od -An -tx1 -v | tr -d ' \n' > %t.hex
* RUN: FileCheck --ignore-case --check-prefix=RLD %s < %t.hex
* REQUIRES: systemz-registered-target
*
* Instructions with 20-bit long displacement referencing a relocatable or
* external symbol produce an RLD item with reference type 9
* (RLD_RT_RLongDisplacement) and referent type 0 (RLD_RO_Label), so f1 == 0x90.
* The target field (the 20-bit displacement) spans 3 bytes (f4 == 3).
*
* All four items refer to ExtSym (R-ESDID 8) from section ESDID 4, at offsets
* 0x0c, 0x12, 0x18 and 0x1e.
* RLD: 0320
* RLD-SAME: 00900000030000000000000800000004
* RLD-SAME: 0000000c
* RLD-SAME: c09000000300000000000012
* RLD-SAME: c09000000300000000000018
* RLD-SAME: c0900000030000000000001e


main#C   CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
         ENTRY main
main     XATTR LINKAGE(XPLINK),REFERENCE(CODE),PSECT(main#S),SCOPE(EXPOR
               RT)
main     DS 0H
         stmg 6,7,1840(4)
         aghi 4,-224

* Case A: Long displacement load address (LAY) referencing an external symbol
         lay  1,ExtSym(2)

* Case B: 64-bit load (LG) with base/index and relocatable displacement
         lg   2,ExtSym(1,3)

* Case C: 64-bit store (STG) with relocatable displacement
         stg  2,ExtSym(1,3)

* Case D: Long displacement add (AGY)
         lay  1,ExtSym(2)

         lg 7,2072(4)
         aghi 4,224
         b 2(7)

main#C   CSECT
C_DATA   CATTR ALIGN(4),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(mai
               in#S)
main#S   XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)

main#C   CSECT
C_CODE64 CATTR
         EXTRN ExtSym
ExtSym   XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
         END
