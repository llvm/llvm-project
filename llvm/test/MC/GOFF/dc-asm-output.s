* RUN: llvm-mc <%s --triple s390x-ibm-zos --filetype=asm | FileCheck %s
* REQUIRES: systemz-registered-target
*
* The HLASM output must keep the constant type and size of V/VD constants, and
* must print the EQU statements that define the symbols used in expressions.

main#C   CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
L#a      DS 0H
         DC XL4'00000000'
L#b      DS 0H
L#d      EQU L#b-L#a
* CHECK: L#d EQU L#b-L#a
         DC A(L#d/2)
* CHECK: DC A(L#d/2)

main#C   CSECT
C_DATA   CATTR ALIGN(4),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(mai
               in#S)
main#S   XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
* CHECK: DC V(Ext4)
         DC V(Ext4)
* CHECK: DC VD(Ext8)
         DC VD(Ext8)
* CHECK: DC RD(Ext8)
         DC RD(Ext8)
* CHECK: DC AD(RD(Ext8)+XL8'0000000000000010')
         DC AD(RD(Ext8)+16)

main#C   CSECT
C_CODE64 CATTR
         EXTRN Ext4
Ext4     XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
         EXTRN Ext8
Ext8     XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
         END
