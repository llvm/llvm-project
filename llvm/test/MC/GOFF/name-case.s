* RUN: llvm-mc <%s --triple s390x-ibm-zos --filetype=asm | \
* RUN:   FileCheck %s
* RUN: llvm-mc <%s --triple s390x-ibm-zos --filetype=obj -o - | \
* RUN:   od -An -tx1 -v | tr -d ' \n' | FileCheck --check-prefix=OBJ %s
* REQUIRES: systemz-registered-target

* HLASM names are case-insensitive: all spellings of a name refer to the same
* symbol. The spelling of the first occurrence is preserved rather than being
* folded to upper case.

* CHECK:      MySect CSECT
* CHECK:      ENTRY FooBar
* CHECK:      FooBar XATTR LINKAGE(XPLINK),SCOPE(SECTION)
* CHECK:      FooBar DS 0H
* CHECK-NEXT: DC A(FooBar)
* CHECK-NEXT: DC A(FooBar)
* CHECK-NEXT: DC A(ExtSym)
* CHECK:      EXTRN ExtSym
* CHECK-NOT:  MYSECT
* CHECK-NOT:  FOOBAR
* CHECK-NOT:  EXTSYM

* "FooBar" (EBCDIC: c6 96 96 c2 81 99) must be present in the object file
* with its original case, and "MySect" (d4 a8 e2 85 83 a3) as well.
* OBJ: d4a8e28583a3
* OBJ: c69696c28199

MySect   CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
FooBar   DS    0H
         DC    A(foobar)
         DC    A(FOOBAR)
         ENTRY fOOBAR
         EXTRN ExtSym
         DC    A(EXTSYM)
         END
