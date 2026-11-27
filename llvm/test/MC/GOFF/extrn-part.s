* RUN: llvm-mc <%s --triple s390x-ibm-zos --filetype=obj -o - | \
* RUN:   od -An -tx1 -v | tr -d ' \n' | FileCheck --ignore-case %s
* REQUIRES: systemz-registered-target

* Verify that EXTRN PART(sym[,sym...]) declares external data parts that are
* associated with the class of the current part. EBCDIC "ExtDat1" and
* "ExtDat2" must appear in the object, and "WeakDat" for WXTRN. HLASM names
* are case-insensitive, but the spelling of the source is preserved.

* CHECK: c5a7a3c481a3f1
* CHECK: c5a7a3c481a3f2
* CHECK: e6858192c481a3

EXTPART  CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
         DC    XL1'00'
EXTPART  CSECT
C_WSA64  CATTR ALIGN(3),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(LOC
               CDAT)
         DC    AD(ExtDat1)
         DC    AD(ExtDat2)
         DC    AD(WeakDat)
         EXTRN PART(ExtDat1,ExtDat2)
         WXTRN PART(WeakDat)
         END
