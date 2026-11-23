* RUN: llvm-mc <%s --triple s390x-ibm-zos --filetype=obj -o - | \
* RUN:   od -An -tx1 -v | tr -d ' \n' | FileCheck --ignore-case %s
* REQUIRES: systemz-registered-target

* Verify GOFFObjectWriter::defineSymbols skips variable symbols (EQU), and
* that a DATA symbol backed by a CATTR PART section is emitted correctly.

* HDR record
* CHECK: 03f0
* The EQU symbol must not get an ESD entry (EBCDIC "ConstVal").
* CHECK-NOT: c39695a2a3e58193
* The exported DATA symbol is emitted (EBCDIC "DataSym").
* CHECK: c481a381e2a894
* CHECK-NOT: c39695a2a3e58193

SKIPVAR  CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
SKIPVAR  XATTR LINKAGE(XPLINK),SCOPE(SECTION)

Main     DS    0H
ConstVal EQU   10+2
         DC    AD(ConstVal)
SKIPVAR  CSECT
C_WSA64  CATTR ALIGN(4),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(DataSym)
DataSym  XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(EXPORT)
         DC    AD(DataSym)
         END   Main
