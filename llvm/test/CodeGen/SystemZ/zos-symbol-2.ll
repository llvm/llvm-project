; RUN: llc <%s --mtriple s390x-ibm-zos | FileCheck %s
; RUN: llc <%s --mtriple s390x-ibm-zos --filetype=obj | \
; RUN:   od -Ax -tx1 -v | FileCheck --check-prefix=CHECKREL --ignore-case %s

@a = hidden global i64 0, align 8
@b = external global i64, align 8

define i64 @calc() {
entry:
  %0 = load i64, ptr @a, align 8
  %1 = load i64, ptr @b, align 8
  %add = add i64 %0, %1
  ret i64 %add
}

; Check the global CSECT definition
; CHECK:      stdin#C CSECT
; CHECK-NEXT: C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)

; Check the attributes on the function
; CHECK:       ENTRY calc
; CHECK-NEXT: calc XATTR LINKAGE(XPLINK),REFERENCE(CODE),PSECT(stdin#S),SCOPE(EXPORT)
; CHECK-NEXT: calc DS 0H

; Check the definition of the variable
; CHECK:      a CSECT
; CHECK-NEXT: C_WSA64 CATTR ALIGN(3),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(a)
; CHECK-NEXT: a XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(LIBRARY)

; Check the declaration of the external variable
; CHECK:      C_WSA64 CATTR PART(b)
; CHECK-NEXT: b XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(EXPORT)

; Check the C_WSA64
; CHECKREL:      0002d0 03 00 00 01 [[C_WSA64:00 00 00 08]] 00 00 00 01 00 00 00 00
; CHECKREL-NEXT: 0002e0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECKREL-NEXT: 0002f0 00 00 00 00 00 00 00 00 03 81 00 00 00 00 00 00
; CHECKREL-NEXT: 000300 00 00 00 00 00 00 00 00 00 00 00 00 00 04 01 00
; CHECKREL-NEXT: 000310 00 40 04 00 00 00 00 07 c3 6d e6 e2 c1 f6 f4 00

; Check the PR symbol for b
; CHECKREL:      0004b0 03 00 00 03 00 00 00 0e [[C_WSA64]] 00 00 00 00
; CHECKREL-NEXT: 0004c0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECKREL-NEXT: 0004d0 00 00 00 00 00 00 00 00 03 00 00 00 00 00 00 00
; CHECKREL-NEXT: 0004e0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01
; CHECKREL-NEXT: 0004f0 00 04 24 00 00 00 00 01 82 00 00 00 00 00 00 00

; Check the relocation data directory.
;  03 is prefix byte
;  2. is header type (RT_RLD)
;  .1 is flag (record is continued)
;  00 is version
; CHECKREL:      000690 03 21 00 00 00 60 00 00 02 00 04 00 00 00 00 00
; CHECKREL-NEXT: 0006a0 00 0b 00 00 00 02 00 00 00 04 60 00 00 00 04 00
; CHECKREL-NEXT: 0006b0 00 00 00 00 00 0c 00 00 00 00 08 00 00 00 00 00
; CHECKREL-NEXT: 0006c0 00 0b 00 00 00 04 00 00 00 00 60 00 02 00 08 00
; CHECKREL-NEXT: 0006d0 00 00 00 00 00 0c 20 00 00 00 08 00 00 00 00 00
; Continuation of relocation data directory.
;  03 is prefix byte
;  2. is header type (RT_RLD)
;  .2 is flag (record is continuation but not continued)
;  00 is version
; CHECKREL-NEXT: 0006e0 03 22 00 00 07 00 00 00 09 40 00 00 00 08 00 00
; CHECKREL-NEXT: 0006f0 00 00 00 00 0e 00 00 00 08 00 00 00 00 00 00 00
; CHECKREL-NEXT: 000700 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECKREL-NEXT: 000710 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECKREL-NEXT: 000720 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
