* RUN: not llvm-mc %s --triple s390x-ibm-zos --filetype=obj -o /dev/null 2>&1 | \
* RUN:   FileCheck %s
* REQUIRES: systemz-registered-target
*
* Diagnostics for malformed DS directives.

DSERR    CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)

* CHECK: error: unsupported DS type 'X'
         DS 0X

* CHECK: error: DS with non-zero duplication factor not yet supported
         DS 2H

* CHECK: error: expected DS type character (B, H, F, D, Q)
         DS 0

         END
