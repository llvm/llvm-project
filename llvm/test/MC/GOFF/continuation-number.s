* RUN: llvm-mc <%s --triple s390x-ibm-zos --filetype=asm | FileCheck %s
* REQUIRES: systemz-registered-target
*
* A decimal number may be split across a continuation line (non-blank in
* column 72, statement resumes in column 16 of the next line).  Here
* PRIORITY(2147418135) is split as "2147" + "418135".
* The asm streamer re-wraps its output at column 72.
* CHECK: PRIORITY(21474
* CHECK-NEXT: {{ +}}418135),PART(.xtor.22)
*
* The continuation indicator in column 72 may also be a digit; it is not part
* of the number.  Here it is "4", so the digits are "2147" + "418135" again.
* CHECK: PRIORITY(21474
* CHECK-NEXT: {{ +}}418135),PART(.xtor.23)
zos_num#C CSECT
C_@@SQINIT CATTR ALIGN(0),FILL(3),NOTEXECUTABLE,RMODE(64),PRIORITY(2147X aaa    
               418135),PART(.xtor.22)
.xtor.22 XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
 DS 0B
 DC XL4'01020304'
zos_num#C CSECT
C_@@SQINIT CATTR ALIGN(0),FILL(3),NOTEXECUTABLE,RMODE(64),PRIORITY(21474 bbb    
               418135),PART(.xtor.23)
.xtor.23 XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
 DS 0B
 DC XL4'01020304'
 END
