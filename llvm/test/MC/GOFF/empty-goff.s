* RUN: llvm-mc <%s --triple s390x-ibm-zos --filetype=obj -o - | \
* RUN:   od -Ax -tx1 -v | FileCheck --ignore-case %s

* Header record:
*  03 is prefix byte
*  f. is header type
*  .0 is version
* The 1 at offset 51 is the architecture level.
* CHECK: 000000 03 f0 00 00 00 00 00 00 00 00 00 00 00 00 00 00
* CHECK: 000010 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
* CHECK: 000020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
* CHECK: 000030 00 00 00 01 00 00 00 00 00 00 00 00 00 00 00 00
* CHECK: 000040 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00

* End record:
*  03 is prefix byte
*  4. is header type
*  .0 is version
* CHECK: 000050 03 40 00 00 00 00 00 00 00 00 00 00 00 00 00 00
* CHECK: 000060 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
* CHECK: 000070 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
* CHECK: 000080 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
* CHECK: 000090 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
