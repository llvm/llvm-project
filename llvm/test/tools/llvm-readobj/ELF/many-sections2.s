# Tests that symbols whose section index is SHN_XINDEX are displayed
# correctly. They should not be treated as processor/OS specific or
# reserved.
# REQUIRES: x86-registered-target
# RUN: llvm-mc %s -filetype=obj -triple=x86_64-pc-linux -o %t
# RUN: llvm-readelf -s %t | FileCheck %s

.irp i, 0, 1, 2, 3, 4, 5, 6,
  .irp j, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    .irp k, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
      .irp l, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        .irp q, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
          .section sec_\i\j\k\l\q
          .globl sym_\i\j\k\l\q
           sym_\i\j\k\l\q:
        .endr
      .endr
    .endr
  .endr
.endr

# CHECK:     65278: 0000000000000000 0 NOTYPE GLOBAL DEFAULT 65280 sym_65277
# CHECK:     65310: 0000000000000000 0 NOTYPE GLOBAL DEFAULT 65312 sym_65309
# CHECK:     65342: 0000000000000000 0 NOTYPE GLOBAL DEFAULT 65344 sym_65341
