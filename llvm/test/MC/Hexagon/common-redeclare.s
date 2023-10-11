# RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump -t - | FileCheck %s

# CHECK: 00000062       O *COM*           00000008 quartet_table_isqrt

.common quartet_table_isqrt, 98, 8
.common quartet_table_isqrt, 98, 8
