# RUN: llvm-mc -filetype=obj -triple=csky -mattr=+2e3 %s -o %t
# RUN: llvm-readelf -rs %t | FileCheck %s --check-prefix=READELF

# READELF: '.rela.data'
# READELF: R_CKCORE_GOT32 00000000 local + 0
# READELF: R_CKCORE_PLT32 00000000 local + 0

# READELF: TLS GLOBAL DEFAULT UND gd
# READELF: TLS GLOBAL DEFAULT UND ld
# READELF: TLS GLOBAL DEFAULT UND ie
# READELF: TLS GLOBAL DEFAULT UND le

lrw16 r0, gd@TLSGD32
lrw16 r0, ld@TLSLDM32
lrw16 r3, ie@GOTTPOFF
lrw16 r3, le@TPOFF

.data
local:
.long local@GOT
.long local@plt
