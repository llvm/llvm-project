# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc --triple=loongarch32 -mattr=+relax --filetype=obj \
# RUN:     -o %t/la32_reloc_addsub.o %s
# RUN: llvm-jitlink --noexec --check %s %t/la32_reloc_addsub.o \
# RUN:     --slab-allocate=1Mb --slab-address=0x1000 --slab-page-size=0x4000
# RUN: llvm-mc --triple=loongarch64 -mattr=+relax --filetype=obj \
# RUN:     -o %t/la64_reloc_addsub.o %s
# RUN: llvm-jitlink --noexec --check %s %t/la64_reloc_addsub.o \
# RUN:     --slab-allocate=1Mb --slab-address=0x1000 --slab-page-size=0x4000

# jitlink-check: *{8}(named_data) = 0x8
# jitlink-check: *{4}(named_data+8) = 0x8
# jitlink-check: *{2}(named_data+12) = 0x8
# jitlink-check: *{1}(named_data+14) = 0x8
# jitlink-check: *{1}(named_data+15) = 0x10

# jitlink-check: *{1}(leb_data) = 0x8
# jitlink-check: *{2}(leb_data+1) = 0x180
# jitlink-check: *{8}(leb_data+3) = 0xfffffffffffffff8
# jitlink-check: *{2}(leb_data+11) = 0x1ff
# jitlink-check: *{1}(leb_data+13) = 0x7f
# jitlink-check: *{2}(leb_data+14) = 0x181

.section .alloc_data,"ax",@progbits
.global main
main:
.L0:
# Referencing named_data symbol to avoid the following relocations be
# skipped. This macro instruction will be expand to two instructions
# (pcalau12i + ld.w/d).
  la.global $t0, named_data
.L1:

named_data:
.reloc named_data+15, R_LARCH_ADD6, .L1
.reloc named_data+15, R_LARCH_SUB6, .L0
.dword .L1 - .L0
.word .L1 - .L0
.half .L1 - .L0
.byte .L1 - .L0
.byte 0x8

.size named_data, 16

leb_data:
.uleb128 .L1 - .L0
.uleb128 .L1 - .L0 + 120
.uleb128 -(.L1 - .L0)
.uleb128 leb_end - leb_data + 111
.uleb128 leb_end - leb_data + 113
leb_end:

.size leb_data, 16
