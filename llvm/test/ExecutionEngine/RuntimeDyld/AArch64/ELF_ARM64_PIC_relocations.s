# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=arm64-none-linux-gnu -filetype=obj -o %t/pic-reloc.o %s
# RUN: llvm-rtdyld -triple=arm64-none-linux-gnu -verify  -check=%s %t/pic-reloc.o \
# RUN:    -map-section pic-reloc.o,.got=0x20000 -dummy-extern f=0x1234 -dummy-extern g=0x5678

_s:
  nop
_a1:
	adrp	x8, :got:f
_a2:
	adrp	x9, :got:g
_a3:
  adrp  x10, :got:_s
_l1:
  ldr x8, [x8, :got_lo12:f]
_l2:
  ldr x9, [x9, :got_lo12:g]
_l3:
  ldr x10, [x10, :got_lo12:_s]


## We'll end up having two sections .text and .got,
## each is located on the start of a memory page

## Test that .got section has three entries pointing to f, g and _s
# *{8}section_addr(pic-reloc.o, .got) = f
# *{8}(section_addr(pic-reloc.o, .got) + 8) = g
# *{8}(section_addr(pic-reloc.o, .got) + 16) = _s

## Test that first adrp instruction really takes address of 
## the .got section (_s label is on the start of a page)
# rtdyld-check: _s + (((*{4}_a1)[30:29] + ((*{4}_a1)[23:5] << 2)) << 12) = section_addr(pic-reloc.o, .got)

## Test that second adrp takes address of .got
# rtdyld-check: _s + (((*{4}_a2)[30:29] + ((*{4}_a2)[23:5] << 2)) << 12) = section_addr(pic-reloc.o, .got)

## Test that third adrp takes address of .got
# rtdyld-check: _s + (((*{4}_a3)[30:29] + ((*{4}_a3)[23:5] << 2)) << 12) = section_addr(pic-reloc.o, .got)

## Test that first ldr immediate value is 0 >> 3 = 0 (1st .got entry)
# rtdyld-check: (*{4}_l1)[21:10] = 0

## Test that second ldr immediate value is 8 >> 3 = 1 (2nd .got entry)
# rtdyld-check: (*{4}_l2)[21:10] = 1

## Test that third ldr immediate value is 16 >> 3 = 2 (3rd .got entry, addend is 0)
# rtdyld-check: (*{4}_l3)[21:10] = 2
