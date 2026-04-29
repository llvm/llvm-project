# REQUIRES: webassembly-registered-target, lld
# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj %s -o %t.o -g
# RUN: llvm-symbolizer --basenames --output-style=GNU -e %t.o 0 1 2 3 4 5 6 7 8 9 10 11 12 13 16 | FileCheck %s

# This is the same test but on a linked wasm file, using file offsets rather than section offsets.
# The code section starts at 0x41, so these are equivalent to above (except 3, see below for why).
# RUN: wasm-ld %t.o -o %t.wasm --no-entry --export=foo --export=bar
# RUN: llvm-symbolizer --basenames --output-style=GNU -e %t.wasm 3 0x42 0x43 0x44 0x45 0x46 0x47 0x48 0x49 0x4a 0x4b 0x4c 0x4d 0x4e 0x51 | FileCheck %s

.globl foo
foo:
    .functype foo () -> ()
    nop
    return
    end_function

.globl bar
bar:
    .functype bar (i32) -> (i32)
    local.get 0
    nop
    return
    end_function


## Symbols start from (including) the function length and should cover all the
## way to the next symbol start.
## TODO: create a loc for .functype? It could go with the local declarations.

## For objects, byte 0 doesn't map to a function body.
## For the linked file, 3 is chosen because if incorrectly interpreted as a
## section offset, it will match a line.
# CHECK: ??
# CHECK-NEXT: ??:0

## Byte 1 is the function length, has no loc but the symbol table considers it
## the start of the function
# CHECK: foo
# CHECK-NEXT: ??:0
## Byte 2 is the local declaration, but for some reason DWARF is marking it as line 13.
## TODO: figure out why.
# CHECK-NEXT: foo
# CHECK-NEXT: wasm-basic.s:13
## Byte 3 is actually the nop, line 13
# CHECK-NEXT: foo
# CHECK-NEXT: wasm-basic.s:13
## Byte 4 is the return, line 14
# CHECK-NEXT: foo
# CHECK-NEXT: wasm-basic.s:14
## Byte 5 is the end_function, line 15
# CHECK-NEXT: foo
# CHECK-NEXT: wasm-basic.s:15
## Byte 6 is bar's function length, symbol table considers it part of bar
# CHECK-NEXT: bar
# CHECK-NEXT: ??:0
## Byte 7 bar's local declaration, but DWARF marks it as line 20, like above
# CHECK-NEXT: bar
# CHECK-NEXT: wasm-basic.s:20
## Byte 8 and 9 are the local.get on line 20
# CHECK-NEXT: bar
# CHECK-NEXT: wasm-basic.s:20
# CHECK-NEXT: bar
# CHECK-NEXT: wasm-basic.s:20
## Byte 10 is the nop
# CHECK-NEXT: bar
# CHECK-NEXT: wasm-basic.s:21
## Byte 11 is the return
# CHECK-NEXT: bar
# CHECK-NEXT: wasm-basic.s:22
## Byte 12 is end_function
# CHECK-NEXT: bar
# CHECK-NEXT: wasm-basic.s:23
## Byte 13 is past the end of the function
# CHECK-NEXT: ??
# CHECK-NEXT: ??:0

## Byte 16 is further past the end of the section.
# CHECK: ??
# CHECK-NEXT: ??:0
