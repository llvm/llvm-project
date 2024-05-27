# REQUIRES: webassembly-registered-target
# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj %s -o %t.o -g
# RUN: llvm-symbolizer --basenames --output-style=GNU -e %t.o 1 2 3 4 5 6 7 8 9 10 11 12 13 | FileCheck %s

foo:
    .functype foo () -> ()
    nop
    return
    end_function

bar:
    .functype bar (i32) -> (i32)
    local.get 0
    nop
    return
    end_function


## Symbols start from (including) the function length and should cover all the
## way to the next symbol start.
## TODO: create a loc for .functype? It could go with the local declarations.

## Byte 1 is the function length, has no loc but the symbol table considers it
## the start of the function
# CHECK: foo
# CHECK-NEXT: ??:0
## Byte 2 is the local declaration, but for some reason DWARF is marking it as line 7.
## TODO: figure out why.
# CHECK-NEXT: foo
# CHECK-NEXT: wasm-basic.s:7
## Byte 3 is actually the nop, line 7
# CHECK-NEXT: foo
# CHECK-NEXT: wasm-basic.s:7
## Byte 4 is the return, line 8
# CHECK-NEXT: foo
# CHECK-NEXT: wasm-basic.s:8
## Byte 5 is the end_function, line 9
# CHECK-NEXT: foo
# CHECK-NEXT: wasm-basic.s:9
## Byte 6 is bar's function length, symbol table considers it part of bar
# CHECK-NEXT: bar
# CHECK-NEXT: ??:0
## Byte 7 bar's local declaration, but DWARF marks it as line 13, like above
# CHECK-NEXT: bar
# CHECK-NEXT: wasm-basic.s:13
## Byte 8 and 9 are actually the local.get on line 13
# CHECK-NEXT: bar
# CHECK-NEXT: wasm-basic.s:13
# CHECK-NEXT: bar
# CHECK-NEXT: wasm-basic.s:13
## Byte 10 is the nop
# CHECK-NEXT: bar
# CHECK-NEXT: wasm-basic.s:14
## Byte b is the return
# CHECK-NEXT: bar
# CHECK-NEXT: wasm-basic.s:15
## Byte c is end_function
# CHECK-NEXT: bar
# CHECK-NEXT: wasm-basic.s:16
