# RUN: llvm-mc -triple x86_64 -output-asm-variant=1 %s | FileCheck %s
# RUN: llvm-mc -triple x86_64 -output-asm-variant=1 -filetype=obj %s | llvm-objdump -dr - | FileCheck %s --check-prefix=DIS

## Round-trip: all calls should be direct (0xe8), not indirect (0xff).
# DIS-NOT: callq *

## Symbols whose names match registers or Intel syntax keywords must be quoted
## in Intel syntax output. Without quoting, the assembler would misparse these
## as registers, size specifiers, or expression operators.

## Register names
# CHECK: call "rsi"
# CHECK: call "rax"
# CHECK: call "EAX"
# CHECK: call "Ah"
callq rsi
callq rax
callq EAX
callq Ah

## Size/type keywords (gas treats these as size constants: byte=1, word=2, etc.)
# CHECK: call "byte"
# CHECK: call "word"
# CHECK: call "dword"
# CHECK: call "fword"
# CHECK: call "qword"
# CHECK: call "mmword"
# CHECK: call "tbyte"
# CHECK: call "oword"
# CHECK: call "xmmword"
# CHECK: call "ymmword"
# CHECK: call "zmmword"
callq byte
callq word
callq dword
callq fword
callq qword
callq mmword
callq tbyte
callq oword
callq xmmword
callq ymmword
callq zmmword

## Other keywords
# CHECK: call "ptr"
# CHECK: call "offset"
# CHECK: call "flat"
# CHECK: call "near"
# CHECK: call "far"
# CHECK: call "short"
callq ptr
callq offset
callq flat
callq near
callq far
callq short

## Expression operator keywords
# CHECK: call "and"
# CHECK: call "eq"
# CHECK: call "ge"
# CHECK: call "gt"
# CHECK: call "le"
# CHECK: call "lt"
# CHECK: call "mod"
# CHECK: call "ne"
# CHECK: call "not"
# CHECK: call "or"
# CHECK: call "shl"
# CHECK: call "shr"
# CHECK: call "xor"
callq and
callq eq
callq ge
callq gt
callq le
callq lt
callq mod
callq ne
callq not
callq or
callq shl
callq shr
callq xor

## Normal symbols remain unquoted.
# CHECK: call bar
callq bar
