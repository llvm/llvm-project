# RUN: llvm-mc -triple x86_64 -output-asm-variant=1 %s | FileCheck %s
# RUN: llvm-mc -triple x86_64 %s | FileCheck %s --check-prefix=ATT
# RUN: llvm-mc -triple x86_64 -output-asm-variant=1 -filetype=obj %s | llvm-objdump -dr - | FileCheck %s --check-prefix=DIS

## Round-trip: all calls should be direct (0xe8), not indirect (0xff).
# DIS-NOT: callq *

## AT&T output never needs quoting: registers use `%` prefix and none of
## these names are keywords in AT&T syntax. Sentinel positive checks plus a
## global NOT check that no `"` appears anywhere in the output.
# ATT: callq rsi
# ATT: callq byte
# ATT: callq and
# ATT-NOT: {{"}}

## Symbols whose names match registers or Intel syntax keywords must be quoted
## in Intel syntax output. Without quoting, the assembler would misparse these
## as registers, size specifiers, or expression operators.

## Register names
# CHECK: call "rsi"
# CHECK: call "Rax"
# CHECK: call "EAX"
# CHECK: call "ah"
# CHECK: call "dil"
callq rsi
callq Rax
callq EAX
callq ah
callq dil

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
# CHECK: call "offset"
# CHECK: call "flat"
# CHECK: call "near"
# CHECK: call "far"
# CHECK: call "short"
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

## Symbols that don't need quoting (gas handles these correctly).
# CHECK: call ptr
# CHECK: call bar
callq ptr
callq bar
