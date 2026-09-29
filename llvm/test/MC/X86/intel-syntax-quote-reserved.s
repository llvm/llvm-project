# RUN: llvm-mc -triple x86_64 -output-asm-variant=1 %s | FileCheck %s --check-prefixes=CHECK,X64
# RUN: llvm-mc -triple i386 -defsym I386=1 -output-asm-variant=1 %s | FileCheck %s
# RUN: llvm-mc -triple x86_64-apple-darwin -output-asm-variant=1 %s | FileCheck %s --check-prefixes=CHECK,X64
# RUN: llvm-mc -triple x86_64-windows-msvc -output-asm-variant=1 %s | FileCheck %s --check-prefixes=CHECK,X64
# RUN: llvm-mc -triple x86_64-windows-gnu -output-asm-variant=1 %s | FileCheck %s --check-prefixes=CHECK,X64

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

## Register-name quoting uses the 64-bit `callq` mnemonic. The input block is
## gated by `.ifndef I386` so i386 skips it at assembly time; the output-check
## lines use the `X64` prefix so i386's FileCheck invocation ignores them too.
## (Register-quoting logic is identical across all X86 MCAsmInfo subclasses,
## so exercising it on 64-bit targets suffices.)
.ifndef I386
# X64:      call "rsi"
# X64-NEXT: call "Rax"
# X64-NEXT: call "EAX"
# X64-NEXT: call "ah"
# X64-NEXT: call "dil"
callq rsi
callq Rax
callq EAX
callq ah
callq dil
.endif

## Size/type keywords (gas treats these as size constants: byte=1, word=2, etc.)
# CHECK:      call "byte"
# CHECK-NEXT: call "word"
# CHECK-NEXT: call "dword"
# CHECK-NEXT: call "fword"
# CHECK-NEXT: call "qword"
# CHECK-NEXT: call "mmword"
# CHECK-NEXT: call "tbyte"
# CHECK-NEXT: call "oword"
# CHECK-NEXT: call "xmmword"
# CHECK-NEXT: call "ymmword"
# CHECK-NEXT: call "zmmword"
call byte
call word
call dword
call fword
call qword
call mmword
call tbyte
call oword
call xmmword
call ymmword
call zmmword

## Other keywords
# CHECK:      call "offset"
# CHECK-NEXT: call "flat"
# CHECK-NEXT: call "near"
# CHECK-NEXT: call "far"
# CHECK-NEXT: call "short"
call offset
call flat
call near
call far
call short

## Expression operator keywords
# CHECK:      call "and"
# CHECK-NEXT: call "eq"
# CHECK-NEXT: call "ge"
# CHECK-NEXT: call "gt"
# CHECK-NEXT: call "le"
# CHECK-NEXT: call "lt"
# CHECK-NEXT: call "mod"
# CHECK-NEXT: call "ne"
# CHECK-NEXT: call "not"
# CHECK-NEXT: call "or"
# CHECK-NEXT: call "shl"
# CHECK-NEXT: call "shr"
# CHECK-NEXT: call "xor"
call and
call eq
call ge
call gt
call le
call lt
call mod
call ne
call not
call or
call shl
call shr
call xor

## Symbols that don't need quoting (gas handles these correctly).
# CHECK:      call ptr
# CHECK-NEXT: call bar
call ptr
call bar

## Intel-syntax input with quoted identifiers round-trips to AT&T output with
## the quotes stripped.
# ATT:      callq rsi
# ATT-NEXT: callq byte
# ATT-NEXT: callq and
.intel_syntax noprefix
call "rsi"
call "byte"
call "and"
.att_syntax
