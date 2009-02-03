; RUN: llvm-as < %s | llc -march=x86 -mtriple=i686-pc-linux-gnu | grep 18446744073709551615 | count 14

; These static initializers are too big to hand off to assemblers
; as monolithic blobs.

@x = global i128 -1
@y = global i256 -1
@z = global i512 -1
