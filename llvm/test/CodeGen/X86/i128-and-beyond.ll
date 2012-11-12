; RUN: llc < %s -march=x86 -mtriple=i686-pc-linux-gnu | grep -- -1 | count 14

; These static initializers are too big to hand off to assemblers
; as monolithic blobs.

@x = global i128 -1
@y = global i256 -1
@z = global i512 -1
