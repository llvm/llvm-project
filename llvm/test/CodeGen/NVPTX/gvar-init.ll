; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

; Error out if initializer is given for address spaces that do not support initializers
; XFAIL: *
@g0 = addrspace(3) global i32 42
