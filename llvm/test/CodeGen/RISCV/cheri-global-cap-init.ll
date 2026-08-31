; RUN: llc -mtriple=riscv32 --relocation-model=pic -target-abi il32pc64f -mattr=+experimental-y,+f -relocation-model=static %s -o -
; RUN: llc -mtriple=riscv32 --relocation-model=pic -target-abi il32pc64f -mattr=+experimental-y,+f %s -o -
; RUN: llc -mtriple=riscv32 --relocation-model=pic -target-abi il32pc64f -mattr=+experimental-y,+f -filetype=obj %s -o -

; Ported from CodeGen/CHERI-Generic/Inputs/cheri-global-cap-init.ll in CHERI downstreams.

; FIXME: Add RV64 RUN lines and asm/disasm checks once .chericap directives are supported.

@a = common addrspace(200) global [5 x i32] zeroinitializer, align 4
@b = addrspace(200) global [3 x ptr addrspace(200)] [
    ptr addrspace(200) getelementptr (i8, ptr addrspace(200) @a, i64 8),
    ptr addrspace(200) getelementptr (i8, ptr addrspace(200) @a, i64 4),
    ptr addrspace(200) @a
  ], align 32
@c = addrspace(200) constant [3 x ptr addrspace(200)] [
    ptr addrspace(200) getelementptr (i8, ptr addrspace(200) @a, i64 16),
    ptr addrspace(200) getelementptr (i8, ptr addrspace(200) @a, i64 12),
    ptr addrspace(200) @a
  ], align 32
