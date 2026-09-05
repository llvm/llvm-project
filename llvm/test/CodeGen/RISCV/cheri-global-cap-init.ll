; RUN: llc -mtriple=riscv32 -target-abi il32pc64 -mattr=+experimental-y %s -o /dev/null
; RUN: llc -mtriple=riscv32 -target-abi il32pc64 -mattr=+experimental-y -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv64 -target-abi l64pc128 -mattr=+experimental-y %s -o /dev/null
; RUN: llc -mtriple=riscv64 -target-abi l64pc128 -mattr=+experimental-y -filetype=obj %s -o /dev/null

; Previously crashed due to using pointer size rather than index size when computing the initializer of @b

@a = external addrspace(200) global i32
@b = addrspace(200) global i32 ptrtoint (ptr addrspace(200) getelementptr (i8, ptr addrspace(200) @a, i64 8) to i32)
