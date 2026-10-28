; RUN: not llc < %s -mtriple=wasm32-unknown-unknown -filetype=obj -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc < %s -mtriple=wasm32-unknown-unknown -filetype=asm -o - 2>&1 | FileCheck %s --check-prefix=ASM

; Initializers that a Wasm init expression cannot represent are diagnosed.

; The declaration is still printed after the diagnostic.
; ASM: error: unsupported initializer for WebAssembly global 'narrow_int'
; ASM: .globl narrow_int
; ASM: narrow_int:

; CHECK: error: unsupported initializer for WebAssembly global 'narrow_int'
@narrow_int = addrspace(1) global i8 5

; CHECK: error: unsupported initializer for WebAssembly global 'narrow_bool'
@narrow_bool = addrspace(1) global i1 true

; CHECK: error: unsupported initializer for WebAssembly global 'short_int'
@short_int = addrspace(1) global i16 4660

; CHECK: error: unsupported initializer for WebAssembly global 'half_fp'
@half_fp = addrspace(1) global half 2.0

; CHECK: error: unsupported initializer for WebAssembly global 'bfloat_fp'
@bfloat_fp = addrspace(1) global bfloat 2.0

@data = global i32 7
; CHECK: error: unsupported initializer for WebAssembly global 'sym_addr'
@sym_addr = addrspace(1) global ptr @data

; CHECK: error: unsupported initializer for WebAssembly global 'sym_int'
@sym_int = addrspace(1) global i32 ptrtoint (ptr @data to i32)

@arr = global [4 x i32] zeroinitializer
; CHECK: error: unsupported initializer for WebAssembly global 'sym_offset'
@sym_offset = addrspace(1) global ptr getelementptr inbounds ([4 x i32], ptr @arr, i32 0, i32 2)
