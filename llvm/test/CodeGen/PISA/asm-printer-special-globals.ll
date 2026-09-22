; RUN: split-file %s %t
; RUN: llc -mtriple=pisa -filetype=asm %t/used.ll -o - | FileCheck --check-prefix=USED %s
; RUN: not llc -mtriple=pisa -filetype=asm %t/ctors.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CTORS %s
; RUN: not llc -mtriple=pisa -filetype=asm %t/dtors.ll -o /dev/null 2>&1 | FileCheck --check-prefix=DTORS %s
; RUN: not llc -mtriple=pisa -filetype=asm %t/unknown.ll -o /dev/null 2>&1 | FileCheck --check-prefix=UNKNOWN %s

; USED-NOT: @llvm.used
; USED-NOT: @llvm.compiler.used
; USED: @data =
; USED-NOT: @llvm.used
; USED-NOT: @llvm.compiler.used
; CTORS: LLVM ERROR: llvm.global_ctors is not supported by the PISA backend
; DTORS: LLVM ERROR: llvm.global_dtors is not supported by the PISA backend
; UNKNOWN: LLVM ERROR: unknown special variable with appending linkage

;--- used.ll
target triple = "pisa"

@data = addrspace(1) global i32 7
@llvm.used = appending global [1 x ptr addrspace(1)] [ptr addrspace(1) @data]
@llvm.compiler.used = appending global [1 x ptr addrspace(1)] [ptr addrspace(1) @data], section "llvm.metadata"

;--- ctors.ll
target triple = "pisa"

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 65535, ptr @init, ptr null }
]

define void @init() {
  ret void
}

;--- dtors.ll
target triple = "pisa"

@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 65535, ptr @fini, ptr null }
]

define void @fini() {
  ret void
}

;--- unknown.ll
target triple = "pisa"

@mystery = appending global [1 x i8] c"x"
