; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-mem2reg=false
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll

; RUN: llc -filetype=obj -O0 < %t.ll -mtriple x86_64-none-linux | \
; RUN:     llvm-dwarfdump - 2>&1 | FileCheck %s
; RUN: llc -filetype=obj -O0 < %t.ll -mtriple i386-none-linux | \
; RUN:     llvm-dwarfdump - 2>&1 | FileCheck %s
; RUN: llc -filetype=obj -O0 < %t.ll -mtriple x86_64-none-mingw32 | \
; RUN:     llvm-dwarfdump - 2>&1 | FileCheck %s
; RUN: llc -filetype=obj -O0 < %t.ll -mtriple i386-none-mingw32 | \
; RUN:     llvm-dwarfdump - 2>&1 | FileCheck %s

; CHECK-NOT: failed to compute relocation

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(file: !1, language: DW_LANG_C99, producer: "clang version 3.6.0 ", isOptimized: false, enums: !2, retainedTypes: !2, globals: !2, imports: !2, emissionKind: FullDebug)
!1 = !DIFile(filename: "empty.c", directory: "/a")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.6.0 "}
target triple = "spir64-unknown-unknown"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
