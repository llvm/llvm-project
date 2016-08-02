; RUN: llc -filetype=obj -O0 < %s -mtriple x86_64-none-linux | \
; RUN:     llvm-dwarfdump - 2>&1 | FileCheck %s
; RUN: llc -filetype=obj -O0 < %s -mtriple i386-none-linux | \
; RUN:     llvm-dwarfdump - 2>&1 | FileCheck %s
; RUN: llc -filetype=obj -O0 < %s -mtriple x86_64-none-mingw32 | \
; RUN:     llvm-dwarfdump - 2>&1 | FileCheck %s
; RUN: llc -filetype=obj -O0 < %s -mtriple i386-none-mingw32 | \
; RUN:     llvm-dwarfdump - 2>&1 | FileCheck %s

; CHECK-NOT: failed to compute relocation

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(file: !1, language: DW_LANG_C99, producer: "clang version 3.6.0 ", isOptimized: false, enums: !2, retainedTypes: !2, subprograms: !2, globals: !2, imports: !2, emissionKind: 1)
!1 = !DIFile(filename: "empty.c", directory: "/a")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.6.0 "}
