; RUN: llc -mtriple=x86_64-w64-mingw32 < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=x86_64-w64-mingw32 -filetype=obj < %s | llvm-objdump -r - | FileCheck %s --check-prefix=OBJ
; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s --check-prefix=ASM

; A difference from __ImageBase is an image-relative reference on mingw as it
; is on MSVC.

@__ImageBase = external dso_local constant i8
@target = dso_local global i32 1
@data = dso_local constant i32 trunc (i64 sub (i64 ptrtoint (ptr @target to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32)

; ASM: data:
; ASM-NEXT: .long target@IMGREL

; OBJ: IMAGE_REL_AMD64_ADDR32NB target
