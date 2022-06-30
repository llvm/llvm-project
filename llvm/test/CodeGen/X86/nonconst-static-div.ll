; RUN: not --crash llc -mtriple=i686-linux-gnu < %s 2>&1 | FileCheck %s

; Targets only support a limited set of relocations. Make sure that unusual
; constant expressions (and in particular potentially trapping ones involving
; division) are already rejected when lowering the Constant to the MC layer,
; rather than only when trying to emit a relocation. This makes sure that an
; error is reported when targeting assembly (without -filetype=obj).

@g = external global i32
@g2 = global i64 sdiv (i64 3, i64 ptrtoint (ptr @g to i64))

; CHECK: Unsupported expression in static initializer: sdiv
