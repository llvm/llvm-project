; RUN: llc %s -mtriple=x86_64-unknown-fuchsia  -o - | FileCheck %s

@vtable = dso_local unnamed_addr constant i32 trunc (i64 sub (i64 ptrtoint (ptr @rtti.proxy to i64), i64 ptrtoint (ptr @vtable to i64)) to i32), align 4
@vtable_with_offset = dso_local unnamed_addr constant [2 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @rtti.proxy to i64), i64 ptrtoint (ptr @vtable_with_offset to i64)) to i32)], align 4
@vtable_with_negative_offset = dso_local unnamed_addr constant [2 x i32] [
  i32 trunc (
    i64 sub (
      i64 ptrtoint (ptr @rtti.proxy to i64),
      i64 ptrtoint (ptr getelementptr inbounds ([2 x i32], ptr @vtable_with_negative_offset, i32 0, i32 1) to i64)
    )
    to i32),
  i32 0
], align 4
@rtti = external global i8, align 8
@rtti.proxy = linkonce_odr hidden unnamed_addr constant ptr @rtti

; CHECK-NOT: rtti.proxy
; CHECK-LABEL: vtable:
; CHECK-NEXT:    .{{word|long}}   rtti@GOTPCREL+0{{$}}

; CHECK-LABEL: vtable_with_offset:
; CHECK-NEXT:    .{{word|long}}   0
; CHECK-NEXT:    .{{word|long}}   rtti@GOTPCREL+4{{$}}

; CHECK-LABEL: vtable_with_negative_offset:
; CHECK-NEXT:    .{{word|long}}   rtti@GOTPCREL-4{{$}}
; CHECK-NEXT:    .{{word|long}}   0
