; RUN: llc -mtriple=aarch64 < %s | FileCheck %s --match-full-lines

declare void @extern_func()

; CHECK-LABEL: const:
; CHECK-NEXT:    .word   %pltpcrel(extern_func)

@const = dso_local constant i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @extern_func to i64), i64 ptrtoint (ptr @const to i64)) to i32)

@_ZTV1B = dso_local constant { [7 x i32] } { [7 x i32] [
  i32 0,
  i32 0,
  i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @f0 to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [7 x i32] }, ptr @_ZTV1B, i32 0, i32 0, i32 2) to i64)) to i32),
  i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @f1 to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [7 x i32] }, ptr @_ZTV1B, i32 0, i32 0, i32 2) to i64)) to i32),
  i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @f2 to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [7 x i32] }, ptr @_ZTV1B, i32 0, i32 0, i32 2) to i64)) to i32),
  i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @f3 to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [7 x i32] }, ptr @_ZTV1B, i32 0, i32 0, i32 2) to i64)) to i32),
  i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @f4 to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [7 x i32] }, ptr @_ZTV1B, i32 0, i32 0, i32 2) to i64)) to i32)
] }, align 4

; CHECK-LABEL: _ZTV1B:
; CHECK-NEXT:    .word   0                               // 0x0
; CHECK-NEXT:    .word   0                               // 0x0
; CHECK-NEXT:    .word   %pltpcrel(f0)
; CHECK-NEXT:    .word   %pltpcrel(f1+4)
; CHECK-NEXT:    .word   %pltpcrel(f2+8)
; CHECK-NEXT:    .word   %pltpcrel(f3+12)
; CHECK-NEXT:    .word   %pltpcrel(f4+16)
; CHECK-NEXT:    .size   _ZTV1B, 28
declare void @f0()
declare void @f1()
define dso_local void @f2() {
  ret void
}
define void @f3() {
  ret void
}
define hidden void @f4() {
  ret void
}
