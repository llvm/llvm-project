; DISABLED: llc -filetype=obj -mtriple mipsel-unknown-linux %s -o - | llvm-readobj -S --sd | FileCheck %s
; RUN: false
; XFAIL: *

; FIXME: use assembler instead of llc when it becomes available.

@string1 = internal global [11 x i8] c"aaaaaaaaaa\00", align 1
@string2 = internal global [10 x i8] c"aaaa\00bbbb\00", align 1

define i32 @foo1(i32 %n) nounwind readonly {
entry:
; check that the immediate fields of lwl and lwr are three apart.
; 8841000e        lwl     at,14(v0)
; 9841000b        lwr     at,11(v0)

; CHECK:          SectionData (
; CHECK:            0000: 00001C3C 00009C27 21E09903 0000828F
; CHECK-NEXT:       0010: 0E004188 0B004198

  %call = tail call i32 @memcmp(ptr @string1, ptr @string2, i32 4) nounwind readonly
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

declare i32 @memcmp(ptr nocapture, ptr nocapture, i32) nounwind readonly
