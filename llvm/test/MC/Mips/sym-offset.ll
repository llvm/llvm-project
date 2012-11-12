; DISABLED: llc -filetype=obj -mtriple mipsel-unknown-linux %s -o - | elf-dump --dump-section-data  | FileCheck %s
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

; CHECK: ('_section_data', '00001c3c 00009c27 21e09903 0000828f 0e004188 0b004198

  %call = tail call i32 @memcmp(i8* getelementptr inbounds ([11 x i8]* @string1, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @string2, i32 0, i32 0), i32 4) nounwind readonly
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

declare i32 @memcmp(i8* nocapture, i8* nocapture, i32) nounwind readonly
