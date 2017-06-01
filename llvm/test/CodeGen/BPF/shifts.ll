; RUN: llc < %s -march=bpfel -show-mc-encoding | FileCheck %s

define zeroext i8 @lshr8(i8 zeroext %a, i8 zeroext %cnt) nounwind readnone {
entry:
; CHECK-LABEL: lshr8:
; CHECK: r1 >>= r2 # encoding: [0x7f,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
  %shr = lshr i8 %a, %cnt
  ret i8 %shr
}

define signext i8 @ashr8(i8 signext %a, i8 zeroext %cnt) nounwind readnone {
entry:
; CHECK-LABEL: ashr8:
; CHECK: r1 s>>= r2 # encoding: [0xcf,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
  %shr = ashr i8 %a, %cnt
  ret i8 %shr
}

define zeroext i8 @shl8(i8 zeroext %a, i8 zeroext %cnt) nounwind readnone {
entry:
; CHECK: shl8
; CHECK: r1 <<= r2 # encoding: [0x6f,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
  %shl = shl i8 %a, %cnt
  ret i8 %shl
}

define zeroext i16 @lshr16(i16 zeroext %a, i16 zeroext %cnt) nounwind readnone {
entry:
; CHECK-LABEL: lshr16:
; CHECK: r1 >>= r2 # encoding: [0x7f,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
  %shr = lshr i16 %a, %cnt
  ret i16 %shr
}

define signext i16 @ashr16(i16 signext %a, i16 zeroext %cnt) nounwind readnone {
entry:
; CHECK-LABEL: ashr16:
; CHECK: r1 s>>= r2 # encoding: [0xcf,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
  %shr = ashr i16 %a, %cnt
  ret i16 %shr
}

define zeroext i16 @shl16(i16 zeroext %a, i16 zeroext %cnt) nounwind readnone {
entry:
; CHECK-LABEL: shl16:
; CHECK: r1 <<= r2 # encoding: [0x6f,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
  %shl = shl i16 %a, %cnt
  ret i16 %shl
}

define zeroext i32 @lshr32(i32 zeroext %a, i32 zeroext %cnt) nounwind readnone {
entry:
; CHECK-LABEL: lshr32:
; CHECK: r1 >>= r2 # encoding: [0x7f,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK: r1 <<= 32 # encoding: [0x67,0x01,0x00,0x00,0x20,0x00,0x00,0x00]
  %shr = lshr i32 %a, %cnt
  ret i32 %shr
}

define signext i32 @ashr32(i32 signext %a, i32 zeroext %cnt) nounwind readnone {
entry:
; CHECK-LABEL: ashr32:
; CHECK: r1 s>>= r2 # encoding: [0xcf,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
  %shr = ashr i32 %a, %cnt
  ret i32 %shr
}

define zeroext i32 @shl32(i32 zeroext %a, i32 zeroext %cnt) nounwind readnone {
entry:
; CHECK-LABEL: shl32:
; CHECK: r1 <<= r2 # encoding: [0x6f,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
  %shl = shl i32 %a, %cnt
  ret i32 %shl
}

define zeroext i64 @lshr64(i64 zeroext %a, i64 zeroext %cnt) nounwind readnone {
entry:
; CHECK-LABEL: lshr64:
; CHECK: r1 >>= r2 # encoding: [0x7f,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
  %shr = lshr i64 %a, %cnt
  ret i64 %shr
}

define signext i64 @ashr64(i64 signext %a, i64 zeroext %cnt) nounwind readnone {
entry:
; CHECK-LABEL: ashr64:
; CHECK: r1 s>>= r2 # encoding: [0xcf,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
  %shr = ashr i64 %a, %cnt
  ret i64 %shr
}

define zeroext i64 @shl64(i64 zeroext %a, i64 zeroext %cnt) nounwind readnone {
entry:
; CHECK-LABEL: shl64:
; CHECK: r1 <<= r2 # encoding: [0x6f,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK: r0 = r1 # encoding: [0xbf,0x10,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK: exit # encoding: [0x95,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
  %shl = shl i64 %a, %cnt
  ret i64 %shl
}
