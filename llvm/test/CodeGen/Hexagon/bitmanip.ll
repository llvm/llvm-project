; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: popcount_16
; CHECK: zxth
; CHECK: popcount
define i16 @popcount_16(i16 %p) #0 {
  %t = call i16 @llvm.ctpop.i16(i16 %p) #0
  ret i16 %t
}

; CHECK-LABEL: popcount_32
; CHECK: popcount
define i32 @popcount_32(i32 %p) #0 {
  %t = call i32 @llvm.ctpop.i32(i32 %p) #0
  ret i32 %t
}

; CHECK-LABEL: popcount_64
; CHECK: popcount
define i64 @popcount_64(i64 %p) #0 {
  %t = call i64 @llvm.ctpop.i64(i64 %p) #0
  ret i64 %t
}

; CHECK-LABEL: ctlz_16
; CHECK: [[REG0:r[0-9]+]] = zxth
; CHECK: [[REG1:r[0-9]+]] = cl0([[REG0]])
; CHECK: add([[REG1]],#-16)
define i16 @ctlz_16(i16 %p) #0 {
  %t = call i16 @llvm.ctlz.i16(i16 %p, i1 true) #0
  ret i16 %t
}

; CHECK-LABEL: ctlz_32
; CHECK: cl0
define i32 @ctlz_32(i32 %p) #0 {
  %t = call i32 @llvm.ctlz.i32(i32 %p, i1 true) #0
  ret i32 %t
}

; CHECK-LABEL: ctlz_64
; CHECK: cl0
define i64 @ctlz_64(i64 %p) #0 {
  %t = call i64 @llvm.ctlz.i64(i64 %p, i1 true) #0
  ret i64 %t
}

; CHECK-LABEL: cttz_16
; CHECK: ct0
define i16 @cttz_16(i16 %p) #0 {
  %t = call i16 @llvm.cttz.i16(i16 %p, i1 true) #0
  ret i16 %t
}

; CHECK-LABEL: cttz_32
; CHECK: ct0
define i32 @cttz_32(i32 %p) #0 {
  %t = call i32 @llvm.cttz.i32(i32 %p, i1 true) #0
  ret i32 %t
}

; CHECK-LABEL: cttz_64
; CHECK: ct0
define i64 @cttz_64(i64 %p) #0 {
  %t = call i64 @llvm.cttz.i64(i64 %p, i1 true) #0
  ret i64 %t
}

; CHECK-LABEL: brev_16
; CHECK: [[REG:r[0-9]+]] = brev
; CHECK: lsr([[REG]],#16)
define i16 @brev_16(i16 %p) #0 {
  %t = call i16 @llvm.bitreverse.i16(i16 %p) #0
  ret i16 %t
}

; CHECK-LABEL: brev_32
; CHECK: brev
define i32 @brev_32(i32 %p) #0 {
  %t = call i32 @llvm.bitreverse.i32(i32 %p) #0
  ret i32 %t
}

; CHECK-LABEL: brev_64
; CHECK: brev
define i64 @brev_64(i64 %p) #0 {
  %t = call i64 @llvm.bitreverse.i64(i64 %p) #0
  ret i64 %t
}

; CHECK-LABEL: bswap_16
; CHECK: [[REG:r[0-9]+]] = swiz
; CHECK: lsr([[REG]],#16)
define i16 @bswap_16(i16 %p) #0 {
  %t = call i16 @llvm.bswap.i16(i16 %p) #0
  ret i16 %t
}

; CHECK-LABEL: bswap_32
; CHECK: swiz
define i32 @bswap_32(i32 %p) #0 {
  %t = call i32 @llvm.bswap.i32(i32 %p) #0
  ret i32 %t
}

; CHECK-LABEL: bswap_64
; CHECK: swiz
; CHECK: swiz
; CHECK: combine
define i64 @bswap_64(i64 %p) #0 {
  %t = call i64 @llvm.bswap.i64(i64 %p) #0
  ret i64 %t
}

declare i16 @llvm.ctpop.i16(i16) #0
declare i32 @llvm.ctpop.i32(i32) #0
declare i64 @llvm.ctpop.i64(i64) #0

declare i16 @llvm.ctlz.i16(i16, i1) #0
declare i32 @llvm.ctlz.i32(i32, i1) #0
declare i64 @llvm.ctlz.i64(i64, i1) #0

declare i16 @llvm.cttz.i16(i16, i1) #0
declare i32 @llvm.cttz.i32(i32, i1) #0
declare i64 @llvm.cttz.i64(i64, i1) #0

declare i16 @llvm.bitreverse.i16(i16) #0
declare i32 @llvm.bitreverse.i32(i32) #0
declare i64 @llvm.bitreverse.i64(i64) #0

declare i16 @llvm.bswap.i16(i16) #0
declare i32 @llvm.bswap.i32(i32) #0
declare i64 @llvm.bswap.i64(i64) #0

attributes #0 = { nounwind readnone }
