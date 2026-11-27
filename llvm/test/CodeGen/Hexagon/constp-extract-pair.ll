; Verify constant propagation of 64-bit extract operations. The internal left
; shift used to position the bitfield must use unsigned arithmetic to avoid
; undefined behavior when high bits in the source value cause signed overflow.
; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s

target triple = "hexagon"

@g0 = common global i64 0, align 8
@g1 = common global i64 0, align 8

; extractup(0xC00000, 15, 8): extract 15 unsigned bits at offset 8.
; The source value 0xC00000 has bit 23 set (outside the [8,23) extraction
; window), which triggers a left shift of the value by 41 that would overflow
; int64_t. Bits [22:8] = 0x4000 = 16384.
define void @test_extractup() #0 {
; CHECK-LABEL: test_extractup:
; CHECK: 16384
; CHECK-NOT: = extractu
entry:
  %0 = call i64 @llvm.hexagon.S2.extractup(i64 12582912, i32 15, i32 8)
  store i64 %0, ptr @g0, align 8
  ret void
}

; extractp(0xC00000, 15, 8): extract 15 signed bits at offset 8.
; Bits [22:8] = 0x4000; bit 14 of the 15-bit field is set (sign bit),
; so the result is sign-extended to -16384.
define void @test_extractp() #0 {
; CHECK-LABEL: test_extractp:
; CHECK: -16384
; CHECK-NOT: = extract
entry:
  %0 = call i64 @llvm.hexagon.S4.extractp(i64 12582912, i32 15, i32 8)
  store i64 %0, ptr @g1, align 8
  ret void
}

declare i64 @llvm.hexagon.S2.extractup(i64, i32, i32) #1
declare i64 @llvm.hexagon.S4.extractp(i64, i32, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" }
attributes #1 = { nounwind readnone }
