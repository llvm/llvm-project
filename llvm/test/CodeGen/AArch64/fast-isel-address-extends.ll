; RUN: llc %s -o - -O0 -verify-machineinstrs -fast-isel=true | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios8.0.0"

; This test was trying to fold the sext %tmp142 in to the address arithmetic in %sunkaddr1.
; This was incorrect as %.mux isn't available in the last bb.

; CHECK: sxtw [[REG0:x[0-9]+]]
; CHECK: str [[REG0]], [sp, [[OFFSET:#[0-9]+]]]
; CHECK: ldr [[REG1:x[0-9]+]], [sp, [[OFFSET]]]
; CHECK: strh wzr, [{{.*}}, [[REG1]], lsl #1]

; Function Attrs: nounwind optsize ssp
define void @EdgeLoop(i32 %dir, i32 %edge, i32 %width, i16* %tmp89, i32 %tmp136, i16 %tmp144) #0 {
bb:
  %tmp2 = icmp eq i32 %dir, 0
  %.mux = select i1 %tmp2, i32 %width, i32 1
  %tmp142 = sext i32 %.mux to i64
  %tmp151 = shl nsw i64 %tmp142, 1
  %tmp153 = getelementptr inbounds i16, i16* %tmp89, i64 %tmp151
  %tmp154 = load i16, i16* %tmp153, align 2
  %tmp155 = zext i16 %tmp154 to i32
  br i1 %tmp2, label %bb225, label %bb212

bb212:                                            ; preds = %bb
  store i16 %tmp144, i16* %tmp89, align 2
  ret void

bb225:                                            ; preds = %bb
  %tmp248 = trunc i32 %tmp155 to i16
  store i16 %tmp248, i16* %tmp89, align 2
  %sunkaddr = ptrtoint i16* %tmp89 to i64
  %sunkaddr1 = mul i64 %tmp142, 2
  %sunkaddr2 = add i64 %sunkaddr, %sunkaddr1
  %sunkaddr3 = inttoptr i64 %sunkaddr2 to i16*
  store i16 0, i16* %sunkaddr3, align 2
  ret void
}

attributes #0 = { nounwind optsize ssp }
