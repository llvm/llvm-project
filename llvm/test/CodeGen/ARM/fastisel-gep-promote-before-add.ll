; fastisel should not fold add with non-pointer bitwidth
; sext(a) + sext(b) != sext(a + b)
; RUN: llc -mtriple=armv7-apple-ios %s -O0 -o - | FileCheck %s

define zeroext i8 @gep_promotion(ptr %ptr) nounwind uwtable ssp {
entry:
  %ptr.addr = alloca ptr, align 8
  %add = add i8 64, 64 ; 0x40 + 0x40
  %0 = load ptr, ptr %ptr.addr, align 8

  ; CHECK-LABEL: _gep_promotion:
  ; CHECK: ldrb {{r[0-9]+}}, {{\[r[0-9]+\]}}
  %arrayidx = getelementptr inbounds i8, ptr %0, i8 %add

  %1 = load i8, ptr %arrayidx, align 1
  ret i8 %1
}

