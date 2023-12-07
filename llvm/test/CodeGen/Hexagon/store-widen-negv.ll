; RUN: llc -march=hexagon < %s | FileCheck %s
; We shouldn't see a 32-bit expansion of -120, just the uint8 value.
; CHECK: #136
define i32 @foo(ptr %ptr) {
entry:
  %msb = getelementptr inbounds [4 x i8], ptr %ptr, i32 0, i32 3
  %lsb = getelementptr inbounds [4 x i8], ptr %ptr, i32 0, i32 2
  store i8 0, ptr %msb
  store i8 -120, ptr %lsb, align 2
  ret i32 0
}
