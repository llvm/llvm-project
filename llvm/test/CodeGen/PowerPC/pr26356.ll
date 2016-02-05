; RUN: llc -O0 -mcpu=pwr7 -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s

define zeroext i32 @f1() {
entry:
  ret i32 65535
}
; CHECK-LABEL: @f1
; CHECK: lis 3, 0
; CHECK: ori 3, 3, 65535

define zeroext i32 @f2() {
entry:
  ret i32 32768
}
; CHECK-LABEL: @f2
; CHECK: lis 3, 0
; CHECK: ori 3, 3, 32768

define zeroext i32 @f3() {
entry:
  ret i32 32767
}
; CHECK-LABEL: @f3
; CHECK: li 3, 32767

define zeroext i16 @f4() {
entry:
  ret i16 65535
}
; CHECK-LABEL: @f4
; CHECK: lis 3, 0
; CHECK: ori 3, 3, 65535

define zeroext i16 @f5() {
entry:
  ret i16 32768
}
; CHECK-LABEL: @f5
; CHECK: lis 3, 0
; CHECK: ori 3, 3, 32768

define zeroext i16 @f6() {
entry:
  ret i16 32767
}
; CHECK-LABEL: @f6
; CHECK: li 3, 32767

define zeroext i16 @f7() {
entry:
  ret i16 -1
}
; CHECK-LABEL: @f7
; CHECK: lis 3, 0
; CHECK: ori 3, 3, 65535

define zeroext i16 @f8() {
entry:
  ret i16 -32768
}
; CHECK-LABEL: @f8
; CHECK: lis 3, 0
; CHECK: ori 3, 3, 32768

define signext i32 @f1s() {
entry:
  ret i32 65535
}
; CHECK-LABEL: @f1s
; CHECK: lis 3, 0
; CHECK: ori 3, 3, 65535

define signext i32 @f2s() {
entry:
  ret i32 32768
}
; CHECK-LABEL: @f2s
; CHECK: lis 3, 0
; CHECK: ori 3, 3, 32768

define signext i32 @f3s() {
entry:
  ret i32 32767
}
; CHECK-LABEL: @f3s
; CHECK: li 3, 32767

define signext i16 @f4s() {
entry:
  ret i16 32767
}
; CHECK-LABEL: @f4s
; CHECK: li 3, 32767

define signext i32 @f1sn() {
entry:
  ret i32 -65535
}
; CHECK-LABEL: @f1sn
; CHECK: lis 3, -1
; CHECK: ori 3, 3, 1

define signext i32 @f2sn() {
entry:
  ret i32 -32768
}
; CHECK-LABEL: @f2sn
; CHECK: li 3, -32768

define signext i32 @f3sn() {
entry:
  ret i32 -32767
}
; CHECK-LABEL: @f3sn
; CHECK: li 3, -32767

define signext i32 @f4sn() {
entry:
  ret i32 -65536
}
; CHECK-LABEL: @f4sn
; CHECK: lis 3, -1

define signext i16 @f5sn() {
entry:
  ret i16 -32767
}
; CHECK-LABEL: @f5sn
; CHECK: li 3, -32767

define signext i16 @f6sn() {
entry:
  ret i16 -32768
}
; CHECK-LABEL: @f6sn
; CHECK: li 3, -32768
