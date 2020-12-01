; RUN: llc -O0 -fast-isel -fast-isel-abort=1 -verify-machineinstrs -mtriple=arm64-apple-darwin < %s | FileCheck %s

; Test load/store of global value from global offset table.
@seed = common global i64 0, align 8

define void @Initrand() nounwind {
entry:
; CHECK: @Initrand
; CHECK: adrp [[REG:x[0-9]+]], _seed@GOTPAGE
; CHECK: ldr  [[REG2:x[0-9]+]], {{\[}}[[REG]], _seed@GOTPAGEOFF{{\]}}
; CHECK: str  {{x[0-9]+}}, {{\[}}[[REG2]]{{\]}}
  store i64 74755, i64* @seed, align 8
  ret void
}

define i32 @Rand() nounwind {
entry:
; CHECK: @Rand
; CHECK: adrp [[REG1:x[0-9]+]], _seed@GOTPAGE
; CHECK: ldr  [[REG2:x[0-9]+]], {{\[}}[[REG1]], _seed@GOTPAGEOFF{{\]}}
; CHECK: ldr  [[REG5:x[0-9]+]], {{\[}}[[REG2]]{{\]}}
; CHECK: mov  [[REG4:x[0-9]+]], #1309
; CHECK: mul  [[REG6:x[0-9]+]], [[REG5]], [[REG4]]
; CHECK: mov  [[REG3:x[0-9]+]], #13849
; CHECK: add  [[REG7:x[0-9]+]], [[REG6]], [[REG3]]
; CHECK: and  [[REG8:x[0-9]+]], [[REG7]], #0xffff
; CHECK: adrp [[REG1:x[0-9]+]], _seed@GOTPAGE
; CHECK: ldr  [[REG1]], {{\[}}[[REG1]], _seed@GOTPAGEOFF{{\]}}
; CHECK: str  [[REG8]], {{\[}}[[REG1]]{{\]}}
; CHECK: adrp [[REG1:x[0-9]+]], _seed@GOTPAGE
; CHECK: ldr  [[REG1]], {{\[}}[[REG1]], _seed@GOTPAGEOFF{{\]}}
; CHECK: ldr  {{x[0-9]+}}, {{\[}}[[REG1]]{{\]}}
  %0 = load i64, i64* @seed, align 8
  %mul = mul nsw i64 %0, 1309
  %add = add nsw i64 %mul, 13849
  %and = and i64 %add, 65535
  store i64 %and, i64* @seed, align 8
  %1 = load i64, i64* @seed, align 8
  %conv = trunc i64 %1 to i32
  ret i32 %conv
}
