; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 -verify-machineinstrs | FileCheck %s

; Test the same constant can be used by different stores.

%struct.S = type { i64, i8, i16, i32 }

define void @foo(%struct.S* %p) {
  %l4 = bitcast %struct.S* %p to i64*
  store i64 0, i64* %l4, align 8
  %c = getelementptr %struct.S, %struct.S* %p, i64 0, i32 1
  store i8 0, i8* %c, align 8
  %s = getelementptr %struct.S, %struct.S* %p, i64 0, i32 2
  store i16 0, i16* %s, align 2
  %i = getelementptr %struct.S, %struct.S* %p, i64 0, i32 3
  store i32 0, i32* %i, align 4
  ret void

; CHECK-LABEL: @foo
; CHECK:       li 4, 0
; CHECK:       stb 4, 8(3)
; CHECK:       std 4, 0(3)
; CHECK:       sth 4, 10(3)
; CHECK:       stw 4, 12(3)
}

define void @bar(%struct.S* %p) {
  %i = getelementptr %struct.S, %struct.S* %p, i64 0, i32 3
  store i32 2, i32* %i, align 4
  %s = getelementptr %struct.S, %struct.S* %p, i64 0, i32 2
  store i16 2, i16* %s, align 2
  %c = getelementptr %struct.S, %struct.S* %p, i64 0, i32 1
  store i8 2, i8* %c, align 8
  %l4 = bitcast %struct.S* %p to i64*
  store i64 2, i64* %l4, align 8
  ret void

; CHECK-LABEL: @bar
; CHECK:       li 4, 2
; CHECK:       stw 4, 12(3)
; CHECK:       sth 4, 10(3)
; CHECK:       std 4, 0(3)
; CHECK:       stb 4, 8(3)
}

