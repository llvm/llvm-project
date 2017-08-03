; RUN: llc < %s -march=x86-64 -mcpu=corei7 -mtriple=x86_64-pc-win64 | FileCheck %s

; CHECK: multiple_stores_on_chain
; CHECK: movabsq
; CHECK: movq
; CHECK: movabsq
; CHECK: movq
; CHECK: ret
define void @multiple_stores_on_chain(i16 * %A) {
entry:
  %a0 = getelementptr inbounds i16, i16* %A, i64 0
  %a1 = getelementptr inbounds i16, i16* %A, i64 1
  %a2 = getelementptr inbounds i16, i16* %A, i64 2
  %a3 = getelementptr inbounds i16, i16* %A, i64 3
  %a4 = getelementptr inbounds i16, i16* %A, i64 4
  %a5 = getelementptr inbounds i16, i16* %A, i64 5
  %a6 = getelementptr inbounds i16, i16* %A, i64 6
  %a7 = getelementptr inbounds i16, i16* %A, i64 7

  store i16 0, i16* %a0
  store i16 1, i16* %a1
  store i16 2, i16* %a2
  store i16 3, i16* %a3
  store i16 4, i16* %a4
  store i16 5, i16* %a5
  store i16 6, i16* %a6
  store i16 7, i16* %a7

  ret void
}

