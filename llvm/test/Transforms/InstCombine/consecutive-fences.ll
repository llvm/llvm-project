; RUN: opt -instcombine -S %s | FileCheck %s

; Make sure we collapse the fences in this case

; CHECK-LABEL: define void @tinkywinky
; CHECK-NEXT:   fence seq_cst
; CHECK-NEXT:   fence syncscope("singlethread") acquire
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

define void @tinkywinky() {
  fence seq_cst
  fence seq_cst
  fence seq_cst
  fence syncscope("singlethread") acquire
  fence syncscope("singlethread") acquire
  fence syncscope("singlethread") acquire
  ret void
}

; CHECK-LABEL: define void @dipsy
; CHECK-NEXT:   fence seq_cst
; CHECK-NEXT:   fence syncscope("singlethread") seq_cst
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

define void @dipsy() {
  fence seq_cst
  fence syncscope("singlethread") seq_cst
  ret void
}

; CHECK-LABEL: define void @patatino
; CHECK-NEXT:   fence acquire
; CHECK-NEXT:   fence seq_cst
; CHECK-NEXT:   fence acquire
; CHECK-NEXT:   fence seq_cst
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

define void @patatino() {
  fence acquire
  fence seq_cst
  fence acquire
  fence seq_cst
  ret void
}
