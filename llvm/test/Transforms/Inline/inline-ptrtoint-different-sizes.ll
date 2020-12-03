; RUN: opt < %s -inline -S | FileCheck %s

; InlineCost used to have problems with the ptrtoint, leading to
; crashes when visiting the trunc in pr47969_help and the icmp in
; pr38500_help.

target datalayout = "p:16:16"
target triple = "x86_64-unknown-linux-gnu"

define void @pr47969_help(i16* %p) {
  %cast = ptrtoint i16* %p to i32
  %sub = sub i32 %cast, %cast
  %conv = trunc i32 %sub to i16
  ret void
}

define void @pr47969(i16* %x) {
  call void @pr47969_help(i16* %x)
  ret void
}

; CHECK-LABEL: @pr47969(i16* %x)
; CHECK-NOT:     call
; CHECK:         ret void

define void @pr38500_help(i16* %p) {
  %cast = ptrtoint i16* %p to i32
  %sub = sub i32 %cast, %cast
  %cmp = icmp eq i32 %sub, 0
  ret void
}

define void @pr38500(i16* %x) {
  call void @pr38500_help(i16* %x)
  ret void
}

; CHECK-LABEL: @pr38500(i16* %x)
; CHECK-NOT:     call
; CHECK:         ret void
