; RUN: llc < %s -mtriple=armv7-apple-darwin | FileCheck %s

; CHECK: .section	__TEXT,__text,regular,pure_instructions
; CHECK: .section	__TEXT,myprecious
; CHECK: .section	__TEXT,__textcoal_nt,coalesced,pure_instructions
; CHECK: .section	__TEXT,__const_coal,coalesced
; CHECK: .section	__TEXT,__picsymbolstub4,symbol_stubs,none,16
; CHECK: .section	__TEXT,__StaticInit,regular,pure_instructions


define void @normal() nounwind readnone {
; CHECK: .section	__TEXT,__text,regular,pure_instructions
; CHECK: _normal:
  ret void
}

define void @special() nounwind readnone section "__TEXT,myprecious" {
; CHECK: .section	__TEXT,myprecious
; CHECK: _special:
  ret void
}
