; RUN: llc < %s -mtriple=powerpc-linux-gnu | FileCheck %s -check-prefix=ELF
; RUN: llc < %s -mtriple=powerpc-apple-darwin9 | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -mtriple=powerpc-apple-darwin8 | FileCheck %s -check-prefix=DARWIN8

@a = global i1 true
; no alignment

@b = global i8 1
; no alignment

@c = global i16 2
;ELF: .align 1
;ELF: c:
;DARWIN: .align 1
;DARWIN: _c:

@d = global i32 3
;ELF: .align 2
;ELF: d:
;DARWIN: .align 2
;DARWIN: _d:

@e = global i64 4
;ELF: .align 3
;ELF: e
;DARWIN: .align 3
;DARWIN: _e:

@f = global float 5.0
;ELF: .align 2
;ELF: f:
;DARWIN: .align 2
;DARWIN: _f:

@g = global double 6.0
;ELF: .align 3
;ELF: g:
;DARWIN: .align 3
;DARWIN: _g:

@bar = common global [75 x i8] zeroinitializer, align 128
;ELF: .comm bar,75,128
;DARWIN: .comm _bar,75,7

;; Darwin8 doesn't support aligned comm.  Just miscompile this.
; DARWIN8: .comm _bar,75 ;
