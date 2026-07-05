; RUN: llc < %s -mtriple=i686-- | FileCheck %s --check-prefix=i686
; RUN: llc < %s -mtriple=i386-apple-darwin10 | FileCheck %s --check-prefix=DARWIN
; RUN: llc < %s -mtriple=i386-apple-darwin9 | FileCheck %s --check-prefix=DARWIN
; RUN: llc < %s -mtriple=i386-apple-darwin8 | FileCheck %s --check-prefix=DARWIN8

; i686: array,16512
; DARWIN8: array,16512
; DARWIN: array,16512,7

; Darwin 9+ should get alignment on common symbols.
@array = common global [4128 x i32] zeroinitializer, align 128
