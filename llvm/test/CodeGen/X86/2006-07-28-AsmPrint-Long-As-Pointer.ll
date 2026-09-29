; RUN: llc < %s -mtriple=i686-- | FileCheck %s
; PR853

; CHECK: 4294967240
@X = global ptr inttoptr (i64 -56 to ptr)		; <ptr> [#uses=0]

