; RUN: llc < %s -mtriple=i686-- | FileCheck %s
; PR853

; CHECK: 4294967240
@X = global i32* inttoptr (i64 -56 to i32*)		; <i32**> [#uses=0]

