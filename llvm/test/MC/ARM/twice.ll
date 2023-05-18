; Check for state persistence bugs in the ARM MC backend
; This should neither fail (in the comparison that the second object
; is bit-identical to the first) nor crash. Either failure would most
; likely indicate some state that is not properly reset in the
; appropriate ::reset method.
; RUN: llc -compile-twice -filetype=obj %s -o -

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv4t-unknown-linux-gnueabi"
