; REQUIRES: mips-registered-target
;; Check that we infer the correct datalayout from a target triple
; RUN: opt -mtriple=mips64-- -S -passes=no-op-module -target-abi=n32 < %s | FileCheck -check-prefix=N32 %s
; RUN: opt -mtriple=mips64-- -S -passes=no-op-module -target-abi=n64 < %s | FileCheck -check-prefix=N64 %s

target datalayout = ""

; N32: target datalayout = "E-m:e-p:32:32-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
; N64: target datalayout = "E-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
