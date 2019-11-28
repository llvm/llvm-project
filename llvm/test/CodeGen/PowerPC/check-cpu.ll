; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:     -mcpu=future < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:     -mcpu=future < %s | FileCheck %s


; Test mcpu=future that should be recognized on PowerPC.

; CHECK-NOT: is not a recognized processor for this target
; CHECK:     .text

