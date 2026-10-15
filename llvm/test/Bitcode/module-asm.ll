; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: module asm
; CHECK:     "asm line 0"
; CHECK: module asm(target_features: "+foo", target_cpu: "foo_cpu")
; CHECK:     "asm line 1"
; CHECK:     "asm line 2"
; CHECK: module asm(target_features: "+bar")
; CHECK:     "asm line 3"
; CHECK: module asm(target_cpu: "bar_cpu")
; CHECK:     "asm line 4"
; CHECK: module asm
; CHECK:     "asm line 5"

module asm
    "asm line 0"

module asm(target_features: "+foo", target_cpu: "foo_cpu")
    "asm line 1"
    "asm line 2"

module asm(target_features: "+bar")
    "asm line 3"

module asm(target_cpu: "bar_cpu")
    "asm line 4"

module asm
    "asm line 5"
