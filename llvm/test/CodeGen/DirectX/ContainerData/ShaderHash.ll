; RUN: opt %s -dxil-embed -dxil-globals -S -o - | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

;; Check that llc reports an error when attempting to calculate shader hash with sources
;; when no debug info is present.
; RUN: not llc -dx-Zss %s --filetype=obj -o - 2>&1 | FileCheck %s --check-prefix=ZSS
; ZSS: /Zss requires debug info

target triple = "dxil-unknown-shadermodel6.5-library"

; CHECK: @dx.hash = private constant [20 x i8] c"\00\00\00\00{{.*}}", section "HASH", align 4

define i32 @add(i32 %a, i32 %b) {
  %sum = add i32 %a, %b
  ret i32 %sum
}

; DXC: - Name:            HASH
; DXC:   Size:            20
; DXC:   Hash:
; DXC:     IncludesSource:  false
; DXC:     Digest:          [ 
