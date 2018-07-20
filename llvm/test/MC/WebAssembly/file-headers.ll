; RUN: llc -filetype=obj %s -o - | llvm-readobj -file-headers | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK: Format: WASM{{$}}
; CHECK: Arch: wasm32{{$}}
; CHECK: AddressSize: 32bit{{$}}
; CHECK: Version: 0x1{{$}}
