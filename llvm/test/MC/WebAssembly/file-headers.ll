; RUN: llc -filetype=obj %s -o - | llvm-readobj --file-headers - | FileCheck %s

target triple = "wasm32-unknown-unknown"

; CHECK: Format: WASM{{$}}
; CHECK: Arch: wasm32{{$}}
; CHECK: AddressSize: 32bit{{$}}
; CHECK: Version: 0x1{{$}}
