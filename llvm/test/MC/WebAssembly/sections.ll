; RUN: llc -filetype=obj %s -o - | llvm-readobj -s | FileCheck %s

target triple = "wasm32-unknown-unknown"

; external function
declare i32 @a()

; global data
@b = global i32 3, align 4

; local function
define i32 @f1() {
entry:
    %tmp1 = call i32 @a()
    ret i32 %tmp1
}

; CHECK: Format: WASM
; CHECK: Arch: wasm32
; CHECK: AddressSize: 32bit
; CHECK: Sections [
; CHECK:   Section {
; CHECK:     Type: TYPE (0x1)
; CHECK:   }
; CHECK:  Section {
; CHECK:    Type: IMPORT (0x2)
; CHECK:  }
; CHECK:  Section {
; CHECK:    Type: FUNCTION (0x3)
; CHECK:  }
; CHECK:  Section {
; CHECK:    Type: CODE (0xA)
; CHECK:  }
; CHECK:  Section {
; CHECK:    Type: DATA (0xB)
; CHECK:  }
; CHECK:  Section {
; CHECK:    Type: CUSTOM (0x0)
; CHECK:    Name: reloc.CODE
; CHECK:  }
; CHECK:]

