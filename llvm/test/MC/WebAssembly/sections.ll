; RUN: llc -mtriple wasm32-unknown-unknown-wasm -filetype=obj %s -o - | llvm-readobj -s | FileCheck %s

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
; CHECK:    Type: TABLE (0x4)
; CHECK:  }
; CHECK:  Section {
; CHECK:    Type: MEMORY (0x5)
; CHECK:    Memories [
; CHECK:      Memory {
; CHECK:        InitialPages: 1
; CHECK:      }
; CHECK:    ]
; CHECK:  }
; CHECK:  Section {
; CHECK:    Type: GLOBAL (0x6)
; CHECK:  }
; CHECK:  Section {
; CHECK:    Type: EXPORT (0x7)
; CHECK:  }
; CHECK:  Section {
; CHECK:    Type: CODE (0xA)
; CHECK:  }
; CHECK:  Section {
; CHECK:    Type: DATA (0xB)
; CHECK:  }
; CHECK:  Section {
; CHECK:    Type: CUSTOM (0x0)
; CHECK:    Name: name
; CHECK:  }
; CHECK:  Section {
; CHECK:    Type: CUSTOM (0x0)
; CHECK:    Name: reloc.CODE
; CHECK:  }
; CHECK:]

