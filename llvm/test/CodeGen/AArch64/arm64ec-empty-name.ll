; RUN: llc -mtriple=arm64ec-pc-windows-msvc %s -o - | FileCheck %s

; Regression test: Arm64EC needs to look at the first character of a function
; to decide if it will be mangled like a C or C++ function name, which caused
; it to crash for empty function names.
define void @""() {
        ret void
}

define void @""() {
        ret void
}

; CHECK: "#__unnamed":
; CHECK: "#__unnamed.1":
