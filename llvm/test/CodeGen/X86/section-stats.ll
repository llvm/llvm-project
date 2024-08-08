; REQUIRES: asserts
; RUN: llc -o /dev/null -filetype=obj -stats %s 2>&1 | FileCheck %s

; CHECK: {{[0-9+]}} elf-object-writer - Total size of SHF_ALLOC text sections

target triple = "x86_64-unknown-linux-gnu"

define void @f() {
    ret void
}
