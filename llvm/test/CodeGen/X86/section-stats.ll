; RUN: llc -o /dev/null -filetype=obj -stats %s 2>&1 | FileCheck %s

; CHECK: {{[0-9+]}} elf-object-writer - Total size of SHF_ALLOC text sections

define void @f() {
    ret void
}
