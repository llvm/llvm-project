; RUN: llc -mtriple=x86_64-linux-gnu -filetype=asm < %s | FileCheck %s --check-prefix=DIRECTIVE
; RUN: llc -mtriple=x86_64-linux-gnu -filetype=obj < %s | llvm-readobj --symbols | FileCheck %s --check-prefix=STT-FILE

; DIRECTIVE: .file "foobar"
; STT-FILE: Name: foobar
; STT-FILE-NEXT: Value: 0x0
; STT-FILE-NEXT: Size: 0
; STT-FILE-NEXT: Binding: Local
; STT-FILE-NEXT: Type: File
; STT-FILE-NEXT: Other: 0
; STT-FILE-NEXT: Section: Absolute

source_filename = "/path/to/foobar"
