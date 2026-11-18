; The object-file streamer (unlike the assembly streamer) starts with no current
; section, so getCurrentSectionOnly() returns a null current fragment. Emitting a
; directive before any section directive must report the "expected section
; directive" diagnostic instead of dereferencing the null fragment and crashing.
; RUN: not llvm-ml -filetype=obj %s /Fo /dev/null 2>&1 | FileCheck %s

; CHECK: :[[# @LINE + 1]]:6: error: expected section directive before assembly directive in 'BYTE' directive
BYTE 2, 3, 4
