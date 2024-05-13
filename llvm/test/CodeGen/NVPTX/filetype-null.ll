; Check that 'llc' does not crash if '-filetype=null' is used.

; RUN: llc %s -filetype=null -march=nvptx -o -
; RUN: llc %s -filetype=null -march=nvptx64 -o -
