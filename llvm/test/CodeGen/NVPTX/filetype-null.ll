; Check that 'llc' does not crash if '-filetype=null' is used.

; RUN: llc %s -filetype=null -mtriple=nvptx -o -
; RUN: llc %s -filetype=null -mtriple=nvptx64 -o -
