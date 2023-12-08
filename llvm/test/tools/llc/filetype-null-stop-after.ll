; -stop-after would normally dump MIR, but with -filetype=null as well check
; there's no output at all.
; REQUIRES: default_triple
; RUN: llc -filetype=null -stop-after=finalize-isel -o - %s | count 0
