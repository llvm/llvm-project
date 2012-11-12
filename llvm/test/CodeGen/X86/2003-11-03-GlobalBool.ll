; RUN: llc < %s -march=x86 | \
; RUN:   not grep ".byte[[:space:]]*true"

@X = global i1 true             ; <i1*> [#uses=0]
