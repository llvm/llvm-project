; RUN: not llvm-ml -filetype=s %s /Fo - 2>&1 \
; RUN:   | FileCheck %s --implicit-check-not="{{[0-9]+:[0-9]+: error:}}"

.code

; These used to be valid label names.
{: ; CHECK: [[@LINE]]:1: error: unexpected token at start of statement
}: ; CHECK: [[@LINE]]:1: error: unexpected token at start of statement
