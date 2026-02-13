; RUN: not llc -debug-pass=Structure -stop-after=dead-mi-elimination,arst %s -o /dev/null 2>&1 \
; RUN:   | FileCheck -check-prefix=NOT-NUM %s

; NOT-NUM: LLVM ERROR: invalid pass instance specifier dead-mi-elimination,arst
