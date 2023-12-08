; RUN: not llvm-c-test --object-list-sections < /dev/null
; This used to cause a segfault
