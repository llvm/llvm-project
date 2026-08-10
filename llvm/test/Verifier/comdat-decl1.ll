; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

$v = comdat any
@v = available_externally global i32 0, comdat
; CHECK: Declaration may not be in a Comdat!
