; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

$v = comdat any
@v = external global i32, comdat
; CHECK: Declaration may not be in a Comdat!
