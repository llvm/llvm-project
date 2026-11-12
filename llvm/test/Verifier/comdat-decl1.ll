; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

$v = comdat any
@v = available_externally global i32 0, comdat
; CHECK: Declaration may not be in a Comdat!
