; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

$v = comdat any
@v = common global i32 0, comdat($v)
; CHECK: 'common' global may not be in a Comdat!
