; RUN: llvm-ml -m64 -filetype=s %s /Fo - | FileCheck %s

.code
foo:
; CHECK: foo:
