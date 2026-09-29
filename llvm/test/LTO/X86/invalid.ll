; RUN: not llvm-lto %S/Inputs/invalid.ll.bc 2>&1 | FileCheck %s


; CHECK: llvm-lto{{.*}}: error loading file '{{.*}}/Inputs/invalid.ll.bc': Unknown attribute kind (255)
