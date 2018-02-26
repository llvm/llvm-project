; RUN: llc -march=hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s

; CHECK: c0 3f 00 48 48003fc0

define i32 @foo() {
ret i32 0
}
