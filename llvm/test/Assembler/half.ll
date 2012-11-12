; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; Basic smoke test for half type.

; CHECK: define half @halftest
define half  @halftest(half %A0) {
; CHECK: ret half %A0
        ret half %A0
}
