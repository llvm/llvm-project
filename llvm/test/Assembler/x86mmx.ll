; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; Basic smoke test for x86_mmx type.

; CHECK: define x86_mmx @sh16
define x86_mmx  @sh16(x86_mmx %A) {
; CHECK: ret x86_mmx %A
        ret x86_mmx %A
}
