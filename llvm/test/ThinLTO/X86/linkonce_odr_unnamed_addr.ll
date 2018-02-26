; This test ensures that when linkonce_odr + unnamed_addr symbols promoted to
; weak symbols, it preserves the auto hide property.

; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %s -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.bc %t.bc %t2.bc
; RUN: llvm-lto -thinlto-action=promote %t.bc -thinlto-index=%t3.bc -o - | llvm-dis -o - | FileCheck %s

; CHECK: @linkonceodrunnamed = weak_odr hidden unnamed_addr constant i32 0
@linkonceodrunnamed = linkonce_odr unnamed_addr constant i32 0
