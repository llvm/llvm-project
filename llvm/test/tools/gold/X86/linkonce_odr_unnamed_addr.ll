; This test ensures that when linkonce_odr + unnamed_addr symbols promoted to
; weak symbols, it preserves the auto hide property when possible.

; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/linkonce_odr_unnamed_addr.ll -o %t2.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=save-temps \
; RUN:    %t.o %t2.o -o %t3.o
; RUN: llvm-dis %t.o.1.promote.bc -o - | FileCheck %s

; Now test when one module is a native object. In that case we must be
; conservative and not auto hide.
; RUN: llc %p/Inputs/linkonce_odr_unnamed_addr.ll -o %t2native.o -filetype=obj
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=save-temps \
; RUN:    %t.o %t2native.o -o %t3.o
; RUN: llvm-dis %t.o.1.promote.bc -o - | FileCheck %s --check-prefix=NOSUMMARY

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

; In this case all copies are linkonce_odr, so it may be hidden.
; CHECK: @linkonceodrunnamed = weak_odr hidden unnamed_addr constant i32 0
; NOSUMMARY: @linkonceodrunnamed = weak_odr dso_local unnamed_addr constant i32 0
@linkonceodrunnamed = linkonce_odr unnamed_addr constant i32 0

; In this case, the other copy was weak_odr, so it may not be hidden.
; CHECK: @odrunnamed = weak_odr dso_local unnamed_addr constant i32 0
; NOSUMMARY: @odrunnamed = weak_odr dso_local unnamed_addr constant i32 0
@odrunnamed = linkonce_odr unnamed_addr constant i32 0
