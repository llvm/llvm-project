; RUN: opt %s -passes=pgo-instr-gen -static-func-full-module-prefix=false -S | FileCheck %s --check-prefix=NOPATH
; RUN: opt %s -passes=pgo-instr-gen -static-func-strip-dirname-prefix=1000 -S | FileCheck %s --check-prefix=NOPATH
; RUN: opt %s -passes=pgo-instr-gen -static-func-strip-dirname-prefix=1 -S | FileCheck %s --check-prefix=HASPATH
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; NOPATH: @__profn_statics_counter_naming.ll_func = private constant [30 x i8] c"statics_counter_naming.ll;func"
; HASPATH-NOT: @__profn_statics_counter_naming.ll_func = private constant [30 x i8] c"statics_counter_naming.ll;func"

define internal i32 @func() {
entry:
  ret i32 0
}
