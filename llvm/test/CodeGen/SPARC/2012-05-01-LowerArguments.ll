; Just check that this doesn't crash:
; RUN: llc < %s
; PR2960

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f128:128:128"
target triple = "sparc-unknown-linux-gnu"
	%"5tango4core9Exception11IOException" = type { ptr, ptr, { i64, ptr }, { i64, ptr }, i64, ptr, ptr }
	%"6Object7Monitor" = type { ptr, ptr }

define fastcc ptr @_D5tango4core9Exception13TextException5_ctorMFAaZC5tango4core9Exception13TextException(ptr %this, { i64, ptr } %msg) {
entry_tango.core.Exception.TextException.this:
	unreachable
}
