; RUN: rm -rf %t
; RUN: mkdir -p %t
; RUN: opt < %s -dot-cfg -cfg-hide-unreachable-paths -cfg-dot-filename-prefix=%t/unreach 2>/dev/null
; RUN: FileCheck %s -input-file=%t/unreach.callee.dot -check-prefix=UNREACH
; RUN: opt < %s -dot-cfg -cfg-hide-deoptimize-paths -cfg-dot-filename-prefix=%t/deopt 2>/dev/null
; RUN: FileCheck %s -input-file=%t/deopt.callee.dot -check-prefix=DEOPT
; RUN: opt < %s -dot-cfg -cfg-dot-filename-prefix=%t/no-flags 2>/dev/null
; RUN: FileCheck %s -input-file=%t/no-flags.callee.dot -check-prefix=NO-FLAGS
; RUN: opt < %s -dot-cfg -cfg-hide-unreachable-paths -cfg-hide-deoptimize-paths -cfg-dot-filename-prefix=%t/both-flags 2>/dev/null
; RUN: FileCheck %s -input-file=%t/both-flags.callee.dot -check-prefix=BOTH-FLAGS

declare i8 @llvm.experimental.deoptimize.i8(...)

define i8 @callee(i1* %c) alwaysinline {
  %c0 = load volatile i1, i1* %c
  br i1 %c0, label %lleft, label %lright
; NO-FLAGS: label="{lleft:                                            \l  %v0 = call i8 (...) @llvm.experimental.deoptimize.i8(i32 1) [ \"deopt\"(i32 1)\l... ]\l  ret i8 %v0\l}"
; DEOPT-NOT: label="{lleft:                                            \l  %v0 = call i8 (...) @llvm.experimental.deoptimize.i8(i32 1) [ \"deopt\"(i32 1)\l... ]\l  ret i8 %v0\l}"
; UNREACH: label="{lleft:                                            \l  %v0 = call i8 (...) @llvm.experimental.deoptimize.i8(i32 1) [ \"deopt\"(i32 1)\l... ]\l  ret i8 %v0\l}"
; BOTH-FLAGS-NOT: label="{lleft:                                            \l  %v0 = call i8 (...) @llvm.experimental.deoptimize.i8(i32 1) [ \"deopt\"(i32 1)\l... ]\l  ret i8 %v0\l}"
lleft:
  %v0 = call i8(...) @llvm.experimental.deoptimize.i8(i32 1) [ "deopt"(i32 1) ]
  ret i8 %v0

; NO-FLAGS: label="{lright:                                           \l  unreachable\l}"
; DEOPT: label="{lright:                                           \l  unreachable\l}"
; UNREACH-NOT: label="{lright:                                           \l  unreachable\l}"
; BOTH-FLAGS-NOT: label="{lright:                                           \l  unreachable\l}"
lright:
  unreachable
}
