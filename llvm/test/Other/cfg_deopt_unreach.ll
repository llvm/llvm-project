; RUN: opt < %s -analyze -dot-cfg -cfg-hide-unreachable-paths -cfg-dot-filename-prefix=unreach 2>/dev/null
; RUN: FileCheck %s -input-file=unreach.callee.dot -check-prefix=UNREACH
; RUN: opt < %s -analyze -dot-cfg -cfg-hide-deoptimize-paths -cfg-dot-filename-prefix=deopt 2>/dev/null
; RUN: FileCheck %s -input-file=deopt.callee.dot -check-prefix=DEOPT
; RUN: opt < %s -analyze -dot-cfg -cfg-dot-filename-prefix=no-flags 2>/dev/null
; RUN: FileCheck %s -input-file=no-flags.callee.dot -check-prefix=NO-FLAGS
; RUN: opt < %s -analyze -dot-cfg -cfg-hide-unreachable-paths -cfg-hide-deoptimize-paths -cfg-dot-filename-prefix=both-flags 2>/dev/null
; RUN: FileCheck %s -input-file=both-flags.callee.dot -check-prefix=BOTH-FLAGS

declare i8 @llvm.experimental.deoptimize.i8(...)

define i8 @callee(i1* %c) alwaysinline {
; NO-FLAGS: [shape=record,label="{%0:\l  %c0 = load volatile i1, i1* %c\l  br i1 %c0, label %lleft, label %lright\l|{<s0>T|<s1>F}}"];
; DEOPT: [shape=record,label="{%0:\l  %c0 = load volatile i1, i1* %c\l  br i1 %c0, label %lleft, label %lright\l|{<s0>T|<s1>F}}"];
; UNREACH: [shape=record,label="{%0:\l  %c0 = load volatile i1, i1* %c\l  br i1 %c0, label %lleft, label %lright\l|{<s0>T|<s1>F}}"];
; BOTH-FLAGS-NOT: [shape=record,label="{%0:\l  %c0 = load volatile i1, i1* %c\l  br i1 %c0, label %lleft, label %lright\l|{<s0>T|<s1>F}}"];
  %c0 = load volatile i1, i1* %c
  br i1 %c0, label %lleft, label %lright
; NO-FLAGS: [shape=record,label="{lleft:                                            \l  %v0 = call i8 (...) @llvm.experimental.deoptimize.i8(i32 1) [ \"deopt\"(i32 1)\l... ]\l  ret i8 %v0\l}"];
; DEOPT-NOT: [shape=record,label="{lleft:                                            \l  %v0 = call i8 (...) @llvm.experimental.deoptimize.i8(i32 1) [ \"deopt\"(i32 1)\l... ]\l  ret i8 %v0\l}"];
; UNREACH: [shape=record,label="{lleft:                                            \l  %v0 = call i8 (...) @llvm.experimental.deoptimize.i8(i32 1) [ \"deopt\"(i32 1)\l... ]\l  ret i8 %v0\l}"];
; BOTH-FLAGS-NOT: [shape=record,label="{lleft:                                            \l  %v0 = call i8 (...) @llvm.experimental.deoptimize.i8(i32 1) [ \"deopt\"(i32 1)\l... ]\l  ret i8 %v0\l}"];
lleft:
  %v0 = call i8(...) @llvm.experimental.deoptimize.i8(i32 1) [ "deopt"(i32 1) ]
  ret i8 %v0

; NO-FLAGS: [shape=record,label="{lright:                                           \l  unreachable\l}"];
; DEOPT: [shape=record,label="{lright:                                           \l  unreachable\l}"];
; UNREACH-NOT: [shape=record,label="{lright:                                           \l  unreachable\l}"];
; BOTH-FLAGS-NOT: [shape=record,label="{lright:                                           \l  unreachable\l}"];
lright:
  unreachable
}
