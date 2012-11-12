; RUN: opt < %s -print-callgraph -disable-output 2>&1 | FileCheck %s

@a = global void ()* @f		; <void ()**> [#uses=0]

; CHECK: calls function 'f'

define internal void @f() {
	unreachable
}
