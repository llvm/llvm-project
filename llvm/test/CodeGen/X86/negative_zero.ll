; RUN: llc < %s -mtriple=i686-- -mattr=-sse2,-sse3 | FileCheck %s

; CHECK: fchs


define double @T() {
	ret double -1.0   ;; codegen as fld1/fchs, not as a load from cst pool
}
