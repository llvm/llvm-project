; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -mxcoff-roptr < %s | FileCheck %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -mxcoff-roptr -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -t --symbol-description %t.o | FileCheck %s --check-prefix=OBJ

; RUN: not llc -mtriple powerpc64-ibm-aix-xcoff -mxcoff-roptr -data-sections=false \
; RUN: < %s 2>&1 | FileCheck %s --check-prefix=DS_ERR
; RUN: not llc -mtriple powerpc64le-unknown-linux-gnu -mxcoff-roptr \
; RUN: < %s 2>&1 | FileCheck %s --check-prefix=OS_ERR

; DS_ERR: -mxcoff-roptr option must be used with -data-sections
; OS_ERR: -mxcoff-roptr option is only supported on AIX

%union.U = type { %"struct.U::A" }
%"struct.U::A" = type { ptr }

@_ZL1p = internal constant i64 ptrtoint (ptr @_ZL1p to i64), align 8
; CHECK:         .csect _ZL1p[RO],3
; CHECK-NEXT:    .lglobl	_ZL1p[RO]
; CHECK-NEXT:    .align	3
; CHECK-NEXT:    .vbyte	8, _ZL1p[RO]
; OBJ-DAG: {{([[:xdigit:]]{16})}} l .text {{([[:xdigit:]]{16})}} (idx: [[#]]) _ZL1p[RO]
@q = thread_local constant ptr @_ZL1p, align 8
; CHECK:         .csect q[TL],3
; CHECK-NEXT:    .globl	q[TL]
; CHECK-NEXT:    .align	3
; CHECK-NEXT:    .vbyte	8, _ZL1p[RO]
; OBJ-DAG: {{([[:xdigit:]]{16})}} g O .tdata {{([[:xdigit:]]{16})}} (idx: [[#]]) q[TL]
@u = local_unnamed_addr constant [1 x %union.U] [%union.U { %"struct.U::A" { ptr @_ZL1p } }], align 8
; CHECK:         .csect u[RO],3
; CHECK-NEXT:    .globl	u[RO]
; CHECK-NEXT:    .align	3
; CHECK-NEXT:    .vbyte	8, _ZL1p[RO]
; OBJ-DAG: {{([[:xdigit:]]{16})}} g .text {{([[:xdigit:]]{16})}} (idx: [[#]]) u[RO]
