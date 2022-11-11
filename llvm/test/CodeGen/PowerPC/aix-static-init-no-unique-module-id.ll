; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo, ptr null }]

define internal void @foo() {
  ret void
}

; FIXME: Adjust the comment after we use source file full path to generate unique
; module id instead.
; Use the Pid and timestamp to generate a unique module id when strong external
; symbols are not available in current module. The module id generated in this
; way is not reproducible. A function name sample would be:
; __sinit80000000_clangPidTime_119189_1597348415_0

; CHECK:              .lglobl        foo[DS]
; CHECK:              .lglobl        .foo
; CHECK:              .csect foo[DS]
; CHECK-NEXT: __sinit80000000_clangPidTime_[[PID:[0-9]+]]_[[TIMESTAMP:[0-9]+]]_0:
; CHECK:      .foo:
; CHECK-NEXT: .__sinit80000000_clangPidTime_[[PID]]_[[TIMESTAMP]]_0:
; CHECK:      .globl	__sinit80000000_clangPidTime_[[PID]]_[[TIMESTAMP]]_0
; CHECK:      .globl	.__sinit80000000_clangPidTime_[[PID]]_[[TIMESTAMP]]_0
