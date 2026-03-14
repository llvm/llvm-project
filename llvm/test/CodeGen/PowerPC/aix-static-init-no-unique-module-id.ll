; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo, ptr null }]

define internal void @foo() {
  ret void
}

; Use the Pid, threadId, and timestamp to generate a unique module id when strong external
; symbols are not available in current module. The module id generated in this
; way is not reproducible. A function name sample would be:
; __sinit80000000_clangPidTidTime_56689326_1_57027228417827568_0

; CHECK:              .lglobl        foo[DS]
; CHECK:              .lglobl        .foo
; CHECK:              .csect foo[DS]
; CHECK-NEXT: __sinit80000000_clangPidTidTime_[[PID:[0-9]+]]_[[TID:[0-9]+]]_[[TIMESTAMP:[0-9]+]]_0:
; CHECK:      .foo:
; CHECK-NEXT: .__sinit80000000_clangPidTidTime_[[PID]]_[[TID]]_[[TIMESTAMP]]_0:
; CHECK:      .globl	__sinit80000000_clangPidTidTime_[[PID]]_[[TID]]_[[TIMESTAMP]]_0
; CHECK:      .globl	.__sinit80000000_clangPidTidTime_[[PID]]_[[TID]]_[[TIMESTAMP]]_0
