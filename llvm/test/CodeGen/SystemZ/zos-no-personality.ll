; RUN: llc -mtriple s390x-zos < %s | FileCheck %s

define { ptr, i32 } @foo() personality ptr null {
  invoke void null(ptr null, ptr null)
          to label %1 unwind label %2

1:
  ret { ptr, i32 } zeroinitializer

2:
  %3 = landingpad { ptr, i32 }
          catch ptr null
  ret { ptr, i32 } %3
}

; CHECK: foo DS 0H
; CHECK-NOT: .gcc_exception_table.foo
; CHECK: L#PPA1_foo_0 DS 0H
; CHECK-NOT: *   Bit 3: 1 = C++ EH block
