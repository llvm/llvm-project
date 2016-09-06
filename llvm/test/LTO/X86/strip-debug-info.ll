; RUN: not llvm-lto -lto-strip-invalid-debug-info=false \
; RUN:     -o %t.o %S/Inputs/strip-debug-info.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty -check-prefix=CHECK-ERR
; RUN: llvm-lto -lto-strip-invalid-debug-info=true \
; RUN:     -exported-symbol foo -exported-symbol _foo \
; RUN:     -o %t.o %S/Inputs/strip-debug-info.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty -check-prefix=CHECK-WARN
; RUN: llvm-nm %t.o | FileCheck %s 

; CHECK-ERR: Broken module found, compilation aborted
; CHECK-WARN: Invalid debug info found, debug info will be stripped
; CHECK: foo
define void @foo() {
  ret void
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DIFile(filename: "broken", directory: "")
