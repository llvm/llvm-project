; RUN: llc -mtriple aarch64-pc-linux -relocation-model=pic < %s | FileCheck %s

@g1 = dso_local global i32 42

define dso_local ptr @get_g1() {
; CHECK:      get_g1:
; CHECK:        adrp x0, g1
; CHECK-NEXT:   add  x0, x0, :lo12:g1
  ret ptr @g1
}

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"PIE Level", i32 2}
