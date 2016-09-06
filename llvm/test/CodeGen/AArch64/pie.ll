; RUN: llc -mtriple aarch64-pc-linux -relocation-model=pic < %s | FileCheck %s

@g1 = global i32 42

define i32* @get_g1() {
; CHECK:      get_g1:
; CHECK:        adrp x0, g1
; CHECK-NEXT:   add  x0, x0, :lo12:g1
  ret i32* @g1
}

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"PIE Level", i32 2}
