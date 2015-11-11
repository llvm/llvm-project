; RUN: llvm-as -function-summary %s -o %t.bc

; Ensure complete comdat group is materialized
; RUN: llvm-link %t.bc -S | FileCheck %s
; CHECK: $linkoncecomdat = comdat any
; CHECK: @linkoncecomdat = linkonce global i32 2
; CHECK: @linkoncecomdat_unref_var = linkonce global i32 2, comdat($linkoncecomdat)
; CHECK: define linkonce void @linkoncecomdat_unref_func() comdat($linkoncecomdat)

$linkoncecomdat = comdat any
@linkoncecomdat = linkonce global i32 2, comdat($linkoncecomdat)
@linkoncecomdat_unref_var = linkonce global i32 2, comdat($linkoncecomdat)
define linkonce void @linkoncecomdat_unref_func() comdat($linkoncecomdat) {
  ret void
}
; Reference one member of comdat so that comdat is generated.
define void @ref_linkoncecomdat() {
  load i32, i32* @linkoncecomdat, align 4
  ret void
}
