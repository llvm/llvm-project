; RUN: not llc -mtriple=riscv64 -mattr=+xsfmmbase,+save-restore -o /dev/null < %s 2>&1 \
; RUN:   | FileCheck %s --implicit-check-not=error:

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
; CHECK: error: libgcc_call: cannot emit call to '__divdi3' from an RISC-V attributed function.
define i64 @libgcc_call(i64 %a, i64 %b) "riscv_inout" {
  %d = sdiv i64 %a, %b
  ret i64 %d
}

; CHECK: error: memcpy_call: cannot emit call to 'memcpy' from an RISC-V attributed function.
define void @memcpy_call(ptr %d, ptr %s) "riscv_in" {
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 1024, i1 false)
  ret void
}

declare void @extern_func()
; CHECK: error: extern_call: cannot emit call to 'extern_func' from an RISC-V attributed function.
define void @extern_call() "riscv_in" {
  tail call void @extern_func()
  ret void
}

declare void @preserves(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) "riscv_preserves"
; spill lib calls should be legal, e.g. __riscv_save_4, __riscv_save_5
define void @legal(i64 %a) "riscv_in" {
  call void @preserves(i64 %a, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11)
  call void @preserves(i64 %a, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11)
  ret void
}
