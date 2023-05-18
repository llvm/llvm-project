; RUN: llc --mtriple=loongarch64 --stop-after loongarch-prera-expand-pseudo \
; RUN:     --relocation-model=pic %s -o %t.mir
; RUN: llc --mtriple=loongarch64 --run-pass loongarch-prera-expand-pseudo \
; RUN:     %t.mir -o - | FileCheck %s

;; This tests the LoongArch-specific serialization and deserialization of
;; `target-flags(...)`

@g_e = external global i32
@g_i = internal global i32 0
@t_un = external thread_local global i32
@t_ld = external thread_local(localdynamic) global i32
@t_ie = external thread_local(initialexec) global i32
@t_le = external thread_local(localexec) global i32

declare void @callee1() nounwind
declare dso_local void @callee2() nounwind

define void @caller() nounwind {
; CHECK-LABEL: name: caller
; CHECK:      target-flags(loongarch-got-pc-hi) @g_e
; CHECK-NEXT: target-flags(loongarch-got-pc-lo) @g_e
; CHECK:      target-flags(loongarch-pcrel-hi) @g_i
; CHECK-NEXT: target-flags(loongarch-pcrel-lo) @g_i
; CHECK:      target-flags(loongarch-gd-pc-hi) @t_un
; CHECK-NEXT: target-flags(loongarch-got-pc-lo) @t_un
; CHECK:      target-flags(loongarch-ld-pc-hi) @t_ld
; CHECK-NEXT: target-flags(loongarch-got-pc-lo) @t_ld
; CHECK:      target-flags(loongarch-ie-pc-hi) @t_ie
; CHECK-NEXT: target-flags(loongarch-ie-pc-lo) @t_ie
; CHECK:      target-flags(loongarch-le-hi) @t_le
; CHECK-NEXT: target-flags(loongarch-le-lo) @t_le
; CHECK:      target-flags(loongarch-call-plt) @callee1
; CHECK:      target-flags(loongarch-call) @callee2
  %a = load volatile i32, ptr @g_e
  %b = load volatile i32, ptr @g_i
  %c = load volatile i32, ptr @t_un
  %d = load volatile i32, ptr @t_ld
  %e = load volatile i32, ptr @t_ie
  %f = load volatile i32, ptr @t_le
  call i32 @callee1()
  call i32 @callee2()
  ret void
}
