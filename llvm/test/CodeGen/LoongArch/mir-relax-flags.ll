; RUN: llc --mtriple=loongarch64 -mattr=+d,+relax --stop-after loongarch-prera-expand-pseudo \
; RUN:     --relocation-model=pic --code-model=medium %s -o %t.mir
; RUN: llc --mtriple=loongarch64 -mattr=+d,+relax --run-pass loongarch-prera-expand-pseudo \
; RUN:     --code-model=medium %t.mir -o - | FileCheck %s --check-prefixes=CHECK,MEDCALL
; RUN: llc --mtriple=loongarch64 -mattr=+d,+relax --run-pass loongarch-expand-pseudo \
; RUN:     --code-model=medium %t.mir -o - | FileCheck %s --check-prefixes=CHECK,CALL36

; RUN: llc --mtriple=loongarch64 -mattr=+d,+relax --stop-after loongarch-prera-expand-pseudo \
; RUN:     --relocation-model=pic --enable-tlsdesc --code-model=medium %s -o %t.desc.mir
; RUN: llc --mtriple=loongarch64 -mattr=+d,+relax --run-pass loongarch-prera-expand-pseudo \
; RUN:     --code-model=medium %t.desc.mir -o - | FileCheck %s --check-prefix=DESC

;; Check target-flags after expand-pseudo pass.

@g_e = external global i32
@g_i = internal global i32 0
@t_un = external thread_local global i32
@t_ld = external thread_local(localdynamic) global i32
@t_ie = external thread_local(initialexec) global i32
@t_le = external thread_local(localexec) global i32

declare void @callee1() nounwind
declare dso_local void @callee2() nounwind
declare dso_local void @callee3() nounwind

define void @caller() nounwind {
; CHECK:      target-flags(loongarch-got-pc-hi, loongarch-relax) @g_e
; CHECK-NEXT: target-flags(loongarch-got-pc-lo, loongarch-relax) @g_e
; CHECK:      target-flags(loongarch-pcrel-hi, loongarch-relax) @g_i
; CHECK-NEXT: target-flags(loongarch-pcrel-lo, loongarch-relax) @g_i
; CHECK:      target-flags(loongarch-gd-pc-hi, loongarch-relax) @t_un
; CHECK-NEXT: target-flags(loongarch-got-pc-lo, loongarch-relax) @t_un
; DESC:       target-flags(loongarch-desc-pc-hi, loongarch-relax) @t_un
; DESC-NEXT:  target-flags(loongarch-desc-pc-lo, loongarch-relax) @t_un
; DESC-NEXT:  target-flags(loongarch-desc-ld, loongarch-relax) @t_un
; DESC-NEXT:  target-flags(loongarch-desc-call, loongarch-relax) @t_un
; CHECK:      target-flags(loongarch-ld-pc-hi, loongarch-relax) @t_ld
; CHECK-NEXT: target-flags(loongarch-got-pc-lo, loongarch-relax) @t_ld
; DESC:       target-flags(loongarch-desc-pc-hi, loongarch-relax) @t_ld
; DESC-NEXT:  target-flags(loongarch-desc-pc-lo, loongarch-relax) @t_ld
; DESC-NEXT:  target-flags(loongarch-desc-ld, loongarch-relax) @t_ld
; DESC-NEXT:  target-flags(loongarch-desc-call, loongarch-relax) @t_ld
; CHECK:      target-flags(loongarch-ie-pc-hi, loongarch-relax) @t_ie
; CHECK-NEXT: target-flags(loongarch-ie-pc-lo, loongarch-relax) @t_ie
; CHECK:      target-flags(loongarch-le-hi-r) @t_le
; CHECK-NEXT: target-flags(loongarch-le-add-r) @t_le
; CHECK-NEXT: target-flags(loongarch-le-lo-r) @t_le
; MEDCALL:    target-flags(loongarch-call-plt) @callee1
; CALL36:     target-flags(loongarch-call36) @callee1
; MEDCALL:    target-flags(loongarch-call) @callee2
; CALL36:     target-flags(loongarch-call36) @callee2
; MEDCALL:    target-flags(loongarch-call) @callee3
; CALL36:     target-flags(loongarch-call36) @callee3
  %a = load volatile i32, ptr @g_e
  %b = load volatile i32, ptr @g_i
  %c = load volatile i32, ptr @t_un
  %d = load volatile i32, ptr @t_ld
  %e = load volatile i32, ptr @t_ie
  %f = load volatile i32, ptr @t_le
  call i32 @callee1()
  call i32 @callee2()
  tail call i32 @callee3()
  ret void
}
