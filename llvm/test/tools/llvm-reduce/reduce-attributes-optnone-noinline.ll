; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=attributes --test FileCheck --test-arg -check-prefixes=INTERESTING,INTERESTING-NOINLINE --test-arg %s --test-arg --input-file %s -o %t.0
; RUN: FileCheck --check-prefix=RESULT-NOINLINE %s < %t.0

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=attributes --test FileCheck --test-arg -check-prefixes=INTERESTING,INTERESTING-OPTNONE --test-arg %s --test-arg --input-file %s -o %t.1
; RUN: FileCheck --check-prefix=RESULT-OPTNONE %s < %t.1


; Make sure this doesn't hit the "Attribute 'optnone' requires
; 'noinline'!" verifier error. optnone can be dropped separately from
; noinline, but removing noinline requires removing the pair together.


; INTERESTING: @keep_func() [[KEEP_ATTRS:#[0-9]+]]
; RESULT-NOINLINE: define void @keep_func() [[KEEP_ATTRS:#[0-9]+]] {
; RESULT-OPTNONE: define void @keep_func() [[KEEP_ATTRS:#[0-9]+]] {
define void @keep_func() #0 {
  ret void
}

; Both should be removed together
; INTERESTING: @drop_func()
; RESULT-NOINLINE: define void @drop_func() {
; RESULT-OPTNONE: define void @drop_func() {
define void @drop_func() #0 {
  ret void
}

; RESULT-NOINLINE: attributes [[KEEP_ATTRS]] = { noinline }
; RESULT-OPTNONE: attributes [[KEEP_ATTRS]] = { noinline optnone }


; INTERESTING-NOINLINE: attributes [[KEEP_ATTRS]] =
; INTERESTING-NOINLINE-SAME: noinline

; INTERESTING-OPTNONE: attributes [[KEEP_ATTRS]] =
; INTERESTING-OPTNONE-SAME: optnone

attributes #0 = { noinline optnone }
