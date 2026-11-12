;; Check that the function regex for VE works as expected.
; RUN: llc -mtriple=ve -exception-model=sjlj -relocation-model=static < %s | FileCheck %s

;; Check that we accept .Ldsolocal$local: below the function label.
; RUN: llc -mtriple=ve -exception-model=sjlj -relocation-model=pic < %s | FileCheck %s --check-prefix=PIC


@gv0 = dso_local global i32 0, align 4
@gv1 = dso_preemptable global i32 0, align 4

define hidden i32 @"_Z54bar$ompvariant$bar"() {
entry:
  ret i32 2
}

define dso_local i32 @dsolocal() {
entry:
  call void @ext()
  ret i32 2
}

declare void @ext()

define i32 @load() {
entry:
  %a = load i32, i32* @gv0
  %b = load i32, i32* @gv1
  %c = add i32 %a, %b
  ret i32 %c
}
