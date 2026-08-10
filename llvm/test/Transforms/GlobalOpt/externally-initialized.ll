; RUN: opt < %s -S -passes=globalopt | FileCheck %s
; RUN: opt < %s -passes=early-cse | opt -S -passes=globalopt | FileCheck %s --check-prefix=CHECK-CONSTANT

; This global is externally_initialized, which may modify the value between
; it's static initializer and any code in this module being run, so the only
; write to it cannot be merged into the static initialiser.
; CHECK: @a = internal unnamed_addr externally_initialized global i32 undef
@a = internal externally_initialized global i32 undef

; This global is stored to by the external initialization, so cannot be
; constant-propagated and removed, despite the fact that there are no writes
; to it.
; CHECK: @b = internal unnamed_addr externally_initialized global i32 undef
@b = internal externally_initialized global i32 undef

; This constant global is externally_initialized, which may modify the value
; between its static const initializer and any code in this module being run, so
; the read from it cannot be const propagated
@c = internal externally_initialized constant i32 42

define void @foo() {
; CHECK-LABEL: foo
entry:
; CHECK: store i32 42, ptr @a
  store i32 42, ptr @a
  ret void
}
define i32 @bar() {
; CHECK-LABEL: bar
entry:
; CHECK: %val = load i32, ptr @a
  %val = load i32, ptr @a
  ret i32 %val
}

define i32 @baz() {
; CHECK-LABEL: baz
entry:
; CHECK: %val = load i32, ptr @b
  %val = load i32, ptr @b
  ret i32 %val
}

define i32 @bam() {
; CHECK-CONSTANT-LABEL: bam
entry:
; CHECK-CONSTANT: %val = load i32, ptr @c
  %val = load i32, ptr @c
  ret i32 %val
}
