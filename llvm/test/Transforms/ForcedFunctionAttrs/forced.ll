; RUN: opt < %s -S -passes=forceattrs | FileCheck %s --check-prefix=CHECK-CONTROL
; RUN: opt < %s -S -passes=forceattrs -force-attribute foo:noinline | FileCheck %s --check-prefix=CHECK-FOO
; RUN: opt < %s -S -passes=forceattrs -force-remove-attribute goo:cold | FileCheck %s --check-prefix=REMOVE-COLD
; RUN: opt < %s -S -passes=forceattrs -force-remove-attribute goo:noinline | FileCheck %s --check-prefix=REMOVE-NOINLINE
; RUN: opt < %s -S -passes=forceattrs -force-attribute goo:cold -force-remove-attribute goo:noinline | FileCheck %s --check-prefix=ADD-COLD-REMOVE-NOINLINE
; RUN: opt < %s -S -passes=forceattrs -force-attribute goo:noinline -force-remove-attribute goo:noinline | FileCheck %s --check-prefix=ADD-NOINLINE-REMOVE-NOINLINE

; CHECK-CONTROL: define void @foo() {
; CHECK-FOO: define void @foo() #0 {
define void @foo() {
  ret void
}

; Ignore `cold` which does not exist before.
; REMOVE-COLD: define void @goo() #0 {

; Remove `noinline` attribute.
; REMOVE-NOINLINE: define void @goo() {

; Add `cold` and remove `noinline` leaving `cold` only.
; ADD-COLD-REMOVE-NOINLINE: define void @goo() #0 {

; `force-remove` takes precedence over `force`.
; `noinline` is removed.
; ADD-NOINLINE-REMOVE-NOINLINE: define void @goo() {

define void @goo() #0 {
  ret void
}
attributes #0 = { noinline }

; CHECK-FOO: attributes #0 = { noinline }
; REMOVE-COLD: attributes #0 = { noinline }
; ADD-COLD-REMOVE-NOINLINE: attributes #0 = { cold }
