; RUN: opt < %s -S -passes=verify,forceattrs,verify | FileCheck %s --check-prefix=CHECK-CONTROL
; RUN: opt < %s -S -passes=verify,forceattrs,verify -force-attribute foo:noinline | FileCheck %s --check-prefix=CHECK-FOO
; RUN: opt < %s -S -passes=verify,forceattrs,verify -force-remove-attribute goo:cold | FileCheck %s --check-prefix=REMOVE-COLD
; RUN: opt < %s -S -passes=verify,forceattrs,verify -force-remove-attribute goo:noinline | FileCheck %s --check-prefix=REMOVE-NOINLINE
; RUN: opt < %s -S -passes=verify,forceattrs,verify -force-attribute goo:cold -force-remove-attribute goo:noinline | FileCheck %s --check-prefix=ADD-COLD-REMOVE-NOINLINE
; RUN: opt < %s -S -passes=verify,forceattrs,verify -force-attribute goo:noinline -force-remove-attribute goo:noinline | FileCheck %s --check-prefix=ADD-NOINLINE-REMOVE-NOINLINE
; RUN: opt < %s -S -passes=verify,forceattrs,verify -force-attribute optsize | FileCheck %s --check-prefix=CHECK-ADD-ALL
; RUN: opt < %s -S -passes=verify,forceattrs,verify -force-remove-attribute noinline | FileCheck %s --check-prefix=CHECK-REMOVE-ALL
; RUN: opt < %s -S -passes=verify,forceattrs,verify -force-attribute alwaysinline | FileCheck %s --check-prefix=CHECK-ALWAYSINLINE-ALL
; RUN: opt < %s -S -passes=verify,forceattrs,verify -force-attribute noinline | FileCheck %s --check-prefix=CHECK-NOINLINE-ALL

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

define void @hoo() #1 {
  ret void
}

attributes #0 = { noinline }
attributes #1 = { alwaysinline }

; CHECK-FOO: attributes #0 = { noinline }
; REMOVE-COLD: attributes #0 = { noinline }
; ADD-COLD-REMOVE-NOINLINE: attributes #0 = { cold }

; When passing an attribute without specifying a function, the attribute
; should be added to all functions in the module.
; CHECK-ADD-ALL: define void @foo() #0 {
; CHECK-ADD-ALL: define void @goo() #1 {
; CHECK-ADD-ALL: define void @hoo() #2 {
; CHECK-ADD-ALL: attributes #0 = { optsize }
; CHECK-ADD-ALL: attributes #1 = { noinline optsize }
; CHECK-ADD-ALL: attributes #2 = { alwaysinline optsize }

; When passing an attribute to be removed without specifying a function,
; the attribute should be removed from all functions in the module that
; have it.
; CHECK-REMOVE-ALL: define void @foo() {
; CHECK-REMOVE-ALL: define void @goo() {
; CHECK-REMOVE-ALL: define void @hoo() #0 {
; CHECK-REMOVE-ALL: attributes #0 = { alwaysinline }

; When forcing alwaysinline on all functions, it should not be added to
; functions that already have noinline (would produce invalid IR).
; CHECK-ALWAYSINLINE-ALL: define void @foo() #0 {
; CHECK-ALWAYSINLINE-ALL: define void @goo() #1 {
; CHECK-ALWAYSINLINE-ALL: define void @hoo() #0 {
; CHECK-ALWAYSINLINE-ALL-DAG: attributes #0 = { alwaysinline }
; CHECK-ALWAYSINLINE-ALL-DAG: attributes #1 = { noinline }

; When forcing noinline on all functions, it should not be added to
; functions that already have alwaysinline (would produce invalid IR).
; CHECK-NOINLINE-ALL: define void @foo() #0 {
; CHECK-NOINLINE-ALL: define void @goo() #0 {
; CHECK-NOINLINE-ALL: define void @hoo() #1 {
; CHECK-NOINLINE-ALL-DAG: attributes #0 = { noinline }
; CHECK-NOINLINE-ALL-DAG: attributes #1 = { alwaysinline }

