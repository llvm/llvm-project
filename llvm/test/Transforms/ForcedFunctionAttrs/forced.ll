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
; RUN: opt < %s -S -passes=verify,forceattrs,verify -force-attribute optnone | FileCheck %s --check-prefix=CHECK-OPTNONE-ALL
; RUN: opt < %s -S -passes=verify,forceattrs,verify -force-attribute minsize | FileCheck %s --check-prefix=CHECK-MINSIZE-ALL
; RUN: opt < %s -S -passes=verify,forceattrs,verify -force-attribute optdebug | FileCheck %s --check-prefix=CHECK-OPTDEBUG-ALL

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

define void @zoo() #2 {
  ret void
}

define void @bar() #3 {
  ret void
}

define void @baz() #4 {
  ret void
}

define void @qoo() #5 {
  ret void
}

attributes #0 = { noinline }
attributes #1 = { alwaysinline }
attributes #2 = { optsize }
attributes #3 = { minsize }
attributes #4 = { optdebug }
attributes #5 = { noinline optnone }

; CHECK-FOO: attributes #0 = { noinline }
; REMOVE-COLD: attributes #0 = { noinline }
; ADD-COLD-REMOVE-NOINLINE: attributes #0 = { cold }

; When passing an attribute without specifying a function, the attribute
; should be added to all compatible functions in the module.
; CHECK-ADD-ALL: define void @foo() #0 {
; CHECK-ADD-ALL: define void @goo() #1 {
; CHECK-ADD-ALL: define void @hoo() #2 {
; CHECK-ADD-ALL: define void @zoo() #0 {
; CHECK-ADD-ALL: define void @bar() #3 {
; CHECK-ADD-ALL: define void @baz() #4 {
; CHECK-ADD-ALL: define void @qoo() #5 {
; CHECK-ADD-ALL-DAG: attributes #0 = { optsize }
; CHECK-ADD-ALL-DAG: attributes #1 = { noinline optsize }
; CHECK-ADD-ALL-DAG: attributes #2 = { alwaysinline optsize }
; CHECK-ADD-ALL-DAG: attributes #3 = { minsize optsize }
; CHECK-ADD-ALL-DAG: attributes #4 = { optdebug }
; CHECK-ADD-ALL-DAG: attributes #5 = { noinline optnone }

; When passing an attribute to be removed without specifying a function,
; the attribute should be removed from all functions in the module that
; have it, unless doing so would create invalid IR (e.g. `optnone` requires
; `noinline`).
; CHECK-REMOVE-ALL: define void @foo() {
; CHECK-REMOVE-ALL: define void @goo() {
; CHECK-REMOVE-ALL: define void @hoo() #0 {
; CHECK-REMOVE-ALL: define void @zoo() #1 {
; CHECK-REMOVE-ALL: define void @bar() #2 {
; CHECK-REMOVE-ALL: define void @baz() #3 {
; CHECK-REMOVE-ALL: define void @qoo() #4 {
; CHECK-REMOVE-ALL-DAG: attributes #0 = { alwaysinline }
; CHECK-REMOVE-ALL-DAG: attributes #1 = { optsize }
; CHECK-REMOVE-ALL-DAG: attributes #2 = { minsize }
; CHECK-REMOVE-ALL-DAG: attributes #3 = { optdebug }
; CHECK-REMOVE-ALL-DAG: attributes #4 = { noinline optnone }

; When forcing alwaysinline on all functions, it should not be added to
; functions that already have noinline or optnone (would produce invalid IR).
; CHECK-ALWAYSINLINE-ALL: define void @foo() #0 {
; CHECK-ALWAYSINLINE-ALL: define void @goo() #1 {
; CHECK-ALWAYSINLINE-ALL: define void @hoo() #0 {
; CHECK-ALWAYSINLINE-ALL: define void @zoo() #2 {
; CHECK-ALWAYSINLINE-ALL: define void @bar() #3 {
; CHECK-ALWAYSINLINE-ALL: define void @baz() #4 {
; CHECK-ALWAYSINLINE-ALL: define void @qoo() #5 {
; CHECK-ALWAYSINLINE-ALL-DAG: attributes #0 = { alwaysinline }
; CHECK-ALWAYSINLINE-ALL-DAG: attributes #1 = { noinline }
; CHECK-ALWAYSINLINE-ALL-DAG: attributes #2 = { alwaysinline optsize }
; CHECK-ALWAYSINLINE-ALL-DAG: attributes #3 = { alwaysinline minsize }
; CHECK-ALWAYSINLINE-ALL-DAG: attributes #4 = { alwaysinline optdebug }
; CHECK-ALWAYSINLINE-ALL-DAG: attributes #5 = { noinline optnone }

; When forcing noinline on all functions, it should not be added to
; functions that already have alwaysinline (would produce invalid IR).
; CHECK-NOINLINE-ALL: define void @foo() #0 {
; CHECK-NOINLINE-ALL: define void @goo() #0 {
; CHECK-NOINLINE-ALL: define void @hoo() #1 {
; CHECK-NOINLINE-ALL: define void @zoo() #2 {
; CHECK-NOINLINE-ALL: define void @bar() #3 {
; CHECK-NOINLINE-ALL: define void @baz() #4 {
; CHECK-NOINLINE-ALL: define void @qoo() #5 {
; CHECK-NOINLINE-ALL-DAG: attributes #0 = { noinline }
; CHECK-NOINLINE-ALL-DAG: attributes #1 = { alwaysinline }
; CHECK-NOINLINE-ALL-DAG: attributes #2 = { noinline optsize }
; CHECK-NOINLINE-ALL-DAG: attributes #3 = { minsize noinline }
; CHECK-NOINLINE-ALL-DAG: attributes #4 = { noinline optdebug }
; CHECK-NOINLINE-ALL-DAG: attributes #5 = { noinline optnone }

; When forcing optnone on all functions, it should not be added to functions
; that already have alwaysinline, optsize, minsize, or optdebug.
; CHECK-OPTNONE-ALL: define void @foo() #0 {
; CHECK-OPTNONE-ALL: define void @goo() #0 {
; CHECK-OPTNONE-ALL: define void @hoo() #1 {
; CHECK-OPTNONE-ALL: define void @zoo() #2 {
; CHECK-OPTNONE-ALL: define void @bar() #3 {
; CHECK-OPTNONE-ALL: define void @baz() #4 {
; CHECK-OPTNONE-ALL: define void @qoo() #0 {
; CHECK-OPTNONE-ALL-DAG: attributes #0 = { noinline optnone }
; CHECK-OPTNONE-ALL-DAG: attributes #1 = { alwaysinline }
; CHECK-OPTNONE-ALL-DAG: attributes #2 = { optsize }
; CHECK-OPTNONE-ALL-DAG: attributes #3 = { minsize }
; CHECK-OPTNONE-ALL-DAG: attributes #4 = { optdebug }

; When forcing minsize on all functions, it should not be added to functions
; that already have optnone or optdebug.
; CHECK-MINSIZE-ALL: define void @foo() #0 {
; CHECK-MINSIZE-ALL: define void @goo() #1 {
; CHECK-MINSIZE-ALL: define void @hoo() #2 {
; CHECK-MINSIZE-ALL: define void @zoo() #3 {
; CHECK-MINSIZE-ALL: define void @bar() #0 {
; CHECK-MINSIZE-ALL: define void @baz() #4 {
; CHECK-MINSIZE-ALL: define void @qoo() #5 {
; CHECK-MINSIZE-ALL-DAG: attributes #0 = { minsize }
; CHECK-MINSIZE-ALL-DAG: attributes #1 = { minsize noinline }
; CHECK-MINSIZE-ALL-DAG: attributes #2 = { alwaysinline minsize }
; CHECK-MINSIZE-ALL-DAG: attributes #3 = { minsize optsize }
; CHECK-MINSIZE-ALL-DAG: attributes #4 = { optdebug }
; CHECK-MINSIZE-ALL-DAG: attributes #5 = { noinline optnone }

; When forcing optdebug on all functions, it should not be added to functions
; that already have optnone, minsize, or optsize.
; CHECK-OPTDEBUG-ALL: define void @foo() #0 {
; CHECK-OPTDEBUG-ALL: define void @goo() #1 {
; CHECK-OPTDEBUG-ALL: define void @hoo() #2 {
; CHECK-OPTDEBUG-ALL: define void @zoo() #3 {
; CHECK-OPTDEBUG-ALL: define void @bar() #4 {
; CHECK-OPTDEBUG-ALL: define void @baz() #0 {
; CHECK-OPTDEBUG-ALL: define void @qoo() #5 {
; CHECK-OPTDEBUG-ALL-DAG: attributes #0 = { optdebug }
; CHECK-OPTDEBUG-ALL-DAG: attributes #1 = { noinline optdebug }
; CHECK-OPTDEBUG-ALL-DAG: attributes #2 = { alwaysinline optdebug }
; CHECK-OPTDEBUG-ALL-DAG: attributes #3 = { optsize }
; CHECK-OPTDEBUG-ALL-DAG: attributes #4 = { minsize }
; CHECK-OPTDEBUG-ALL-DAG: attributes #5 = { noinline optnone }
