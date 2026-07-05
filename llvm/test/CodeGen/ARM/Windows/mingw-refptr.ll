; RUN: llc < %s -mtriple=thumbv7-w64-mingw32 | FileCheck %s

@var = external local_unnamed_addr global i32, align 4
@dsolocalvar = external dso_local local_unnamed_addr global i32, align 4
@localvar = dso_local local_unnamed_addr global i32 0, align 4
@localcommon = common dso_local local_unnamed_addr global i32 0, align 4
@extvar = external dllimport local_unnamed_addr global i32, align 4

define dso_local i32 @getVar() {
; CHECK-LABEL: getVar:
; CHECK:    movw r0, :lower16:.refptr.var
; CHECK:    movt r0, :upper16:.refptr.var
; CHECK:    ldr  r0, [r0]
; CHECK:    ldr  r0, [r0]
; CHECK:    bx   lr
entry:
  %0 = load i32, ptr @var, align 4
  ret i32 %0
}

define dso_local i32 @getDsoLocalVar() {
; CHECK-LABEL: getDsoLocalVar:
; CHECK:    movw r0, :lower16:dsolocalvar
; CHECK:    movt r0, :upper16:dsolocalvar
; CHECK:    ldr  r0, [r0]
; CHECK:    bx   lr
entry:
  %0 = load i32, ptr @dsolocalvar, align 4
  ret i32 %0
}

define dso_local i32 @getLocalVar() {
; CHECK-LABEL: getLocalVar:
; CHECK:    movw r0, :lower16:localvar
; CHECK:    movt r0, :upper16:localvar
; CHECK:    ldr  r0, [r0]
; CHECK:    bx   lr
entry:
  %0 = load i32, ptr @localvar, align 4
  ret i32 %0
}

define dso_local i32 @getLocalCommon() {
; CHECK-LABEL: getLocalCommon:
; CHECK:    movw r0, :lower16:localcommon
; CHECK:    movt r0, :upper16:localcommon
; CHECK:    ldr  r0, [r0]
; CHECK:    bx   lr
entry:
  %0 = load i32, ptr @localcommon, align 4
  ret i32 %0
}

define dso_local i32 @getExtVar() {
; CHECK-LABEL: getExtVar:
; CHECK:    movw r0, :lower16:__imp_extvar
; CHECK:    movt r0, :upper16:__imp_extvar
; CHECK:    ldr  r0, [r0]
; CHECK:    ldr  r0, [r0]
; CHECK:    bx   lr
entry:
  %0 = load i32, ptr @extvar, align 4
  ret i32 %0
}

define dso_local void @callFunc() {
; CHECK-LABEL: callFunc:
; CHECK:    b.w otherFunc
entry:
  tail call void @otherFunc()
  ret void
}

declare dso_local void @otherFunc()

; CHECK:        .section        .rdata$.refptr.var,"dr",discard,.refptr.var
; CHECK:        .globl  .refptr.var
; CHECK: .refptr.var:
; CHECK:        .long   var
