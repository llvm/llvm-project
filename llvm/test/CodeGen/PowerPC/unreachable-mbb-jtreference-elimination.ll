; RUN: llc -O2 --verify-machineinstrs -stop-before=livevars \
; RUN:   -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s

define dso_local void @foo() #0 {
; CHECK-LABEL: fixedStack:
; CHECK-NEXT:  stack: []
; CHECK-NEXT:  callSites: []
; CHECK-NEXT:  debugValueSubstitutions: []
; CHECK-NEXT:  constants: []
; CHECK-NEXT:  machineFunctionInfo: {}
; CHECK-NEXT:  jumpTable:
; CHECK-NEXT:    kind:            label-difference32
; CHECK-NEXT:    entries:
; CHECK-NEXT:      - id:              0
; CHECK-NEXT:        blocks:          [  ]
; CHECK-NEXT:  body:             |
; CHECK-NEXT:    bb.0.entry:
; CHECK-NEXT:      successors: %bb.1(0x80000000)
; CHECK:           B %bb.1
; CHECK:         bb.1.next11:
; CHECK-NEXT:      successors: %bb.2(0x80000000)
; CHECK:           B %bb.2
; CHECK:         bb.2.if.end139:
entry:
  br label %next11
next11:                                           ; preds = %entry
  br i1 false, label %if.then12, label %if.end139
if.then12:                                        ; preds = %next11
  br label %for.cond14
for.cond14:                                       ; preds = %if.then12
  switch i32 undef, label %sw.epilog [
    i32 1, label %sw.bb
    i32 4, label %sw.bb
    i32 6, label %sw.bb
    i32 7, label %sw.bb
    i32 9, label %sw.bb
    i32 12, label %sw.bb
    i32 15, label %sw.bb
    i32 16, label %sw.bb
    i32 24, label %sw.bb
    i32 0, label %sw.bb26
    i32 2, label %sw.bb26
    i32 3, label %sw.bb26
    i32 8, label %sw.bb26
    i32 10, label %sw.bb26
    i32 11, label %sw.bb26
    i32 13, label %sw.bb26
    i32 17, label %sw.bb26
    i32 18, label %sw.bb26
    i32 20, label %sw.bb26
    i32 19, label %sw.bb26
    i32 21, label %sw.bb26
    i32 22, label %sw.bb26
    i32 23, label %sw.bb26
    i32 25, label %sw.bb26
    i32 27, label %sw.bb26
    i32 28, label %sw.bb26
    i32 26, label %sw.bb37
    i32 29, label %sw.bb37
    i32 30, label %sw.bb53
  ]

sw.bb:                                            ; preds = %for.cond14
  unreachable
sw.bb26:                                          ; preds = %for.cond14
  unreachable
sw.bb37:                                          ; preds = %for.cond14
  unreachable
sw.bb53:                                          ; preds = %for.cond14
  unreachable
sw.epilog:                                        ; preds = %for.cond14
  unreachable
if.end139:                                        ; preds = %next11
  unreachable
}
attributes #0 = { noinline optnone }
