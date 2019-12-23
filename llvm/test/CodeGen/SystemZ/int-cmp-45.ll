; Test that compares are omitted if CC already has the right value
; (z196 version).
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 -no-integrated-as | FileCheck %s

; Addition provides enough for comparisons with zero if we know no
; signed overflow happens, which is when the "nsw" flag is set.
; First test the EQ case with LOC.
define i32 @f1(i32 %a, i32 %b, i32 *%cptr) {
; CHECK-LABEL: f1:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: loce %r3, 0(%r4)
; CHECK: br %r14
  %add = add nsw i32 %a, 1000000
  %cmp = icmp eq i32 %add, 0
  %c = load i32, i32 *%cptr
  %arg = select i1 %cmp, i32 %c, i32 %b
  call void asm sideeffect "blah $0", "{r3}"(i32 %arg)
  ret i32 %add
}

; ...and again with STOC.
define i32 @f2(i32 %a, i32 %b, i32 *%cptr) {
; CHECK-LABEL: f2:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: stoce %r3, 0(%r4)
; CHECK: br %r14
  %add = add nsw i32 %a, 1000000
  %cmp = icmp eq i32 %add, 0
  %c = load i32, i32 *%cptr
  %newval = select i1 %cmp, i32 %b, i32 %c
  store i32 %newval, i32 *%cptr
  ret i32 %add
}

; Reverse the select order and test with LOCR.
define i32 @f3(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f3:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: locrlh %r3, %r4
; CHECK: br %r14
  %add = add nsw i32 %a, 1000000
  %cmp = icmp eq i32 %add, 0
  %arg = select i1 %cmp, i32 %b, i32 %c
  call void asm sideeffect "blah $0", "{r3}"(i32 %arg)
  ret i32 %add
}

; ...and again with LOC.
define i32 @f4(i32 %a, i32 %b, i32 *%cptr) {
; CHECK-LABEL: f4:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: loclh %r3, 0(%r4)
; CHECK: br %r14
  %add = add nsw i32 %a, 1000000
  %cmp = icmp eq i32 %add, 0
  %c = load i32, i32 *%cptr
  %arg = select i1 %cmp, i32 %b, i32 %c
  call void asm sideeffect "blah $0", "{r3}"(i32 %arg)
  ret i32 %add
}

; ...and again with STOC.
define i32 @f5(i32 %a, i32 %b, i32 *%cptr) {
; CHECK-LABEL: f5:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: stoclh %r3, 0(%r4)
; CHECK: br %r14
  %add = add nsw i32 %a, 1000000
  %cmp = icmp eq i32 %add, 0
  %c = load i32, i32 *%cptr
  %newval = select i1 %cmp, i32 %c, i32 %b
  store i32 %newval, i32 *%cptr
  ret i32 %add
}

; Change the EQ in f3 to NE.
define i32 @f6(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f6:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: locre %r3, %r4
; CHECK: br %r14
  %add = add nsw i32 %a, 1000000
  %cmp = icmp ne i32 %add, 0
  %arg = select i1 %cmp, i32 %b, i32 %c
  call void asm sideeffect "blah $0", "{r3}"(i32 %arg)
  ret i32 %add
}

; ...and again with LOC.
define i32 @f7(i32 %a, i32 %b, i32 *%cptr) {
; CHECK-LABEL: f7:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: loce %r3, 0(%r4)
; CHECK: br %r14
  %add = add nsw i32 %a, 1000000
  %cmp = icmp ne i32 %add, 0
  %c = load i32, i32 *%cptr
  %arg = select i1 %cmp, i32 %b, i32 %c
  call void asm sideeffect "blah $0", "{r3}"(i32 %arg)
  ret i32 %add
}

; ...and again with STOC.
define i32 @f8(i32 %a, i32 %b, i32 *%cptr) {
; CHECK-LABEL: f8:
; CHECK: afi %r2, 1000000
; CHECK-NEXT: stoce %r3, 0(%r4)
; CHECK: br %r14
  %add = add nsw i32 %a, 1000000
  %cmp = icmp ne i32 %add, 0
  %c = load i32, i32 *%cptr
  %newval = select i1 %cmp, i32 %c, i32 %b
  store i32 %newval, i32 *%cptr
  ret i32 %add
}
