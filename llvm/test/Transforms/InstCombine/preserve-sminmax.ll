; RUN: opt < %s -instcombine -S | FileCheck %s

; Instcombine normally would fold the sdiv into the comparison,
; making "icmp slt i32 %h, 2", but in this case the sdiv has
; another use, so it wouldn't a big win, and it would also
; obfuscate an otherise obvious smax pattern to the point where
; other analyses wouldn't recognize it.

define i32 @foo(i32 %h) {
  %sd = sdiv i32 %h, 2
  %t = icmp slt i32 %sd, 1
  %r = select i1 %t, i32 %sd, i32 1
  ret i32 %r
}

; CHECK:  %sd = sdiv i32 %h, 2
; CHECK:  %t = icmp slt i32 %sd, 1
; CHECK:  %r = select i1 %t, i32 %sd, i32 1
; CHECK:  ret i32 %r

define i32 @bar(i32 %h) {
  %sd = sdiv i32 %h, 2
  %t = icmp sgt i32 %sd, 1
  %r = select i1 %t, i32 %sd, i32 1
  ret i32 %r
}

; CHECK:  %sd = sdiv i32 %h, 2
; CHECK:  %t = icmp sgt i32 %sd, 1
; CHECK:  %r = select i1 %t, i32 %sd, i32 1
; CHECK:  ret i32 %r

