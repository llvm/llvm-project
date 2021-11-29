; Test that constant structs are folded.
; RUN: opt %s -passes=sccp -S | FileCheck %s

define internal {i64} @struct1() {
  %a = insertvalue {i64} undef, i64 24, 0
  ret {i64} %a
}

; CHECK: define internal { i64 } @struct1() {
; CHECK-NEXT:   ret { i64 } { i64 24 }
; CHECK-NEXT: }

define internal {i64, i64} @struct2() {
  %a = insertvalue {i64, i64} undef, i64 24, 0
  ret {i64, i64} %a
}

; CHECK: define internal { i64, i64 } @struct2() {
; CHECK-NEXT:  ret { i64, i64 } { i64 24, i64 undef }
; CHECK-NEXT: }

define internal {i64, i64, i64} @struct3(i64 %x) {
  %a = insertvalue {i64, i64, i64} undef, i64 24, 0
  %b = insertvalue {i64, i64, i64} %a, i64 36, 1
  %c = insertvalue {i64, i64, i64} %b, i64 %x, 2
  ret {i64, i64, i64} %c
}

; CHECK: define internal { i64, i64, i64 } @struct3(i64 %x) {
; CHECK-NEXT:  %c = insertvalue { i64, i64, i64 } { i64 24, i64 36, i64 undef }, i64 %x, 2
; CHECK-NEXT:  ret { i64, i64, i64 } %c
; CHECK-NEXT: }

; Test(s) for overdefined values.
define internal {i64, i32} @struct4(i32 %x) {
  %a = insertvalue {i64, i32} {i64 12, i32 24}, i32 %x, 1
  ret {i64, i32} %a
}

; CHECK: define internal { i64, i32 } @struct4(i32 %x) {
; CHECK-NEXT:  %a = insertvalue { i64, i32 } { i64 12, i32 24 }, i32 %x, 1
; CHECK-NEXT:  ret { i64, i32 } %a
; CHECK-NEXT: }

define internal {i32} @struct5(i32 %x) {
  %a = insertvalue {i32} undef, i32 %x, 0
  ret {i32} %a
}

; CHECK: define internal { i32 } @struct5(i32 %x) {
; CHECK-NEXT:  %a = insertvalue { i32 } undef, i32 %x, 0
; CHECK-NEXT:  ret { i32 } %a
; CHECK-NEXT: }


define internal {i32} @struct6({i32} %x) {
  %a = insertvalue {i32} %x, i32 12, 0
  ret {i32} %a
}

; CHECK: define internal { i32 } @struct6({ i32 } %x) {
; CHECK-NEXT:  ret { i32 } { i32 12 }
; CHECK-NEXT: }

define internal {i16} @struct7() {
  %a = insertvalue {i16} {i16 4}, i16 7, 0
  ret {i16} %a
}

; CHECK: define internal { i16 } @struct7() {
; CHECK-NEXT:  ret { i16 } { i16 7 }
; CHECK-NEXT: }
