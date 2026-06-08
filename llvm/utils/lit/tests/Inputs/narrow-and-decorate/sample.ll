; REQUIRES: opt
;
; RUN: opt -S < %s -passes=instcombine | FileCheck %s --check-prefix=ALL
; RUN: opt -S < %s -passes=instcombine | FileCheck %s --check-prefix=ONLYFOO

define i32 @foo(i32 %x) {
  ret i32 %x
}

define i32 @bar(i32 %x) {
  ret i32 %x
}

; ALL-LABEL: @foo
; ALL: ret i32 %x

; ALL-LABEL: @bar
; ALL: ret i32 FILTER_AWAY

; ONLYFOO-LABEL: @foo
; ONLYFOO: ret i32 %x
