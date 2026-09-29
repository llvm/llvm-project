;; Test that function attributes "no-frame-pointer-elim" ("true" or "false") and
;; "no-frame-pointer-elim-non-leaf" (value is ignored) can be upgraded to
;; "frame-pointer".

; RUN: llvm-as %s -o - | llvm-dis - | FileCheck %s

; CHECK: define void @all0() #0
define void @all0() "no-frame-pointer-elim"="true" { ret void }
; CHECK: define void @all1() #1
define void @all1() #0 { ret void }

; CHECK: define void @non_leaf0() #2
define void @non_leaf0() "no-frame-pointer-elim-non-leaf" { ret void }
; CHECK: define void @non_leaf1() #3
define void @non_leaf1() #1 { ret void }

; CHECK: define void @none() #4
define void @none() "no-frame-pointer-elim"="false" { ret void }

;; Don't add "frame-pointer" if neither "no-frame-pointer-elim" nor
;; "no-frame-pointer-elim-non-leaf" is present.
; CHECK: define void @no_attr() {
define void @no_attr() { ret void }

attributes #0 = { readnone "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" }
attributes #1 = { readnone "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf" }

;; Other attributes (e.g. readnone) are unaffected.
; CHECK: attributes #0 = { "frame-pointer"="all" }
; CHECK: attributes #1 = { memory(none) "frame-pointer"="all" }
; CHECK: attributes #2 = { "frame-pointer"="non-leaf" }
; CHECK: attributes #3 = { memory(none) "frame-pointer"="non-leaf" }
; CHECK: attributes #4 = { "frame-pointer"="none" }
