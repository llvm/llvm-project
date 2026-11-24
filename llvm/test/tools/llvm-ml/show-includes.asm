; RUN: llvm-ml -filetype=s -showIncludes /I %S/Inputs %s /Fo %t.s 2> %t.err
; RUN: FileCheck --input-file=%t.err --implicit-check-not=Note: %s

; RUN: llvm-ml -filetype=s /I %S/Inputs %s /Fo %t.s 2> %t.none.err
; RUN: count 0 < %t.none.err

;; A directory named like an include is skipped, not reported.
; RUN: rm -rf %t.dir && mkdir -p %t.dir/show-includes-outer.inc
; RUN: cd %t.dir && llvm-ml -filetype=s -showIncludes /I %S/Inputs %s \
; RUN:   /Fo %t.s 2> %t.dir.err
; RUN: FileCheck --input-file=%t.dir.err --implicit-check-not=Note: %s

;; Nested and repeated includes are all reported; the main file and INCLUDELIB
;; are not.
; CHECK:      Note: including file: {{.*}}show-includes-outer.inc
; CHECK-NEXT: Note: including file: {{.*}}show-includes-inner.inc
; CHECK-NEXT: Note: including file: {{.*}}show-includes-inner.inc

include show-includes-outer.inc
include show-includes-inner.inc

.code
includelib show_includes_lib

end