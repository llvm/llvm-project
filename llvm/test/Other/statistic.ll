; RUN: opt < %s -disable-output -p=instsimplify -stats -stats-json 2>&1 | FileCheck %s --check-prefix=JSON
; RUN: opt < %s -disable-output -p=instsimplify -stats -stats-json -info-output-file %t && FileCheck %s < %t --check-prefix=JSON
; RUN: opt < %s -disable-output -p=instsimplify -stats 2>&1 | FileCheck %s --check-prefix=DEFAULT
; RUN: opt < %s -disable-output -p=instsimplify -stats -info-output-file %t && FileCheck %s < %t --check-prefix=DEFAULT
; REQUIRES: asserts

; JSON: {
; JSON-DAG:   "instsimplify.NumSimplified": 1
; JSON: }

; DEFAULT: 1 instsimplify - Number of redundant instructions removed

define i32 @foo() {
  %res = add i32 5, 4
  ret i32 %res
}
