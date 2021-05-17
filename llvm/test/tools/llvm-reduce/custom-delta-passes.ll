; RUN: llvm-reduce --delta-passes=module-inline-asm --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-NOCHANGE %s < %t
; RUN: llvm-reduce --delta-passes=function-bodies --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-CHANGE %s < %t
; RUN: llvm-reduce --delta-passes=function-bodies,module-inline-asm --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-CHANGE %s < %t

; RUN: not llvm-reduce --delta-passes=foo --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not llvm-reduce --delta-passes='function-bodies;module-inline-asm' --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t 2>&1 | FileCheck %s --check-prefix=ERROR

; RUN: llvm-reduce --print-delta-passes --test FileCheck %s 2>&1 | FileCheck %s --check-prefix=PRINT

; CHECK-INTERESTINGNESS: @f

; CHECK-NOCHANGE: define {{.*}} @f
; CHECK-CHANGE: declare {{.*}} @f

; ERROR: unknown

; PRINT: function-bodies

define void @f() {
  ret void
}
