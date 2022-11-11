; RUN: opt -disable-verify -passes=instcombine < %s -opt-bisect-limit=0 -opt-bisect-print-ir-path=%t -disable-output 
; RUN: FileCheck %s --check-prefix=LIMIT0 --input-file %t
; RUN: opt -disable-verify -passes=instcombine < %s -opt-bisect-limit=1 -opt-bisect-print-ir-path=%t -disable-output 
; RUN: FileCheck %s --check-prefix=LIMIT1 --input-file %t
; RUN: opt -disable-verify -passes=instcombine < %s -opt-bisect-limit=2 -opt-bisect-print-ir-path=%t -disable-output 
; FIXME: print IR if limit is higher than number of opt-bisect invocations

; Check that we only print the module once
; RUN: opt -disable-verify -passes=instcombine < %s -opt-bisect-limit=1 -opt-bisect-print-ir-path=- -disable-output 2>&1 | FileCheck %s

; LIMIT0: ret i32 %r
; LIMIT0: ret i32 %r

; LIMIT1: ret i32 2
; LIMIT1: ret i32 %r

; CHECK: ModuleID
; CHECK-NOT: ModuleID

define i32 @f1() {
  %r = add i32 1, 1
  ret i32 %r
}

define i32 @f2() {
  %r = add i32 1, 1
  ret i32 %r
}
