; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/a.ll -thin-link-bitcode-file=%t1.thinlink.bc -o %t1.bc
; RUN: opt -thinlto-bc %t/b.ll -thin-link-bitcode-file=%t1.thinlink.bc -o %t2.bc
; RUN: opt -thinlto-bc %t/c.ll -thin-link-bitcode-file=%t1.thinlink.bc -o %t3.bc

; If the prevailing weak symbol is defined in a native file, the IR copies should be dead and propagation should not occur
; RUN: llvm-lto2 run -disable-thinlto-funcattrs=0 %t1.bc %t2.bc %t3.bc -o %t.o \
; RUN:               -r %t1.bc,caller,px -r %t1.bc,callee,lx \
; RUN:               -r %t2.bc,callee,x \
; RUN:               -r %t3.bc,callee,x \
; RUN:               -save-temps

; RUN: llvm-dis -o - %t.o.1.3.import.bc | FileCheck %s

; If the prevailing weak symbol is in an IR file, it should be the one used in the final binary and thus propagation
; should be based off of that copy
; RUN: llvm-lto2 run -O3 -disable-thinlto-funcattrs=0 %t1.bc %t2.bc %t3.bc -o %t.2.o \
; RUN:               -r %t1.bc,caller,px -r %t1.bc,callee,lx \
; RUN:               -r %t2.bc,callee,px \
; RUN:               -r %t3.bc,callee,x \
; RUN:               -save-temps

; RUN: llvm-dis -o - %t.2.o.1.3.import.bc | FileCheck %s --check-prefix=PREVAILING
; RUN: llvm-dis -o - %t.2.o.2.3.import.bc | FileCheck %s --check-prefix=PREVAILING-B
; RUN: llvm-dis -o - %t.2.o.3.3.import.bc | FileCheck %s --check-prefix=PREVAILING-C

;--- a.ll
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @callee()

; CHECK-NOT: Function Attrs: 
; CHECK: define i32 @caller()

; PREVAILING: Function Attrs: norecurse nounwind
; PREVAILING-NEXT: define i32 @caller()
define i32 @caller() {
  %res = call i32 @callee()
  ret i32 %res
}

;--- b.ll
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; PREVAILING-B: define weak i32 @callee()
define weak i32 @callee() {
  ret i32 5
}

;--- c.ll
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; PREVAILING-C: declare i32 @callee()
define weak i32 @callee() {
  ret i32 6
}

