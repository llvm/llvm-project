; Tests for correct behavior for non-prevailing resolutions in cases involving
; *_odr and available_externally linkages.

; RUN: llvm-as %s -o %t1
; RUN: llvm-as %S/Inputs/link-odr-availextern-ae.ll -o %t2ae
; RUN: llvm-as %S/Inputs/link-odr-availextern-odr.ll -o %t2odr

; RUN: llvm-lto2 run -o %t3 %t1 %t2ae -r %t1,f,p -r %t2ae,f, -save-temps
; RUN: llvm-dis < %t3.0.0.preopt.bc -o - | FileCheck --check-prefix=PREVAILING %s

; RUN: llvm-lto2 run -o %t3 %t1 %t2odr -r %t1,f,p -r %t2odr,f, -save-temps
; RUN: llvm-dis < %t3.0.0.preopt.bc -o - | FileCheck --check-prefix=PREVAILING %s

; RUN: llvm-lto2 run -o %t3 %t2ae %t1 -r %t1,f,p -r %t2ae,f, -save-temps
; RUN: llvm-dis < %t3.0.0.preopt.bc -o - | FileCheck --check-prefix=PREVAILING %s

; RUN: llvm-lto2 run -o %t3 %t2odr %t1 -r %t1,f,p -r %t2odr,f, -save-temps
; RUN: llvm-dis < %t3.0.0.preopt.bc -o - | FileCheck --check-prefix=PREVAILING %s

; RUN: llvm-lto2 run -o %t3 %t2ae -r %t2ae,f, -save-temps
; RUN: llvm-dis < %t3.0.0.preopt.bc -o - | FileCheck --check-prefix=NONPREVAILING %s

; RUN: llvm-lto2 run -o %t3 %t2odr -r %t2odr,f, -save-temps
; RUN: llvm-dis < %t3.0.0.preopt.bc -o - | FileCheck --check-prefix=NONPREVAILING %s

; RUN: llvm-lto2 run -o %t3 %t2odr %t1 -r %t1,f, -r %t2odr,f, -save-temps
; RUN: llvm-dis < %t3.0.0.preopt.bc -o - | FileCheck --check-prefix=NONPREVAILING %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; PREVAILING: define weak_odr i32 @f()
; PREVAILING-NEXT: ret i32 1
; NONPREVAILING: define available_externally i32 @f()
; NONPREVAILING-NEXT: ret i32 2
define linkonce_odr i32 @f() {
  ret i32 1
}
