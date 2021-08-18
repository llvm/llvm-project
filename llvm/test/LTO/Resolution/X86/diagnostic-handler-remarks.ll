; Test of LTO with opt remarks YAML output.

; First try with Regular LTO
; RUN: llvm-as < %s >%t.bc
; RUN: rm -f %t.yaml
; RUN: llvm-lto2 run -pass-remarks-output=%t.yaml \
; RUN:           -pass-remarks-filter=inline \
; RUN:           -r %t.bc,tinkywinky,p \
; RUN:           -r %t.bc,patatino,px \
; RUN:           -r %t.bc,main,px -o %t.o %t.bc
; RUN: cat %t.yaml | FileCheck %s -check-prefix=YAML

; Try again with ThinLTO
; RUN: opt -module-summary %s -o %t.bc
; RUN: rm -f %t.thin.1.yaml
; RUN: llvm-lto2 run -pass-remarks-output=%t \
; RUN:           -pass-remarks-filter=inline \
; RUN:           -r %t.bc,tinkywinky,p \
; RUN:           -r %t.bc,patatino,px \
; RUN:           -r %t.bc,main,px -o %t.o %t.bc
; RUN: cat %t.thin.1.yaml | FileCheck %s -check-prefix=YAML

; YAML:      --- !Passed
; YAML-NEXT: Pass:            inline
; YAML-NEXT: Name:            Inlined
; YAML-NEXT: Function:        main
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          ''''
; YAML-NEXT:   - Callee:          tinkywinky
; YAML-NEXT:   - String:          ''' inlined into '''
; YAML-NEXT:   - Caller:          main
; YAML-NEXT:   - String:          ''''
; YAML-NEXT:   - String:          ' with '
; YAML-NEXT:   - String:          '(cost='
; YAML-NEXT:   - Cost:            '-15000'
; YAML-NEXT:   - String:          ', threshold='
; YAML-NEXT:   - Threshold:       '337'
; YAML-NEXT:   - String:          ')'
; YAML-NEXT: ...

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-scei-ps4"

declare i32 @patatino()

define i32 @tinkywinky() {
  %a = call i32 @patatino()
  ret i32 %a
}

define i32 @main() {
  %i = call i32 @tinkywinky()
  ret i32 %i
}
