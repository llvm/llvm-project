; RUN: llvm-as < %s >%t.bc

; Check pass remarks emitted to YAML file
; RUN: rm -f %t.yaml
; RUN: llvm-lto2 run -pass-remarks-output=%t.yaml \
; RUN:           -pass-remarks-with-hotness \
; RUN:           -r %t.bc,tinkywinky,p \
; RUN:           -r %t.bc,patatino,px \
; RUN:           -r %t.bc,main,px -o %t.o %t.bc
; RUN: cat %t.yaml | FileCheck %s -check-prefix=YAML

; Check pass remarks emitted to stderr
; RUN: llvm-lto2 run -pass-remarks=inline \
; RUN:           -pass-remarks-with-hotness \
; RUN:           -r %t.bc,tinkywinky,p \
; RUN:           -r %t.bc,patatino,px \
; RUN:           -r %t.bc,main,px -o %t.o %t.bc 2>&1 | FileCheck %s

; YAML: --- !Passed
; YAML-NEXT: Pass:            inline
; YAML-NEXT: Name:            Inlined
; YAML-NEXT: Function:        main
; YAML-NEXT: Hotness:         300
; YAML-NEXT: Args:
; YAML-NEXT:   - Callee:          tinkywinky
; YAML-NEXT:   - String:          ' inlined into '
; YAML-NEXT:   - Caller:          main
; YAML-NEXT:   - String:          ' with '
; YAML-NEXT:   - String:          '(cost='
; YAML-NEXT:   - Cost:            '-15000'
; YAML-NEXT:   - String:          ', threshold='
; YAML-NEXT:   - Threshold:       '337'
; YAML-NEXT:   - String:          ')'
; YAML-NEXT: ...

; CHECK: tinkywinky inlined into main with (cost=-15000, threshold=337) (hotness: 300)

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-scei-ps4"

declare i32 @patatino()

define i32 @tinkywinky() {
  %a = call i32 @patatino()
  ret i32 %a
}

define i32 @main() !prof !0 {
  %i = call i32 @tinkywinky()
  ret i32 %i
}

!0 = !{!"function_entry_count", i64 300}
