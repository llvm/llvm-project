; RUN: llvm-as %s -o %t.bc
; RUN: llvm-c-test --module-list-functions < %t.bc| FileCheck %s
; RUN: llvm-c-test --module-dump < %t.bc| FileCheck --check-prefix=MOD %s
; RUN: llvm-c-test --lazy-module-dump < %t.bc| FileCheck --check-prefix=LMOD %s

; MOD:      define i32 @X() {
; MOD-NEXT:   entry:

; LMOD:      ; Materializable
; LMOD-NEXT: define i32 @X() {}

define i32 @X() {
entry:
  br label %l1

l1:
  br label %l2

l2:
  br label %l3

l3:
  ret i32 1234
}
;CHECK: FunctionDefinition: X [#bb=4]


define i32 @Z(i32 %a) {
entry:
  %0 = tail call i32 @Y(i32 %a)
  ret i32 %0
}

;CHECK: FunctionDefinition: Z [#bb=1]
;CHECK:  calls: Y
;CHECK:  #isn: 2

declare i32 @Y(i32)
;CHECK: FunctionDeclaration: Y

