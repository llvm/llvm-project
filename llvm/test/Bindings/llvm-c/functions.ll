; RUN: llvm-as < %s | llvm-c-test --module-list-functions | FileCheck %s

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

