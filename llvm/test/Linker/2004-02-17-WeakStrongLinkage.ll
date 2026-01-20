; RUN: llvm-as < %s > %t.out2.bc
; RUN: echo "@me = global ptr zeroinitializer" | llvm-as > %t.out1.bc
; RUN: llvm-link %t.out1.bc %t.out2.bc -o /dev/null

@me = weak global ptr zeroinitializer		; <ptr> [#uses=0]


