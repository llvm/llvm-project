; RUN: llvm-as %s -o /dev/null
; This testcase comes from the following really simple c file:
;; int foo[30000]
;;; We should not be soo slow for such a simple case!

@foo = global [30000 x i32] zeroinitializer		; <[30000 x i32]*> [#uses=0]

declare void @__main()
