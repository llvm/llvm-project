; RUN: llc < %s -march=x86-64 -mcpu=corei7 -mtriple=x86_64-pc-win64 | FileCheck %s

; CHECK: merge_stores_can
; CHECK: callq foo
; CHECK: xorps %xmm0, %xmm0
; CHECK-NEXT: movups  %xmm0
; CHECK-NEXT: movl 36(%rsp), %ebp
; CHECK: callq foo
; CHECK: ret
declare i32 @foo([10 x i32]* )

define i32 @merge_stores_can() nounwind ssp {
  %object1 = alloca [10 x i32]

  %ret0 = call i32 @foo([10 x i32]* %object1) nounwind

  %O1_1 = getelementptr [10 x i32], [10 x i32]* %object1, i64 0, i32 1
  %O1_2 = getelementptr [10 x i32], [10 x i32]* %object1, i64 0, i32 2
  %O1_3 = getelementptr [10 x i32], [10 x i32]* %object1, i64 0, i32 3
  %O1_4 = getelementptr [10 x i32], [10 x i32]* %object1, i64 0, i32 4
  %ld_ptr = getelementptr [10 x i32], [10 x i32]* %object1, i64 0, i32 9

  store i32 0, i32* %O1_1
  store i32 0, i32* %O1_2
  %ret = load  i32,  i32* %ld_ptr  ; <--- does not alias.
  store i32 0, i32* %O1_3
  store i32 0, i32* %O1_4

  %ret1 = call i32 @foo([10 x i32]* %object1) nounwind

  ret i32 %ret
}

; CHECK: merge_stores_cant
; CHECK-NOT: xorps %xmm0, %xmm0
; CHECK-NOT: movups  %xmm0
; CHECK: ret
define i32 @merge_stores_cant([10 x i32]* %in0, [10 x i32]* %in1) nounwind ssp {

  %O1_1 = getelementptr [10 x i32], [10 x i32]* %in1, i64 0, i32 1
  %O1_2 = getelementptr [10 x i32], [10 x i32]* %in1, i64 0, i32 2
  %O1_3 = getelementptr [10 x i32], [10 x i32]* %in1, i64 0, i32 3
  %O1_4 = getelementptr [10 x i32], [10 x i32]* %in1, i64 0, i32 4
  %ld_ptr = getelementptr [10 x i32], [10 x i32]* %in0, i64 0, i32 2

  store i32 0, i32* %O1_1
  store i32 0, i32* %O1_2
  %ret = load  i32,  i32* %ld_ptr  ;  <--- may alias
  store i32 0, i32* %O1_3
  store i32 0, i32* %O1_4

  ret i32 %ret
}
