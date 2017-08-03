; RUN: llc < %s -march=x86-64 -mcpu=corei7-avx -mattr=+avx
; Various missing patterns causing crashes.
; rdar://10538793

define void @t1() nounwind {
entry:
  br label %loop.cond

loop.cond:                                        ; preds = %t1.exit, %entry
  br i1 false, label %return, label %loop

loop:                                             ; preds = %loop.cond
  br i1 undef, label %0, label %t1.exit

; <label>:0                                       ; preds = %loop
  %1 = load <16 x i32>, <16 x i32> addrspace(1)* undef, align 64
  %2 = shufflevector <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>, <16 x i32> %1, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 16, i32 0, i32 0>
  store <16 x i32> %2, <16 x i32> addrspace(1)* undef, align 64
  br label %t1.exit

t1.exit:                                 ; preds = %0, %loop
  br label %loop.cond

return:                                           ; preds = %loop.cond
  ret void
}

define void @t2() nounwind {
  br i1 undef, label %1, label %4

; <label>:1                                       ; preds = %0
  %2 = load <16 x i32>, <16 x i32> addrspace(1)* undef, align 64
  %3 = shufflevector <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>, <16 x i32> %2, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 20, i32 0, i32 0, i32 0, i32 0>
  store <16 x i32> %3, <16 x i32> addrspace(1)* undef, align 64
  br label %4

; <label>:4                                       ; preds = %1, %0
  ret void
}

define void @t3() nounwind {
entry:
  br label %loop.cond

loop.cond:                                        ; preds = %t2.exit, %entry
  br i1 false, label %return, label %loop

loop:                                             ; preds = %loop.cond
  br i1 undef, label %0, label %t2.exit

; <label>:0                                       ; preds = %loop
  %1 = shufflevector <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>, <16 x i32> undef, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 25, i32 0>
  %2 = load <16 x i32>, <16 x i32> addrspace(1)* undef, align 64
  %3 = shufflevector <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>, <16 x i32> %2, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 28, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  store <16 x i32> %3, <16 x i32> addrspace(1)* undef, align 64
  br label %t2.exit

t2.exit:                                 ; preds = %0, %loop
  br label %loop.cond

return:                                           ; preds = %loop.cond
  ret void
}

define <3 x i64> @t4() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64> addrspace(1)* undef, align 16
  %1 = extractelement <2 x i64> %0, i32 0
  %2 = insertelement <3 x i64> <i64 undef, i64 0, i64 0>, i64 %1, i32 0
  ret <3 x i64> %2
}

define void @t5() nounwind {
entry:
  %0 = shufflevector <2 x i64> zeroinitializer, <2 x i64> undef, <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %1 = shufflevector <8 x i64> <i64 0, i64 0, i64 0, i64 undef, i64 undef, i64 0, i64 0, i64 0>, <8 x i64> %0, <8 x i32> <i32 0, i32 1, i32 2, i32 9, i32 8, i32 5, i32 6, i32 7>
  store <8 x i64> %1, <8 x i64> addrspace(1)* undef, align 64

  ret void
}
