; RUN: opt -passes='loop-mssa(licm<allowspeculation>)' -S %s | FileCheck --check-prefixes=COMMON,SPEC_ON %s
; RUN: opt -passes='loop-mssa(licm<no-allowspeculation>)' -S %s | FileCheck --check-prefixes=COMMON,SPEC_OFF %s
; RUN: opt -passes='loop-mssa(lnicm<allowspeculation>)' -S %s | FileCheck --check-prefixes=COMMON,SPEC_ON %s
; RUN: opt -passes='loop-mssa(lnicm<no-allowspeculation>)' -S %s | FileCheck --check-prefixes=COMMON,SPEC_OFF %s

define void @test([10 x i32]* %ptr, i32 %N) {
; COMMON-LABEL: @test(
; COMMON-NEXT:  entry:
; SPEC_ON-NEXT:   [[GEP:%.*]] = getelementptr [10 x i32], [10 x i32]* [[PTR:%.*]], i32 0, i32 1
; COMMON-NEXT:    br label [[LOOP_HEADER:%.*]]
; COMMON:       loop.header:
; COMMON-NEXT:    [[IV:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[IV_NEXT:%.*]], [[LOOP_LATCH:%.*]] ]
; COMMON-NEXT:    [[CMP:%.*]] = icmp ult i32 [[IV]], [[N:%.*]]
; COMMON-NEXT:    br i1 [[CMP]], label [[LOOP_LATCH]], label [[EXIT:%.*]]
; COMMON:       loop.latch:
; SPEC_OFF-NEXT:  [[GEP:%.*]] = getelementptr [10 x i32], [10 x i32]* [[PTR:%.*]], i32 0, i32 1
; COMMON-NEXT:    [[GEP_IV:%.*]] = getelementptr i32, i32* [[GEP]], i32 [[IV]]
; COMMON-NEXT:    store i32 9999, i32* [[GEP_IV]], align 4
; COMMON-NEXT:    [[IV_NEXT]] = add i32 [[IV]], 1
; COMMON-NEXT:    br label [[LOOP_HEADER]]
; COMMON:       exit:
; COMMON-NEXT:    ret void
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %cmp = icmp ult i32 %iv, %N
  br i1 %cmp, label %loop.latch, label %exit

loop.latch:
  %gep = getelementptr [10 x i32], [10 x i32]* %ptr, i32 0, i32 1
  %gep.iv = getelementptr i32, i32* %gep, i32 %iv
  store i32 9999, i32* %gep.iv
  %iv.next = add i32 %iv, 1
  br label %loop.header

exit:
  ret void
}
