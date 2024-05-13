; RUN: opt < %s -passes=indvars -S | FileCheck %s
; PR10946: Vector IVs are not SCEVable.
; CHECK-NOT: phi
define void @test() nounwind {
allocas:
  br i1 undef, label %cif_done, label %for_loop398

cif_done:                                         ; preds = %allocas
  ret void

for_loop398:                                      ; preds = %for_loop398, %allocas
  %storemerge35 = phi <4 x i32> [ %storemerge, %for_loop398 ], [ undef, %allocas ]
  %bincmp431 = icmp sge <4 x i32> %storemerge35, <i32 5, i32 5, i32 5, i32 5>
  %storemerge = bitcast <4 x float> undef to <4 x i32>
  br label %for_loop398
}
