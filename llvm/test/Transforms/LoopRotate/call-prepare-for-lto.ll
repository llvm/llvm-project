; RUN: opt -S -loop-rotate < %s | FileCheck --check-prefix=FULL %s
; RUN: opt -S -loop-rotate -rotation-prepare-for-lto < %s | FileCheck --check-prefix=PREPARE %s
; RUN: opt -S -passes='require<targetir>,require<assumptions>,loop(loop-rotate)' < %s | FileCheck --check-prefix=FULL %s
; RUN: opt -S -passes='require<targetir>,require<assumptions>,loop(loop-rotate)' -rotation-prepare-for-lto < %s | FileCheck --check-prefix=PREPARE %s

; Test case to make sure loop-rotate avoids rotating during the prepare-for-lto
; stage, when the header contains a call which may be inlined during the LTO stage.
define void @test_prepare_for_lto() {
; FULL-LABEL: @test_prepare_for_lto(
; FULL-NEXT:  entry:
; FULL-NEXT:    %array = alloca [20 x i32], align 16
; FULL-NEXT:    %arrayidx = getelementptr inbounds [20 x i32], [20 x i32]* %array, i64 0, i64 0
; FULL-NEXT:    call void @may_be_inlined()
; FULL-NEXT:    br label %for.body
;
; PREPARE-LABEL: @test_prepare_for_lto(
; PREPARE-NEXT:  entry:
; PREPARE-NEXT:    %array = alloca [20 x i32], align 16
; PREPARE-NEXT:    br label %for.cond
;
entry:
  %array = alloca [20 x i32], align 16
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, 100
  %arrayidx = getelementptr inbounds [20 x i32], [20 x i32]* %array, i64 0, i64 0
  call void @may_be_inlined()
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %arrayidx, align 16
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

define void @may_be_inlined() {
  ret void
}
