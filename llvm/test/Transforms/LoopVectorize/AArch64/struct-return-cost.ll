; RUN: opt -passes=loop-vectorize,instcombine,simplifycfg < %s -S -o - | FileCheck %s --check-prefix=CHECK
; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=CHECK-COST
; REQUIRES: asserts

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; CHECK-COST-LABEL: struct_return_widen
; CHECK-COST: LV: Found an estimated cost of 10 for VF 1 For instruction:   %call = tail call { half, half } @foo(half %in_val) #0
; CHECK-COST: Cost of 10 for VF 2: WIDEN-CALL ir<%call> = call  @foo(ir<%in_val>) (using library function: fixed_vec_foo)
; CHECK-COST: Cost of 58 for VF 4: REPLICATE ir<%call> = call @foo(ir<%in_val>)
; CHECK-COST: Cost of 122 for VF 8: REPLICATE ir<%call> = call @foo(ir<%in_val>)

define void @struct_return_widen(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; CHECK-LABEL: define void @struct_return_widen(
; CHECK-SAME: ptr noalias [[IN:%.*]], ptr noalias writeonly [[OUT_A:%.*]], ptr noalias writeonly [[OUT_B:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[VECTOR_BODY:.*]]
; CHECK:       [[VECTOR_BODY]]:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %[[ENTRY]] ], [ [[INDEX_NEXT:%.*]], %[[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds half, ptr [[IN]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds nuw i8, ptr [[TMP0]], i64 4
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <2 x half>, ptr [[TMP0]], align 2
; CHECK-NEXT:    [[WIDE_LOAD1:%.*]] = load <2 x half>, ptr [[TMP1]], align 2
; CHECK-NEXT:    [[TMP2:%.*]] = call { <2 x half>, <2 x half> } @fixed_vec_foo(<2 x half> [[WIDE_LOAD]])
; CHECK-NEXT:    [[TMP3:%.*]] = call { <2 x half>, <2 x half> } @fixed_vec_foo(<2 x half> [[WIDE_LOAD1]])
; CHECK-NEXT:    [[TMP4:%.*]] = extractvalue { <2 x half>, <2 x half> } [[TMP2]], 0
; CHECK-NEXT:    [[TMP5:%.*]] = extractvalue { <2 x half>, <2 x half> } [[TMP3]], 0
; CHECK-NEXT:    [[TMP6:%.*]] = extractvalue { <2 x half>, <2 x half> } [[TMP2]], 1
; CHECK-NEXT:    [[TMP7:%.*]] = extractvalue { <2 x half>, <2 x half> } [[TMP3]], 1
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds half, ptr [[OUT_A]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP9:%.*]] = getelementptr inbounds nuw i8, ptr [[TMP8]], i64 4
; CHECK-NEXT:    store <2 x half> [[TMP4]], ptr [[TMP8]], align 2
; CHECK-NEXT:    store <2 x half> [[TMP5]], ptr [[TMP9]], align 2
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr inbounds half, ptr [[OUT_B]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr inbounds nuw i8, ptr [[TMP10]], i64 4
; CHECK-NEXT:    store <2 x half> [[TMP6]], ptr [[TMP10]], align 2
; CHECK-NEXT:    store <2 x half> [[TMP7]], ptr [[TMP11]], align 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP12:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1024
; CHECK-NEXT:    br i1 [[TMP12]], label %[[EXIT:.*]], label %[[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds half, ptr %in, i64 %iv
  %in_val = load half, ptr %arrayidx, align 2
  %call = tail call { half, half } @foo(half %in_val) #0
  %extract_a = extractvalue { half, half } %call, 0
  %extract_b = extractvalue { half, half } %call, 1
  %arrayidx2 = getelementptr inbounds half, ptr %out_a, i64 %iv
  store half %extract_a, ptr %arrayidx2, align 2
  %arrayidx4 = getelementptr inbounds half, ptr %out_b, i64 %iv
  store half %extract_b, ptr %arrayidx4, align 2
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

; CHECK-COST-LABEL: struct_return_replicate
; CHECK-COST: LV: Found an estimated cost of 10 for VF 1 For instruction:   %call = tail call { half, half } @foo(half %in_val) #0
; CHECK-COST: Cost of 26 for VF 2: REPLICATE ir<%call> = call @foo(ir<%in_val>)
; CHECK-COST: Cost of 58 for VF 4: REPLICATE ir<%call> = call @foo(ir<%in_val>)
; CHECK-COST: Cost of 122 for VF 8: REPLICATE ir<%call> = call @foo(ir<%in_val>)

define void @struct_return_replicate(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; CHECK-LABEL: define void @struct_return_replicate(
; CHECK-SAME: ptr noalias [[IN:%.*]], ptr noalias writeonly [[OUT_A:%.*]], ptr noalias writeonly [[OUT_B:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[VECTOR_BODY:.*]]
; CHECK:       [[VECTOR_BODY]]:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %[[ENTRY]] ], [ [[INDEX_NEXT:%.*]], %[[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds half, ptr [[IN]], i64 [[INDEX]]
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <2 x half>, ptr [[TMP0]], align 2
; CHECK-NEXT:    [[TMP1:%.*]] = extractelement <2 x half> [[WIDE_LOAD]], i64 0
; CHECK-NEXT:    [[TMP2:%.*]] = tail call { half, half } @foo(half [[TMP1]]) #[[ATTR0:[0-9]+]]
; CHECK-NEXT:    [[TMP3:%.*]] = extractelement <2 x half> [[WIDE_LOAD]], i64 1
; CHECK-NEXT:    [[TMP4:%.*]] = tail call { half, half } @foo(half [[TMP3]]) #[[ATTR0]]
; CHECK-NEXT:    [[TMP5:%.*]] = extractvalue { half, half } [[TMP2]], 0
; CHECK-NEXT:    [[TMP6:%.*]] = insertelement <2 x half> poison, half [[TMP5]], i64 0
; CHECK-NEXT:    [[TMP7:%.*]] = extractvalue { half, half } [[TMP2]], 1
; CHECK-NEXT:    [[TMP8:%.*]] = insertelement <2 x half> poison, half [[TMP7]], i64 0
; CHECK-NEXT:    [[TMP9:%.*]] = extractvalue { half, half } [[TMP4]], 0
; CHECK-NEXT:    [[TMP10:%.*]] = insertelement <2 x half> [[TMP6]], half [[TMP9]], i64 1
; CHECK-NEXT:    [[TMP11:%.*]] = extractvalue { half, half } [[TMP4]], 1
; CHECK-NEXT:    [[TMP12:%.*]] = insertelement <2 x half> [[TMP8]], half [[TMP11]], i64 1
; CHECK-NEXT:    [[TMP13:%.*]] = getelementptr inbounds half, ptr [[OUT_A]], i64 [[INDEX]]
; CHECK-NEXT:    store <2 x half> [[TMP10]], ptr [[TMP13]], align 2
; CHECK-NEXT:    [[TMP14:%.*]] = getelementptr inbounds half, ptr [[OUT_B]], i64 [[INDEX]]
; CHECK-NEXT:    store <2 x half> [[TMP12]], ptr [[TMP14]], align 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; CHECK-NEXT:    [[TMP15:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1024
; CHECK-NEXT:    br i1 [[TMP15]], label %[[EXIT:.*]], label %[[VECTOR_BODY]], !llvm.loop [[LOOP3:![0-9]+]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds half, ptr %in, i64 %iv
  %in_val = load half, ptr %arrayidx, align 2
  ; #3 does not have a fixed-size vector mapping (so replication is used)
  %call = tail call { half, half } @foo(half %in_val) #1
  %extract_a = extractvalue { half, half } %call, 0
  %extract_b = extractvalue { half, half } %call, 1
  %arrayidx2 = getelementptr inbounds half, ptr %out_a, i64 %iv
  store half %extract_a, ptr %arrayidx2, align 2
  %arrayidx4 = getelementptr inbounds half, ptr %out_b, i64 %iv
  store half %extract_b, ptr %arrayidx4, align 2
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

declare { half, half } @foo(half)

declare { <2 x half>, <2 x half> } @fixed_vec_foo(<2 x half>)
declare { <vscale x 4 x half>, <vscale x 4 x half> } @scalable_vec_masked_foo(<vscale x 4 x half>, <vscale x 4 x i1>)

attributes #0 = { nounwind "vector-function-abi-variant"="_ZGVnN2v_foo(fixed_vec_foo)" }
attributes #1 = { nounwind "vector-function-abi-variant"="_ZGVsMxv_foo(scalable_vec_masked_foo)" }
