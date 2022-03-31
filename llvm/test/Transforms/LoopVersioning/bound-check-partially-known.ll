; RUN: opt -aa-pipeline=basic-aa -passes=loop-versioning -S %s | FileCheck %s

%struct.foo = type { [32000 x double], [32000 x double] }

@global = external global %struct.foo, align 32

define void @bound_check_partially_known_1(i32 %N) {
; CHECK-LABEL: @bound_check_partially_known_1(
; CHECK-NEXT:  loop.lver.check:
; CHECK-NEXT:    [[N_EXT:%.*]] = zext i32 [[N:%.*]] to i64
; CHECK-NEXT:    [[SCEVGEP:%.*]] = getelementptr [[STRUCT_FOO:%.*]], %struct.foo* @global, i64 0, i32 0, i64 [[N_EXT]]
; CHECK-NEXT:    [[SCEVGEP1:%.*]] = bitcast double* [[SCEVGEP]] to i8*
; CHECK-NEXT:    [[TMP0:%.*]] = shl nuw nsw i64 [[N_EXT]], 1
; CHECK-NEXT:    [[SCEVGEP2:%.*]] = getelementptr [[STRUCT_FOO]], %struct.foo* @global, i64 0, i32 0, i64 [[TMP0]]
; CHECK-NEXT:    [[SCEVGEP23:%.*]] = bitcast double* [[SCEVGEP2]] to i8*
; CHECK-NEXT:    [[TMP1:%.*]] = add nuw nsw i64 [[N_EXT]], 32000
; CHECK-NEXT:    [[SCEVGEP4:%.*]] = getelementptr [[STRUCT_FOO]], %struct.foo* @global, i64 0, i32 0, i64 [[TMP1]]
; CHECK-NEXT:    [[SCEVGEP45:%.*]] = bitcast double* [[SCEVGEP4]] to i8*
; CHECK-NEXT:    [[BOUND1:%.*]] = icmp ult i8* bitcast (%struct.foo* @global to i8*), [[SCEVGEP23]]
; CHECK-NEXT:    [[BOUND06:%.*]] = icmp ult i8* [[SCEVGEP1]], [[SCEVGEP45]]
; CHECK-NEXT:    [[BOUND17:%.*]] = icmp ult i8* bitcast (double* getelementptr inbounds ([[STRUCT_FOO]], %struct.foo* @global, i64 0, i32 1, i64 0) to i8*), [[SCEVGEP23]]
; CHECK-NEXT:    [[FOUND_CONFLICT8:%.*]] = and i1 [[BOUND06]], [[BOUND17]]
; CHECK-NEXT:    br i1 [[FOUND_CONFLICT8]], label %loop.ph.lver.orig, label %loop.ph
;
entry:
  %N.ext = zext i32 %N to i64
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.0.iv = getelementptr inbounds %struct.foo, %struct.foo* @global, i64 0, i32 0, i64 %iv
  %l.0 = load double, double* %gep.0.iv, align 8
  %gep.1.iv = getelementptr inbounds %struct.foo, %struct.foo* @global, i64 0, i32 1, i64 %iv
  %l.1 = load double, double* %gep.1.iv, align 8
  %add = fadd double %l.0, %l.1
  %iv.N = add nuw nsw i64 %iv, %N.ext
  %gep.0.iv.N = getelementptr inbounds %struct.foo, %struct.foo* @global, i64 0, i32 0, i64 %iv.N
  store double %add, double* %gep.0.iv.N, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %N.ext
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}
