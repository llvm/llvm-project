; RUN: opt < %s -mtriple=x86_64-unknown-unknown -passes=mergeicmps -verify-dom-info -S 2>&1 | FileCheck %s
; RUN: opt < %s -mtriple=x86_64-unknown-unknown -passes='mergeicmps,expand-memcmp' -verify-dom-info -S 2>&1 | FileCheck %s --check-prefix=EXPANDED

; adjacent byte pointer accesses compared to constants, should be merged into single memcmp, spanning multiple basic blocks

; CHECK: [[MEMCMP_OP:@memcmp_const_op]] = private constant <{ i8, i8, i8 }> <{ i8 -1, i8 -56, i8 -66 }>

; Global should be removed once its constant has been folded.
; EXPANDED-NOT: [[MEMCMP_OP:@memcmp_const_op]] = private constant <{ i8, i8, i8 }> <{ i8 -1, i8 -56, i8 -66 }>

define zeroext i1 @test(ptr nocapture noundef nonnull dereferenceable(3) %p) local_unnamed_addr #0 {
; CHECK-LABEL: @test(
; CHECK-NEXT:  "entry+land.lhs.true+land.rhs":
; CHECK-NEXT:    [[MEMCMP:%.*]] = call i32 @memcmp(ptr [[p:%.*]], ptr [[MEMCMP_OP]], i64 3)
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq i32 [[MEMCMP]], 0
; CHECK-NEXT:    br label [[LAND_END5:%.*]]
; CHECK:       land.end:
; CHECK-NEXT:    ret i1 [[TMP1]]
;
; EXPANDED-LABEL: define zeroext i1 @test(
; EXPANDED-SAME: ptr nocapture noundef nonnull dereferenceable(3) [[P:%.*]]) local_unnamed_addr {
; EXPANDED-NEXT:  "entry+land.lhs.true+land.rhs":
; EXPANDED-NEXT:    [[TMP0:%.*]] = load i16, ptr [[P]], align 1
; EXPANDED-NEXT:    [[TMP8:%.*]] = xor i16 [[TMP0]], -14081
; EXPANDED-NEXT:    [[TMP2:%.*]] = getelementptr i8, ptr [[P]], i64 2
; EXPANDED-NEXT:    [[TMP3:%.*]] = load i8, ptr [[TMP2]], align 1
; EXPANDED-NEXT:    [[TMP4:%.*]] = zext i8 [[TMP3]] to i16
; EXPANDED-NEXT:    [[TMP5:%.*]] = xor i16 [[TMP4]], 190
; EXPANDED-NEXT:    [[TMP6:%.*]] = or i16 [[TMP8]], [[TMP5]]
; EXPANDED-NEXT:    [[TMP7:%.*]] = icmp ne i16 [[TMP6]], 0
; EXPANDED-NEXT:    [[CMP:%.*]] = zext i1 [[TMP7]] to i32
; EXPANDED-NEXT:    [[RES:%.*]] = icmp eq i32 [[CMP]], 0
; EXPANDED-NEXT:    br label %[[LAND_END:.*]]
; EXPANDED:       [[LAND_END]]:
; EXPANDED-NEXT:    ret i1 [[RES]]
;
entry:
  %0 = load i8, ptr %p, align 1
  %cmp = icmp eq i8 %0, -1
  br i1 %cmp, label %land.lhs.true, label %land.end

land.lhs.true:                                    ; preds = %entry
  %arrayidx1 = getelementptr inbounds nuw i8, ptr %p, i64 1
  %1 = load i8, ptr %arrayidx1, align 1
  %cmp5 = icmp eq i8 %1, -56
  br i1 %cmp5, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %land.lhs.true
  %arrayidx2 = getelementptr inbounds nuw i8, ptr %p, i64 2
  %2 = load i8, ptr %arrayidx2, align 1
  %cmp8 = icmp eq i8 %2, -66
  br label %land.end

land.end:                                         ; preds = %land.rhs, %land.lhs.true, %entry
  %3 = phi i1 [ false, %land.lhs.true ], [ false, %entry ], [ %cmp8, %land.rhs ]
  ret i1 %3
}
