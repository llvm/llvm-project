; RUN: opt -passes='default<O3>' -S %s | FileCheck %s

@g0 = dso_local local_unnamed_addr global i64 0, align 8
@g15 = dso_local local_unnamed_addr global i64 0, align 8
@g14 = dso_local local_unnamed_addr global i64 0, align 8
@g5 = dso_local local_unnamed_addr global i64 0, align 8
@f4_c7 = dso_local local_unnamed_addr global i8 0, align 1
@g8 = dso_local local_unnamed_addr global i64 0, align 8
@f4_c15 = dso_local local_unnamed_addr global i8 0, align 1
@f4_c8 = dso_local local_unnamed_addr global i8 0, align 1
@__chk = dso_local local_unnamed_addr global i64 0, align 8
@.str = private unnamed_addr constant [20 x i8] c"checksum=0x%016llx\0A\00", align 1

; CHECK-LABEL: define dso_local void @f4(
; CHECK: store i64 5, ptr @__chk
; CHECK-LABEL: define dso_local noundef i32 @main(
define dso_local void @f4() local_unnamed_addr #0 {
entry:
  %0 = load i64, ptr @g0, align 8
  %cmp.not = icmp eq i64 %0, 908375363948206739
  br i1 %cmp.not, label %if.end, label %if.then

if.then:
  store i64 1, ptr @g15, align 8
  br label %if.end

if.end:
  %1 = load i64, ptr @g15, align 8
  store i64 %1, ptr @g14, align 8
  store i64 %1, ptr @g0, align 8
  br label %lbl_b5

lbl_b5:
  %bb13.0 = phi i32 [ 0, %if.end ], [ %bb13.1.lcssa, %if.then8 ]
  %loadedv9.not = phi i1 [ true, %if.end ], [ false, %if.then8 ]
  %ov6.0 = phi i8 [ 0, %if.end ], [ 1, %if.then8 ]
  br label %lbl_b10

lbl_b10:
  %loadedv.not = phi i1 [ true, %lbl_b5 ], [ false, %lbl_b10 ]
  %2 = phi i8 [ 0, %lbl_b5 ], [ 1, %lbl_b10 ]
  %3 = phi i64 [ %1, %lbl_b5 ], [ 0, %lbl_b10 ]
  %bb13.1 = phi i32 [ %bb13.0, %lbl_b5 ], [ %5, %lbl_b10 ]
  %conv1 = trunc i64 %3 to i32
  %4 = tail call range(i32 0, 33) i32 @llvm.ctpop.i32(i32 %conv1)
  %5 = and i32 %4, 1
  br i1 %loadedv.not, label %lbl_b10, label %if.then3

if.then3:
  %.lcssa20 = phi i8 [ %2, %lbl_b10 ]
  %.lcssa18 = phi i64 [ %3, %lbl_b10 ]
  %bb13.1.lcssa = phi i32 [ %bb13.1, %lbl_b10 ]
  %.lcssa = phi i32 [ %5, %lbl_b10 ]
  %tobool4.not = icmp eq i32 %bb13.1.lcssa, 0
  br i1 %tobool4.not, label %return.loopexit, label %if.then8

if.then8:
  br i1 %loadedv9.not, label %lbl_b5, label %if.then10

if.then10:
  %ov6.0.lcssa22 = phi i8 [ %ov6.0, %if.then8 ]
  %.lcssa20.lcssa21 = phi i8 [ %.lcssa20, %if.then8 ]
  %.lcssa18.lcssa19 = phi i64 [ %.lcssa18, %if.then8 ]
  %bb13.1.lcssa.lcssa17 = phi i32 [ %bb13.1.lcssa, %if.then8 ]
  %.lcssa.lcssa16 = phi i32 [ %.lcssa, %if.then8 ]
  %conv2.le.le = zext nneg i32 %.lcssa.lcssa16 to i64
  %storedv5.le = trunc nuw i32 %bb13.1.lcssa.lcssa17 to i8
  store i64 %.lcssa18.lcssa19, ptr @g5, align 8
  store i8 %.lcssa20.lcssa21, ptr @f4_c7, align 1
  store i64 %conv2.le.le, ptr @g8, align 8
  store i8 %ov6.0.lcssa22, ptr @f4_c15, align 1
  store i8 %storedv5.le, ptr @f4_c8, align 1
  store i64 5, ptr @__chk, align 8
  br label %return

return.loopexit:
  %ov6.0.lcssa = phi i8 [ %ov6.0, %if.then3 ]
  %.lcssa20.lcssa = phi i8 [ %.lcssa20, %if.then3 ]
  %.lcssa18.lcssa = phi i64 [ %.lcssa18, %if.then3 ]
  %.lcssa.lcssa = phi i32 [ %.lcssa, %if.then3 ]
  %conv2.le.le14 = zext nneg i32 %.lcssa.lcssa to i64
  store i64 %.lcssa18.lcssa, ptr @g5, align 8
  store i8 %.lcssa20.lcssa, ptr @f4_c7, align 1
  store i64 %conv2.le.le14, ptr @g8, align 8
  store i8 %ov6.0.lcssa, ptr @f4_c15, align 1
  store i8 0, ptr @f4_c8, align 1
  br label %return

return:
  ret void
}

declare i32 @llvm.ctpop.i32(i32)

define dso_local noundef i32 @main() local_unnamed_addr #0 {
entry:
  call void @f4()
  %0 = load i64, ptr @__chk, align 8
  %call = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %0)
  ret i32 0
}

declare noundef i32 @printf(ptr noundef readonly captures(none), ...)

attributes #0 = { noinline nounwind uwtable }
