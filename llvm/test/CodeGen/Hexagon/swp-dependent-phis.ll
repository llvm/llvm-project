;RUN: llc -march=hexagon -mv71t -O2 < %s -o - 2>&1 > /dev/null

; Validate that we do not crash while running this test.
;%3:intregs = PHI %21:intregs, %bb.6, %7:intregs, %bb.1 - SU0
;%7:intregs = PHI %21:intregs, %bb.6, %13:intregs, %bb.1 - SU1
;%27:intregs = A2_zxtb %3:intregs - SU2
;%13:intregs = C2_muxri %45:predregs, 0, %46:intreg
;If we have dependent phis, SU0 should be the successor of SU1 not
;the other way around. (it used to be SU1 is the successor of SU0).
;In some cases, SU0 is scheduled earlier than SU1 resulting in bad
;IR as we do not have a value that can be used by SU2.

@global = common dso_local local_unnamed_addr global ptr zeroinitializer, align 4
@global.1 = common dso_local local_unnamed_addr global i32 0, align 4
@global.2 = common dso_local local_unnamed_addr global i16 0, align 2
@global.3 = common dso_local local_unnamed_addr global i16 0, align 2
@global.4 = common dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree norecurse nosync nounwind
define dso_local i16 @wombat(i8 zeroext %arg, i16 %dummy) local_unnamed_addr #0 {
bb:
  %load = load ptr, ptr @global, align 4
  %load1 = load i32, ptr @global.1, align 4
  %add2 = add nsw i32 %load1, -1
  store i32 %add2, ptr @global.1, align 4
  %icmp = icmp eq i32 %load1, 0
  br i1 %icmp, label %bb36, label %bb3

bb3:                                              ; preds = %bb3, %bb
  %phi = phi i32 [ %add30, %bb3 ], [ %add2, %bb ]
  %phi4 = phi i8 [ %phi8, %bb3 ], [ %arg, %bb ]
  %phi5 = phi i16 [ %select23, %bb3 ], [ %dummy, %bb ]
  %phi6 = phi i16 [ %select26, %bb3 ], [ %dummy, %bb ]
  %phi7 = phi i16 [ %select, %bb3 ], [ %dummy, %bb ]
  %phi8 = phi i8 [ %select29, %bb3 ], [ %arg, %bb ]
  %zext = zext i8 %phi4 to i32
  %getelementptr = getelementptr inbounds i32, ptr %load, i32 %zext
  %getelementptr9 = getelementptr inbounds i32, ptr %getelementptr, i32 2
  %ptrtoint = ptrtoint ptr %getelementptr9 to i32
  %trunc = trunc i32 %ptrtoint to i16
  %sext10 = sext i16 %phi7 to i32
  %shl11 = shl i32 %ptrtoint, 16
  %ashr = ashr exact i32 %shl11, 16
  %icmp12 = icmp slt i32 %ashr, %sext10
  %select = select i1 %icmp12, i16 %trunc, i16 %phi7
  %getelementptr13 = getelementptr inbounds i32, ptr %getelementptr, i32 3
  %load14 = load i32, ptr %getelementptr13, align 4
  %shl = shl i32 %load14, 8
  %getelementptr15 = getelementptr inbounds i32, ptr %getelementptr, i32 1
  %load16 = load i32, ptr %getelementptr15, align 4
  %shl17 = shl i32 %load16, 16
  %ashr18 = ashr exact i32 %shl17, 16
  %add = add nsw i32 %ashr18, %load14
  %lshr = lshr i32 %add, 8
  %or = or i32 %lshr, %shl
  %sub = sub i32 %or, %load16
  %trunc19 = trunc i32 %sub to i16
  %sext = sext i16 %phi5 to i32
  %shl20 = shl i32 %sub, 16
  %ashr21 = ashr exact i32 %shl20, 16
  %icmp22 = icmp sgt i32 %ashr21, %sext
  %select23 = select i1 %icmp22, i16 %trunc19, i16 %phi5
  %sext24 = sext i16 %phi6 to i32
  %icmp25 = icmp slt i32 %ashr21, %sext24
  %select26 = select i1 %icmp25, i16 %trunc19, i16 %phi6
  %icmp27 = icmp eq i8 %phi8, 0
  %add28 = add i8 %phi8, -1
  %select29 = select i1 %icmp27, i8 0, i8 %add28
  %add30 = add nsw i32 %phi, -1
  %icmp31 = icmp eq i32 %phi, 0
  br i1 %icmp31, label %bb32, label %bb3

bb32:                                             ; preds = %bb3
  store i16 %trunc, ptr @global.2, align 2
  store i16 %trunc19, ptr @global.3, align 2
  store i32 -1, ptr @global.1, align 4
  %sext33 = sext i16 %select to i32
  %sext34 = sext i16 %select23 to i32
  %sext35 = sext i16 %select26 to i32
  br label %bb36

bb36:                                             ; preds = %bb32, %bb
  %phi37 = phi i32 [ %sext33, %bb32 ], [ 0, %bb ]
  %phi38 = phi i32 [ %sext35, %bb32 ], [ 0, %bb ]
  %phi39 = phi i32 [ %sext34, %bb32 ], [ 0, %bb ]
  %sub40 = sub nsw i32 %phi39, %phi38
  %icmp41 = icmp slt i32 %sub40, %phi37
  br i1 %icmp41, label %bb43, label %bb42

bb42:                                             ; preds = %bb36
  store i32 0, ptr @global.4, align 4
  br label %bb43

bb43:                                             ; preds = %bb42, %bb36
  ret i16 %dummy
}
