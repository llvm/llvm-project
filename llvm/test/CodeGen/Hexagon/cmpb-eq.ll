; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK-NOT: cmpb.eq(r{{[0-9]+}},#-1)

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

%struct.wms_address_s = type { i32, i32, i32, i32, i8, [48 x i8] }

define zeroext i8 @qmi_wmsi_bin_to_addr(ptr %str, i8 zeroext %len, ptr %addr) nounwind optsize {
entry:
  %cmp = icmp eq ptr %str, null
  %cmp2 = icmp eq ptr %addr, null
  %or.cond = or i1 %cmp, %cmp2
  br i1 %or.cond, label %if.then12, label %if.then

if.then:                                          ; preds = %entry
  %dec = add i8 %len, -1
  %cmp3 = icmp ugt i8 %dec, 24
  %tobool27 = icmp eq i8 %dec, 0
  %or.cond31 = or i1 %cmp3, %tobool27
  br i1 %or.cond31, label %if.then12, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %if.then
  %dec626 = add i8 %len, -2
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %if.end21
  %indvars.iv = phi i32 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %if.end21 ]
  %dec630 = phi i8 [ %dec626, %for.body.lr.ph ], [ %dec6, %if.end21 ]
  %str.pn = phi ptr [ %str, %for.body.lr.ph ], [ %str.addr.029, %if.end21 ]
  %str.addr.029 = getelementptr inbounds i8, ptr %str.pn, i32 1
  %0 = load i8, ptr %str.addr.029, align 1, !tbaa !0
  %cmp10 = icmp ugt i8 %0, -49
  br i1 %cmp10, label %if.then12.loopexit, label %if.end21

if.then12.loopexit:                               ; preds = %if.end21, %for.body
  br label %if.then12

if.then12:                                        ; preds = %if.then12.loopexit, %if.then, %entry
  ret i8 0

if.end21:                                         ; preds = %for.body
  %shr24 = lshr i8 %0, 4
  %arrayidx = getelementptr inbounds %struct.wms_address_s, ptr %addr, i32 0, i32 5, i32 %indvars.iv
  store i8 %shr24, ptr %arrayidx, align 1, !tbaa !0
  %dec6 = add i8 %dec630, -1
  %tobool = icmp eq i8 %dec630, 0
  %indvars.iv.next = add i32 %indvars.iv, 1
  br i1 %tobool, label %if.then12.loopexit, label %for.body
}

!0 = !{!"omnipotent char", !1}
!1 = !{!"Simple C/C++ TBAA"}
