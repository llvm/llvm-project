; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=-crbits | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s -check-prefix=CHECK-CRB
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.lua_TValue.17.692 = type { %union.Value.16.691, i32 }
%union.Value.16.691 = type { ptr }
%union.GCObject.15.690 = type { %struct.lua_State.14.689 }
%struct.lua_State.14.689 = type { ptr, i8, i8, i8, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i16, i16, i8, i8, i32, i32, ptr, %struct.lua_TValue.17.692, %struct.lua_TValue.17.692, ptr, ptr, ptr, i64 }
%struct.global_State.10.685 = type { %struct.stringtable.0.675, ptr, ptr, i8, i8, i32, ptr, ptr, ptr, ptr, ptr, ptr, %struct.Mbuffer.1.676, i64, i64, i64, i64, i32, i32, ptr, %struct.lua_TValue.17.692, ptr, %struct.UpVal.3.678, [9 x ptr], [17 x ptr] }
%struct.stringtable.0.675 = type { ptr, i32, i32 }
%struct.Mbuffer.1.676 = type { ptr, i64, i64 }
%struct.UpVal.3.678 = type { ptr, i8, i8, ptr, %union.anon.2.677 }
%union.anon.2.677 = type { %struct.lua_TValue.17.692 }
%struct.Table.7.682 = type { ptr, i8, i8, i8, i8, ptr, ptr, ptr, ptr, ptr, i32 }
%struct.Node.6.681 = type { %struct.lua_TValue.17.692, %union.TKey.5.680 }
%union.TKey.5.680 = type { %struct.anon.0.4.679 }
%struct.anon.0.4.679 = type { %union.Value.16.691, i32, ptr }
%union.TString.9.684 = type { %struct.anon.1.8.683 }
%struct.anon.1.8.683 = type { ptr, i8, i8, i8, i32, i64 }
%struct.CallInfo.11.686 = type { ptr, ptr, ptr, ptr, i32, i32 }
%struct.lua_Debug.12.687 = type { i32, ptr, ptr, ptr, ptr, i32, i32, i32, i32, [60 x i8], i32 }
%struct.lua_longjmp.13.688 = type opaque

define void @lua_xmove(i32 signext %n) #0 {
entry:
  br i1 undef, label %for.end, label %if.end

if.end:                                           ; preds = %entry
  br i1 undef, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %if.end
  br label %for.body

for.body:                                         ; preds = %for.body.for.body_crit_edge, %for.body.lr.ph
  %0 = phi ptr [ undef, %for.body.lr.ph ], [ %.pre, %for.body.for.body_crit_edge ]
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body.for.body_crit_edge ]
  %tt = getelementptr inbounds %struct.lua_TValue.17.692, ptr %0, i64 %indvars.iv, i32 1
  %1 = load i32, ptr %tt, align 4
  %2 = add i32 %1, %1
  store i32 %2, ptr %tt, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body.for.body_crit_edge

for.body.for.body_crit_edge:                      ; preds = %for.body
  %.pre = load ptr, ptr undef, align 8
  br label %for.body

for.end:                                          ; preds = %for.body, %if.end, %entry
  ret void

; CHECK: @lua_xmove
; CHECK: bnelr
; CHECK: bnelr
; CHECK: bdzlr
; CHECK-NOT: blr

; CHECK-CRB: @lua_xmove
; CHECK-CRB: bclr 12,
; CHECK-CRB: bclr 12,
; CHECK-CRB: bdzlr
; CHECK-CRB-NOT: blr
}

attributes #0 = { nounwind }
