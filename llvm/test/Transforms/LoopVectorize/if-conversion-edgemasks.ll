; RUN: opt -S -loop-vectorize < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@a = global i32* null, align 8
@b = global i32* null, align 8
@c = global i32* null, align 8

; Don't create an exponetial IR for the edge masks needed when if-converting
; this code.

; PR16472

; CHECK-NOT: %6000000 =

define void @_Z3fn4i(i32 %p1) {
entry:
  %cmp88 = icmp sgt i32 %p1, 0
  br i1 %cmp88, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %0 = load i32*, i32** @b, align 8
  %1 = load i32*, i32** @a, align 8
  %2 = load i32*, i32** @c, align 8
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %_ZL3fn3ii.exit58 ]
  %arrayidx = getelementptr inbounds i32, i32* %0, i64 %indvars.iv
  %3 = load i32, i32* %arrayidx, align 4  %4 = trunc i64 %indvars.iv to i32
  %and.i = and i32 %4, 1
  %tobool.i.i = icmp eq i32 %and.i, 0
  br i1 %tobool.i.i, label %if.end.i, label %if.then.i

if.then.i:
  %and.i.i = lshr i32 %3, 2
  %and.lobit.i.i = and i32 %and.i.i, 1
  %5 = xor i32 %and.lobit.i.i, 1
  %or.i.i = or i32 %5, %3
  %cmp.i = icmp sgt i32 %or.i.i, 0
  %conv.i = zext i1 %cmp.i to i32
  br label %if.end.i

if.end.i:
  %tobool.i87 = phi i1 [ true, %if.then.i ], [ false, %for.body ]
  %p1.addr.0.i = phi i32 [ %conv.i, %if.then.i ], [ %3, %for.body ]
  %6 = trunc i64 %indvars.iv to i32
  %and1.i = and i32 %6, 7
  %tobool2.i = icmp eq i32 %and1.i, 0
  br i1 %tobool2.i, label %if.end7.i, label %if.then3.i

if.then3.i:
  %p1.addr.0.lobit.i = lshr i32 %p1.addr.0.i, 31
  %and6.i = and i32 %p1.addr.0.i, 1
  %or.i = or i32 %p1.addr.0.lobit.i, %and6.i
  br label %if.end7.i

if.end7.i:
  %p1.addr.1.i = phi i32 [ %or.i, %if.then3.i ], [ %p1.addr.0.i, %if.end.i ]
  br i1 %tobool.i87, label %if.then10.i, label %if.end13.i

if.then10.i:
  %cmp11.i = icmp sgt i32 %p1.addr.1.i, 0
  %conv12.i = zext i1 %cmp11.i to i32
  br label %if.end13.i

if.end13.i:
  %p1.addr.2.i = phi i32 [ %conv12.i, %if.then10.i ], [ %p1.addr.1.i, %if.end7.i ]
  br i1 %tobool.i.i, label %_Z3fn2iii.exit, label %if.then16.i

if.then16.i:
  %and17.i = lshr i32 %p1.addr.2.i, 3
  %and17.lobit.i = and i32 %and17.i, 1
  br label %_Z3fn2iii.exit

_Z3fn2iii.exit:
  %p1.addr.3.i = phi i32 [ %and17.lobit.i, %if.then16.i ], [ %p1.addr.2.i, %if.end13.i ]
  %7 = trunc i64 %indvars.iv to i32
  %shr.i = ashr i32 %7, 1
  %and.i18.i = and i32 %shr.i, 1
  %tobool.i19.i = icmp ne i32 %and.i18.i, 0
  br i1 %tobool.i19.i, label %if.then.i20.i, label %if.end.i.i

if.then.i20.i:
  %cmp.i.i = icmp sgt i32 %p1.addr.3.i, 0
  %conv.i.i = zext i1 %cmp.i.i to i32
  br label %if.end.i.i

if.end.i.i:
  %p1.addr.0.i21.i = phi i32 [ %conv.i.i, %if.then.i20.i ], [ %p1.addr.3.i, %_Z3fn2iii.exit ]
  %and1.i.i = and i32 %shr.i, 7
  %tobool2.i.i = icmp eq i32 %and1.i.i, 0
  br i1 %tobool2.i.i, label %if.end7.i.i, label %if.then3.i.i

if.then3.i.i:
  %p1.addr.0.lobit.i.i = lshr i32 %p1.addr.0.i21.i, 31
  %and6.i.i = and i32 %p1.addr.0.i21.i, 1
  %or.i22.i = or i32 %p1.addr.0.lobit.i.i, %and6.i.i
  br label %if.end7.i.i

if.end7.i.i:
  %p1.addr.1.i.i = phi i32 [ %or.i22.i, %if.then3.i.i ], [ %p1.addr.0.i21.i, %if.end.i.i ]
  br i1 %tobool.i19.i, label %if.then10.i.i, label %if.end13.i.i

if.then10.i.i:
  %cmp11.i.i = icmp sgt i32 %p1.addr.1.i.i, 0
  %conv12.i.i = zext i1 %cmp11.i.i to i32
  br label %if.end13.i.i

if.end13.i.i:
  %p1.addr.2.i.i = phi i32 [ %conv12.i.i, %if.then10.i.i ], [ %p1.addr.1.i.i, %if.end7.i.i ]
  %and14.i.i = and i32 %shr.i, 5
  %tobool15.i.i = icmp eq i32 %and14.i.i, 0
  br i1 %tobool15.i.i, label %_Z3fn2iii.exit.i, label %if.then16.i.i

if.then16.i.i:
  %and17.i.i = lshr i32 %p1.addr.2.i.i, 3
  %and17.lobit.i.i = and i32 %and17.i.i, 1
  br label %_Z3fn2iii.exit.i

_Z3fn2iii.exit.i:
  %p1.addr.3.i.i = phi i32 [ %and17.lobit.i.i, %if.then16.i.i ], [ %p1.addr.2.i.i, %if.end13.i.i ]
  %8 = trunc i64 %indvars.iv to i32
  %tobool.i11.i = icmp eq i32 %8, 0
  br i1 %tobool.i11.i, label %_ZL3fn3ii.exit, label %if.then.i15.i

if.then.i15.i:
  %and.i12.i = lshr i32 %p1.addr.3.i.i, 2
  %and.lobit.i13.i = and i32 %and.i12.i, 1
  %9 = xor i32 %and.lobit.i13.i, 1
  %or.i14.i = or i32 %9, %p1.addr.3.i.i
  br label %_ZL3fn3ii.exit

_ZL3fn3ii.exit:
  %p1.addr.0.i16.i = phi i32 [ %or.i14.i, %if.then.i15.i ], [ %p1.addr.3.i.i, %_Z3fn2iii.exit.i ]
  %arrayidx2 = getelementptr inbounds i32, i32* %1, i64 %indvars.iv
  store i32 %p1.addr.0.i16.i, i32* %arrayidx2, align 4  %arrayidx4 = getelementptr inbounds i32, i32* %0, i64 %indvars.iv
  %10 = load i32, i32* %arrayidx4, align 4  br i1 %tobool.i.i, label %_Z3fn1ii.exit.i26, label %if.then.i.i21

if.then.i.i21:
  %and.i.i18 = lshr i32 %10, 2
  %and.lobit.i.i19 = and i32 %and.i.i18, 1
  %11 = xor i32 %and.lobit.i.i19, 1
  %or.i.i20 = or i32 %11, %10
  br label %_Z3fn1ii.exit.i26

_Z3fn1ii.exit.i26:
  %p1.addr.0.i.i22 = phi i32 [ %or.i.i20, %if.then.i.i21 ], [ %10, %_ZL3fn3ii.exit ]
  br i1 %tobool.i87, label %if.then.i63, label %if.end.i67

if.then.i63:
  %cmp.i61 = icmp sgt i32 %p1.addr.0.i.i22, 0
  %conv.i62 = zext i1 %cmp.i61 to i32
  br label %if.end.i67

if.end.i67:
  %p1.addr.0.i64 = phi i32 [ %conv.i62, %if.then.i63 ], [ %p1.addr.0.i.i22, %_Z3fn1ii.exit.i26 ]
  br i1 %tobool2.i, label %if.end7.i73, label %if.then3.i71

if.then3.i71:
  %p1.addr.0.lobit.i68 = lshr i32 %p1.addr.0.i64, 31
  %and6.i69 = and i32 %p1.addr.0.i64, 1
  %or.i70 = or i32 %p1.addr.0.lobit.i68, %and6.i69
  br label %if.end7.i73

if.end7.i73:
  %p1.addr.1.i72 = phi i32 [ %or.i70, %if.then3.i71 ], [ %p1.addr.0.i64, %if.end.i67 ]
  br i1 %tobool.i87, label %if.then10.i76, label %if.end13.i80

if.then10.i76:
  %cmp11.i74 = icmp sgt i32 %p1.addr.1.i72, 0
  %conv12.i75 = zext i1 %cmp11.i74 to i32
  br label %if.end13.i80

if.end13.i80:
  %p1.addr.2.i77 = phi i32 [ %conv12.i75, %if.then10.i76 ], [ %p1.addr.1.i72, %if.end7.i73 ]
  br i1 %tobool.i.i, label %_Z3fn2iii.exit85, label %if.then16.i83

if.then16.i83:
  %and17.i81 = lshr i32 %p1.addr.2.i77, 3
  %and17.lobit.i82 = and i32 %and17.i81, 1
  br label %_Z3fn2iii.exit85

_Z3fn2iii.exit85:
  %p1.addr.3.i84 = phi i32 [ %and17.lobit.i82, %if.then16.i83 ], [ %p1.addr.2.i77, %if.end13.i80 ]
  br i1 %tobool.i19.i, label %if.then.i20.i29, label %if.end.i.i33

if.then.i20.i29:
  %cmp.i.i27 = icmp sgt i32 %p1.addr.3.i84, 0
  %conv.i.i28 = zext i1 %cmp.i.i27 to i32
  br label %if.end.i.i33

if.end.i.i33:
  %p1.addr.0.i21.i30 = phi i32 [ %conv.i.i28, %if.then.i20.i29 ], [ %p1.addr.3.i84, %_Z3fn2iii.exit85 ]
  br i1 %tobool2.i.i, label %if.end7.i.i39, label %if.then3.i.i37

if.then3.i.i37:
  %p1.addr.0.lobit.i.i34 = lshr i32 %p1.addr.0.i21.i30, 31
  %and6.i.i35 = and i32 %p1.addr.0.i21.i30, 1
  %or.i22.i36 = or i32 %p1.addr.0.lobit.i.i34, %and6.i.i35
  br label %if.end7.i.i39

if.end7.i.i39:
  %p1.addr.1.i.i38 = phi i32 [ %or.i22.i36, %if.then3.i.i37 ], [ %p1.addr.0.i21.i30, %if.end.i.i33 ]
  br i1 %tobool.i19.i, label %if.then10.i.i42, label %if.end13.i.i46

if.then10.i.i42:
  %cmp11.i.i40 = icmp sgt i32 %p1.addr.1.i.i38, 0
  %conv12.i.i41 = zext i1 %cmp11.i.i40 to i32
  br label %if.end13.i.i46

if.end13.i.i46:
  %p1.addr.2.i.i43 = phi i32 [ %conv12.i.i41, %if.then10.i.i42 ], [ %p1.addr.1.i.i38, %if.end7.i.i39 ]
  br i1 %tobool15.i.i, label %_Z3fn2iii.exit.i52, label %if.then16.i.i49

if.then16.i.i49:
  %and17.i.i47 = lshr i32 %p1.addr.2.i.i43, 3
  %and17.lobit.i.i48 = and i32 %and17.i.i47, 1
  br label %_Z3fn2iii.exit.i52

_Z3fn2iii.exit.i52:
  %p1.addr.3.i.i50 = phi i32 [ %and17.lobit.i.i48, %if.then16.i.i49 ], [ %p1.addr.2.i.i43, %if.end13.i.i46 ]
  br i1 %tobool.i11.i, label %_ZL3fn3ii.exit58, label %if.then.i15.i56

if.then.i15.i56:
  %and.i12.i53 = lshr i32 %p1.addr.3.i.i50, 2
  %and.lobit.i13.i54 = and i32 %and.i12.i53, 1
  %12 = xor i32 %and.lobit.i13.i54, 1
  %or.i14.i55 = or i32 %12, %p1.addr.3.i.i50
  br label %_ZL3fn3ii.exit58

_ZL3fn3ii.exit58:
  %p1.addr.0.i16.i57 = phi i32 [ %or.i14.i55, %if.then.i15.i56 ], [ %p1.addr.3.i.i50, %_Z3fn2iii.exit.i52 ]
  %arrayidx7 = getelementptr inbounds i32, i32* %2, i64 %indvars.iv
  store i32 %p1.addr.0.i16.i57, i32* %arrayidx7, align 4  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, %p1
  br i1 %exitcond, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:
  br label %for.end

for.end:
  ret void
}
