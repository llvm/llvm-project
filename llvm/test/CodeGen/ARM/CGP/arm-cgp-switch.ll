; RUN: llc -mtriple=thumbv7em %s -arm-disable-cgp=false -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7-linux-android %s -arm-disable-cgp=false -o - | FileCheck %s

; CHECK-LABEL: truncate_source_phi_switch
; CHECK: ldrb
; CHECK: uxtb
define void @truncate_source_phi_switch(i8* %memblock, i8* %store, i16 %arg) {
entry:
  %pre = load i8, i8* %memblock, align 1
  %conv = trunc i16 %arg to i8
  br label %header

header:
  %phi.0 = phi i8 [ %pre, %entry ], [ %count, %latch ]
  %phi.1 = phi i8 [ %conv, %entry ], [ %phi.3, %latch ]
  %phi.2 = phi i8 [ 0, %entry], [ %count, %latch ]
  switch i8 %phi.0, label %default [
    i8 43, label %for.inc.i
    i8 45, label %for.inc.i.i
  ]

for.inc.i:
  %xor = xor i8 %phi.1, 1
  br label %latch

for.inc.i.i:
  %and = and i8 %phi.1, 3
  br label %latch

default:
  %sub = sub i8 %phi.0, 1
  %cmp2 = icmp ugt i8 %sub, 4
  br i1 %cmp2, label %latch, label %exit

latch:
  %phi.3 = phi i8 [ %xor, %for.inc.i ], [ %and, %for.inc.i.i ], [ %phi.2, %default ]
  %count = add nuw i8 %phi.2, 1
  store i8 %count, i8* %store, align 1
  br label %header

exit:
  ret void
}

; CHECK-LABEL: icmp_switch_source:
; CHECK-NOT: uxt
define i16 @icmp_switch_source(i16 zeroext %arg) {
entry:
  %conv = add nuw i16 %arg, 15
  %mul = mul nuw nsw i16 %conv, 3
  switch i16 %arg, label %default [
    i16 0, label %sw.bb
    i16 1, label %sw.bb.i
  ]

sw.bb:
  %cmp0 = icmp ult i16 %mul, 127
  %select = select i1 %cmp0, i16 %mul, i16 127
  br label %exit

sw.bb.i:
  %cmp1 = icmp ugt i16 %mul, 34
  %select.i = select i1 %cmp1, i16 %mul, i16 34
  br label %exit

default:
  br label %exit

exit:
  %res = phi i16 [ %select, %sw.bb ], [ %select.i, %sw.bb.i ], [ %mul, %default ]
  ret i16 %res
}

; CHECK-LABEL: icmp_switch_narrow_source:
; CHECK-NOT: uxt
define i16 @icmp_switch_narrow_source(i8 zeroext %arg) {
entry:
  %conv = zext i8 %arg to i16
  %add = add nuw i16 %conv, 15
  %mul = mul nuw nsw i16 %add, 3
  switch i8 %arg, label %default [
    i8 0, label %sw.bb
    i8 1, label %sw.bb.i
  ]

sw.bb:
  %cmp0 = icmp ult i16 %mul, 127
  %select = select i1 %cmp0, i16 %mul, i16 127
  br label %exit

sw.bb.i:
  %cmp1 = icmp ugt i16 %mul, 34
  %select.i = select i1 %cmp1, i16 %mul, i16 34
  br label %exit

default:
  br label %exit

exit:
  %res = phi i16 [ %select, %sw.bb ], [ %select.i, %sw.bb.i ], [ %mul, %default ]
  ret i16 %res
}

; CHECK-LABEL: icmp_switch_trunc:
; CHECK-NOT: uxt
define i16 @icmp_switch_trunc(i16 zeroext %arg) {
entry:
  %conv = add nuw i16 %arg, 15
  %mul = mul nuw nsw i16 %conv, 3
  %trunc = trunc i16 %arg to i3
  switch i3 %trunc, label %default [
    i3 0, label %sw.bb
    i3 1, label %sw.bb.i
  ]

sw.bb:
  %cmp0 = icmp ult i16 %mul, 127
  %select = select i1 %cmp0, i16 %mul, i16 127
  br label %exit

sw.bb.i:
  %cmp1 = icmp ugt i16 %mul, 34
  %select.i = select i1 %cmp1, i16 %mul, i16 34
  br label %exit

default:
  br label %exit

exit:
  %res = phi i16 [ %select, %sw.bb ], [ %select.i, %sw.bb.i ], [ %mul, %default ]
  ret i16 %res
}

%class.ae = type { i8 }
%class.x = type { i8 }
%class.v = type { %class.q }
%class.q = type { i16 }
declare %class.x* @_ZNK2ae2afEv(%class.ae*) local_unnamed_addr
declare %class.v* @_ZN1x2acEv(%class.x*) local_unnamed_addr

; CHECK-LABEL: trunc_i16_i9_switch
; CHECK-NOT: uxt
define i32 @trunc_i16_i9_switch(%class.ae* %this) {
entry:
  %call = tail call %class.x* @_ZNK2ae2afEv(%class.ae* %this)
  %call2 = tail call %class.v* @_ZN1x2acEv(%class.x* %call)
  %0 = getelementptr inbounds %class.v, %class.v* %call2, i32 0, i32 0, i32 0
  %1 = load i16, i16* %0, align 2
  %2 = trunc i16 %1 to i9
  %trunc = and i9 %2, -64
  switch i9 %trunc, label %cleanup.fold.split [
    i9 0, label %cleanup
    i9 -256, label %if.then7
  ]

if.then7:
  %3 = and i16 %1, 7
  %tobool = icmp eq i16 %3, 0
  %cond = select i1 %tobool, i32 2, i32 1
  br label %cleanup

cleanup.fold.split:
  br label %cleanup

cleanup:
  %retval.0 = phi i32 [ %cond, %if.then7 ], [ 0, %entry ], [ 2, %cleanup.fold.split ]
  ret i32 %retval.0
}
