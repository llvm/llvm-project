; RUN: opt -S -passes=gvn-hoist < %s | FileCheck %s

; Checking gvn-hoist in case of infinite loops and irreducible control flow.

; Check that bitcast is not hoisted beacuse down safety is not guaranteed.
; CHECK-LABEL: @bazv1
; CHECK: if.then.i:
; CHECK: bitcast
; CHECK-NEXT: load
; CHECK: if.then4.i:
; CHECK: bitcast
; CHECK-NEXT: load

%class.bar = type { i8*, %class.base* }
%class.base = type { i32 (...)** }

; Function Attrs: noreturn nounwind uwtable
define void @bazv1() local_unnamed_addr {
entry:
  %agg.tmp = alloca %class.bar, align 8
  %x.sroa.2.0..sroa_idx2 = getelementptr inbounds %class.bar, %class.bar* %agg.tmp, i64 0, i32 1
  store %class.base* null, %class.base** %x.sroa.2.0..sroa_idx2, align 8
  call void @_Z3foo3bar(%class.bar* nonnull %agg.tmp)
  %0 = load %class.base*, %class.base** %x.sroa.2.0..sroa_idx2, align 8
  %1 = bitcast %class.bar* %agg.tmp to %class.base*
  %cmp.i = icmp eq %class.base* %0, %1
  br i1 %cmp.i, label %if.then.i, label %if.else.i

if.then.i:                                        ; preds = %entry
  %2 = bitcast %class.base* %0 to void (%class.base*)***
  %vtable.i = load void (%class.base*)**, void (%class.base*)*** %2, align 8
  %vfn.i = getelementptr inbounds void (%class.base*)*, void (%class.base*)** %vtable.i, i64 2
  %3 = load void (%class.base*)*, void (%class.base*)** %vfn.i, align 8
  call void %3(%class.base* %0)
  br label %while.cond.preheader

if.else.i:                                        ; preds = %entry
  %tobool.i = icmp eq %class.base* %0, null
  br i1 %tobool.i, label %while.cond.preheader, label %if.then4.i

if.then4.i:                                       ; preds = %if.else.i
  %4 = bitcast %class.base* %0 to void (%class.base*)***
  %vtable6.i = load void (%class.base*)**, void (%class.base*)*** %4, align 8
  %vfn7.i = getelementptr inbounds void (%class.base*)*, void (%class.base*)** %vtable6.i, i64 3
  %5 = load void (%class.base*)*, void (%class.base*)** %vfn7.i, align 8
  call void %5(%class.base* nonnull %0)
  br label %while.cond.preheader

while.cond.preheader:                             ; preds = %if.then.i, %if.else.i, %if.then4.i
  br label %while.cond

while.cond:                                       ; preds = %while.cond.preheader, %while.cond
  %call = call i32 @sleep(i32 10)
  br label %while.cond
}

declare void @_Z3foo3bar(%class.bar*) local_unnamed_addr

declare i32 @sleep(i32) local_unnamed_addr

; Check that the load is hoisted even if it is inside an irreducible control flow
; because the load is anticipable on all paths.

; CHECK-LABEL: @bazv
; CHECK: bb2:
; CHECK-NOT: load
; CHECK-NOT: bitcast

define void @bazv() {
entry:
  %agg.tmp = alloca %class.bar, align 8
  %x= getelementptr inbounds %class.bar, %class.bar* %agg.tmp, i64 0, i32 1
  %0 = load %class.base*, %class.base** %x, align 8
  %1 = bitcast %class.bar* %agg.tmp to %class.base*
  %cmp.i = icmp eq %class.base* %0, %1
  br i1 %cmp.i, label %bb1, label %bb4

bb1:
  %b1 = bitcast %class.base* %0 to void (%class.base*)***
  %i = load void (%class.base*)**, void (%class.base*)*** %b1, align 8
  %vfn.i = getelementptr inbounds void (%class.base*)*, void (%class.base*)** %i, i64 2
  %cmp.j = icmp eq %class.base* %0, %1
  br i1 %cmp.j, label %bb2, label %bb3

bb2:
  %l1 = load void (%class.base*)*, void (%class.base*)** %vfn.i, align 8
  br label %bb3

bb3:
  %l2 = load void (%class.base*)*, void (%class.base*)** %vfn.i, align 8
  br label %bb2

bb4:
  %b2 = bitcast %class.base* %0 to void (%class.base*)***
  ret void
}
