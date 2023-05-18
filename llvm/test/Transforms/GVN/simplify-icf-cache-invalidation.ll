; RUN: opt -passes=gvn -S < %s | FileCheck %s

; CHECK: define {{.*}}@eggs

%struct.zot = type { ptr }
%struct.wombat = type { ptr }
%struct.baz = type { i8, ptr }

@global = hidden unnamed_addr constant ptr @quux

declare ptr @f()

define hidden void @eggs(ptr %arg, i1 %arg2, ptr %arg3, i32 %arg4, ptr %arg5) unnamed_addr align 2 {
bb:
  %tmp = alloca %struct.wombat, align 8
  store ptr @global, ptr %arg, align 8, !invariant.group !0
  br i1 %arg2, label %bb4, label %bb2

bb2:                                              ; preds = %bb
  %tmp3 = atomicrmw sub ptr %arg3, i32 %arg4 acq_rel, align 4
  br label %bb4

bb4:                                              ; preds = %bb2, %bb
  %tmp5 = load ptr, ptr %arg5, align 8
  %tmp6 = getelementptr inbounds %struct.baz, ptr %tmp5, i64 0, i32 1
  br i1 %arg2, label %bb9, label %bb7

bb7:                                              ; preds = %bb4
  %tmp8 = tail call ptr @f()
  br label %bb9

bb9:                                              ; preds = %bb7, %bb4
  %tmp10 = load ptr, ptr %arg5, align 8
  %tmp13 = load ptr, ptr %arg, align 8, !invariant.group !0
  %tmp15 = load ptr, ptr %tmp13, align 8
  tail call void %tmp15(ptr %arg, i1 %arg2)
  %tmp17 = load ptr, ptr %tmp, align 8
  %tmp18 = icmp eq ptr %tmp17, null
  ret void
}

; Function Attrs: nounwind willreturn
declare hidden void @quux(ptr, i1) unnamed_addr #0 align 2

attributes #0 = { nounwind willreturn }

!0 = !{}
