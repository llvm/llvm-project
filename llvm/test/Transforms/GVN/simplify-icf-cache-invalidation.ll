; RUN: opt -passes=gvn -S < %s | FileCheck %s

; CHECK: define {{.*}}@eggs

%struct.zot = type { i32 (...)** }
%struct.wombat = type { i8* }
%struct.baz = type { i8, i8* }

@global = hidden unnamed_addr constant i8* bitcast (void (%struct.zot*, i1)* @quux to i8*)

declare i8* @f()

define hidden void @eggs(%struct.zot* %arg, i1 %arg2, i32* %arg3, i32 %arg4, %struct.baz** %arg5) unnamed_addr align 2 {
bb:
  %tmp = alloca %struct.wombat, align 8
  %tmp1 = getelementptr %struct.zot, %struct.zot* %arg, i64 0, i32 0
  store i32 (...)** bitcast (i8** @global to i32 (...)**), i32 (...)*** %tmp1, align 8, !invariant.group !0
  br i1 %arg2, label %bb4, label %bb2

bb2:                                              ; preds = %bb
  %tmp3 = atomicrmw sub i32* %arg3, i32 %arg4 acq_rel, align 4
  br label %bb4

bb4:                                              ; preds = %bb2, %bb
  %tmp5 = load %struct.baz*, %struct.baz** %arg5, align 8
  %tmp6 = getelementptr inbounds %struct.baz, %struct.baz* %tmp5, i64 0, i32 1
  br i1 %arg2, label %bb9, label %bb7

bb7:                                              ; preds = %bb4
  %tmp8 = tail call i8* @f()
  br label %bb9

bb9:                                              ; preds = %bb7, %bb4
  %tmp10 = load %struct.baz*, %struct.baz** %arg5, align 8
  %tmp11 = getelementptr inbounds %struct.baz, %struct.baz* %tmp10, i64 0, i32 0
  %tmp12 = bitcast %struct.zot* %arg to void (%struct.zot*, i1)***
  %tmp13 = load void (%struct.zot*, i1)**, void (%struct.zot*, i1)*** %tmp12, align 8, !invariant.group !0
  %tmp14 = getelementptr inbounds void (%struct.zot*, i1)*, void (%struct.zot*, i1)** %tmp13, i64 0
  %tmp15 = load void (%struct.zot*, i1)*, void (%struct.zot*, i1)** %tmp14, align 8
  tail call void %tmp15(%struct.zot* %arg, i1 %arg2)
  %tmp16 = getelementptr inbounds %struct.wombat, %struct.wombat* %tmp, i64 0, i32 0
  %tmp17 = load i8*, i8** %tmp16, align 8
  %tmp18 = icmp eq i8* %tmp17, null
  ret void
}

; Function Attrs: nounwind willreturn
declare hidden void @quux(%struct.zot*, i1) unnamed_addr #0 align 2

attributes #0 = { nounwind willreturn }

!0 = !{}
