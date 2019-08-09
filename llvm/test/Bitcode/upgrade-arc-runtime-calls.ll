; Test that calls to ARC runtime functions are converted to intrinsic calls if
; the bitcode has the arm64 retainAutoreleasedReturnValueMarker metadata.

; upgrade-arc-runtime-calls.bc and upgrade-mrr-runtime-calls.bc are identical
; except that the former has the arm64 retainAutoreleasedReturnValueMarker
; metadata.

; RUN: llvm-dis < %S/upgrade-arc-runtime-calls.bc | FileCheck -check-prefixes=ARC %s
; RUN: llvm-dis < %S/upgrade-mrr-runtime-calls.bc | FileCheck -check-prefixes=MRR %s

// ARC: define void @testRuntimeCalls(i8* %[[A:.*]], i8** %[[B:.*]], i8** %[[C:.*]]) {
// ARC: %[[V0:.*]] = tail call i8* @llvm.objc.autorelease(i8* %[[A]])
// ARC-NEXT: tail call void @llvm.objc.autoreleasePoolPop(i8* %[[A]])
// ARC-NEXT: %[[V1:.*]] = tail call i8* @llvm.objc.autoreleasePoolPush()
// ARC-NEXT: %[[V2:.*]] = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %[[A]])
// ARC-NEXT: tail call void @llvm.objc.copyWeak(i8** %[[B]], i8** %[[C]])
// ARC-NEXT: tail call void @llvm.objc.destroyWeak(i8** %[[B]])
// ARC-NEXT: %[[V3:.*]] = tail call i8* @llvm.objc.initWeak(i8** %[[B]], i8* %[[A]])
// ARC-NEXT: %[[V4:.*]] = tail call i8* @llvm.objc.loadWeak(i8** %[[B]])
// ARC-NEXT: %[[V5:.*]] = tail call i8* @llvm.objc.loadWeakRetained(i8** %[[B]])
// ARC-NEXT: tail call void @llvm.objc.moveWeak(i8** %[[B]], i8** %[[C]])
// ARC-NEXT: tail call void @llvm.objc.release(i8* %[[A]])
// ARC-NEXT: %[[V6:.*]] = tail call i8* @llvm.objc.retain(i8* %[[A]])
// ARC-NEXT: %[[V7:.*]] = tail call i8* @llvm.objc.retainAutorelease(i8* %[[A]])
// ARC-NEXT: %[[V8:.*]] = tail call i8* @llvm.objc.retainAutoreleaseReturnValue(i8* %[[A]])
// ARC-NEXT: %[[V9:.*]] = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[A]])
// ARC-NEXT: %[[V10:.*]] = tail call i8* @llvm.objc.retainBlock(i8* %[[A]])
// ARC-NEXT: tail call void @llvm.objc.storeStrong(i8** %[[B]], i8* %[[A]])
// ARC-NEXT: %[[V11:.*]] = tail call i8* @llvm.objc.storeWeak(i8** %[[B]], i8* %[[A]])
// ARC-NEXT: tail call void (...) @llvm.objc.clang.arc.use(i8* %[[A]])
// ARC-NEXT: %[[V12:.*]] = tail call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %[[A]])
// ARC-NEXT: %[[V13:.*]] = tail call i8* @llvm.objc.retainedObject(i8* %[[A]])
// ARC-NEXT: %[[V14:.*]] = tail call i8* @llvm.objc.unretainedObject(i8* %[[A]])
// ARC-NEXT: %[[V15:.*]] = tail call i8* @llvm.objc.unretainedPointer(i8* %[[A]])
// ARC-NEXT: %[[V16:.*]] = tail call i8* @objc_retain.autorelease(i8* %[[A]])
// ARC-NEXT: %[[V17:.*]] = tail call i32 @objc_sync.enter(i8* %[[A]])
// ARC-NEXT: %[[V18:.*]] = tail call i32 @objc_sync.exit(i8* %[[A]])
// ARC-NEXT: tail call void @llvm.objc.arc.annotation.topdown.bbstart(i8** %[[B]], i8** %[[C]])
// ARC-NEXT: tail call void @llvm.objc.arc.annotation.topdown.bbend(i8** %[[B]], i8** %[[C]])
// ARC-NEXT: tail call void @llvm.objc.arc.annotation.bottomup.bbstart(i8** %[[B]], i8** %[[C]])
// ARC-NEXT: tail call void @llvm.objc.arc.annotation.bottomup.bbend(i8** %[[B]], i8** %[[C]])
// ARC-NEXT: ret void

// MRR: define void @testRuntimeCalls(i8* %[[A:.*]], i8** %[[B:.*]], i8** %[[C:.*]]) {
// MRR: %[[V0:.*]] = tail call i8* @objc_autorelease(i8* %[[A]])
// MRR-NEXT: tail call void @objc_autoreleasePoolPop(i8* %[[A]])
// MRR-NEXT: %[[V1:.*]] = tail call i8* @objc_autoreleasePoolPush()
// MRR-NEXT: %[[V2:.*]] = tail call i8* @objc_autoreleaseReturnValue(i8* %[[A]])
// MRR-NEXT: tail call void @objc_copyWeak(i8** %[[B]], i8** %[[C]])
// MRR-NEXT: tail call void @objc_destroyWeak(i8** %[[B]])
// MRR-NEXT: %[[V3:.*]] = tail call i8* @objc_initWeak(i8** %[[B]], i8* %[[A]])
// MRR-NEXT: %[[V4:.*]] = tail call i8* @objc_loadWeak(i8** %[[B]])
// MRR-NEXT: %[[V5:.*]] = tail call i8* @objc_loadWeakRetained(i8** %[[B]])
// MRR-NEXT: tail call void @objc_moveWeak(i8** %[[B]], i8** %[[C]])
// MRR-NEXT: tail call void @objc_release(i8* %[[A]])
// MRR-NEXT: %[[V6:.*]] = tail call i8* @objc_retain(i8* %[[A]])
// MRR-NEXT: %[[V7:.*]] = tail call i8* @objc_retainAutorelease(i8* %[[A]])
// MRR-NEXT: %[[V8:.*]] = tail call i8* @objc_retainAutoreleaseReturnValue(i8* %[[A]])
// MRR-NEXT: %[[V9:.*]] = tail call i8* @objc_retainAutoreleasedReturnValue(i8* %[[A]])
// MRR-NEXT: %[[V10:.*]] = tail call i8* @objc_retainBlock(i8* %[[A]])
// MRR-NEXT: tail call void @objc_storeStrong(i8** %[[B]], i8* %[[A]])
// MRR-NEXT: %[[V11:.*]] = tail call i8* @objc_storeWeak(i8** %[[B]], i8* %[[A]])
// MRR-NEXT: tail call void (...) @llvm.objc.clang.arc.use(i8* %[[A]])
// MRR-NEXT: %[[V12:.*]] = tail call i8* @objc_unsafeClaimAutoreleasedReturnValue(i8* %[[A]])
// MRR-NEXT: %[[V13:.*]] = tail call i8* @objc_retainedObject(i8* %[[A]])
// MRR-NEXT: %[[V14:.*]] = tail call i8* @objc_unretainedObject(i8* %[[A]])
// MRR-NEXT: %[[V15:.*]] = tail call i8* @objc_unretainedPointer(i8* %[[A]])
// MRR-NEXT: %[[V16:.*]] = tail call i8* @objc_retain.autorelease(i8* %[[A]])
// MRR-NEXT: %[[V17:.*]] = tail call i32 @objc_sync.enter(i8* %[[A]])
// MRR-NEXT: %[[V18:.*]] = tail call i32 @objc_sync.exit(i8* %[[A]])
// MRR-NEXT: tail call void @objc_arc_annotation_topdown_bbstart(i8** %[[B]], i8** %[[C]])
// MRR-NEXT: tail call void @objc_arc_annotation_topdown_bbend(i8** %[[B]], i8** %[[C]])
// MRR-NEXT: tail call void @objc_arc_annotation_bottomup_bbstart(i8** %[[B]], i8** %[[C]])
// MRR-NEXT: tail call void @objc_arc_annotation_bottomup_bbend(i8** %[[B]], i8** %[[C]])
// MRR-NEXT: ret void
