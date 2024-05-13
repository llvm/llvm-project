; Test that calls to ARC runtime functions are converted to intrinsic calls if
; the bitcode has the arm64 retainAutoreleasedReturnValueMarker metadata.

; upgrade-arc-runtime-calls.bc and upgrade-mrr-runtime-calls.bc are identical
; except that the former has the arm64 retainAutoreleasedReturnValueMarker
; metadata. upgrade-arc-runtime-calls-new.bc has the new module flag format of
; marker, it should not be upgraded.

; RUN: llvm-dis < %S/upgrade-arc-runtime-calls.bc | FileCheck -check-prefixes=ARC %s
; RUN: llvm-dis < %S/upgrade-mrr-runtime-calls.bc | FileCheck -check-prefixes=NOUPGRADE %s
; RUN: llvm-dis < %S/upgrade-arc-runtime-calls-new.bc | FileCheck -check-prefixes=NOUPGRADE %s

define void @testRuntimeCalls(i8* %a, i8** %b, i8** %c, i32* %d, i32** %e) personality i32 (...)* @__gxx_personality_v0 {
entry:
  %v0 = tail call i8* @objc_autorelease(i8* %a) #0
  tail call void @objc_autoreleasePoolPop(i8* %a) #0
  %v1 = tail call i8* @objc_autoreleasePoolPush() #0
  %v2 = tail call i8* @objc_autoreleaseReturnValue(i8* %a) #0
  tail call void @objc_copyWeak(i8** %b, i8** %c) #0
  tail call void @objc_destroyWeak(i8** %b) #0
  %v3 = tail call i32* @objc_initWeak(i32** %e, i32* %d) #0
  %v4 = tail call i8* @objc_loadWeak(i8** %b) #0
  %v5 = tail call i8* @objc_loadWeakRetained(i8** %b) #0
  tail call void @objc_moveWeak(i8** %b, i8** %c) #0
  tail call void @objc_release(i8* %a) #0
  %v6 = tail call i8* @objc_retain(i8* %a) #0
  %v7 = tail call i8* @objc_retainAutorelease(i8* %a) #0
  %v8 = tail call i8* @objc_retainAutoreleaseReturnValue(i8* %a) #0
  %v9 = tail call i8* @objc_retainAutoreleasedReturnValue(i8* %a) #0
  %v10 = tail call i8* @objc_retainBlock(i8* %a) #0
  tail call void @objc_storeStrong(i8** %b, i8* %a) #0
  %v11 = tail call i8* @objc_storeWeak(i8** %b, i8* %a) #0
  tail call void (...) @clang.arc.use(i8* %a) #0
  %v12 = tail call i8* @objc_unsafeClaimAutoreleasedReturnValue(i8* %a) #0
  %v13 = tail call i8* @objc_retainedObject(i8* %a) #0
  %v14 = tail call i8* @objc_unretainedObject(i8* %a) #0
  %v15 = tail call i8* @objc_unretainedPointer(i8* %a) #0
  %v16 = tail call i8* @objc_retain.autorelease(i8* %a) #0
  %v17 = tail call i32 @objc_sync.enter(i8* %a) #0
  %v18 = tail call i32 @objc_sync.exit(i8* %a) #0
  tail call void @objc_arc_annotation_topdown_bbstart(i8** %b, i8** %c) #0
  tail call void @objc_arc_annotation_topdown_bbend(i8** %b, i8** %c) #0
  tail call void @objc_arc_annotation_bottomup_bbstart(i8** %b, i8** %c) #0
  tail call void @objc_arc_annotation_bottomup_bbend(i8** %b, i8** %c) #0
  invoke void @objc_autoreleasePoolPop(i8* %a)
          to label %normalBlock unwind label %unwindBlock
normalBlock:
  ret void
unwindBlock:
  %ll = landingpad { i8*, i32 }
          cleanup
  ret void
}

// Check that auto-upgrader converts function calls to intrinsic calls. Note that
// the auto-upgrader doesn't touch invoke instructions.

// ARC: define void @testRuntimeCalls(ptr %[[A:.*]], ptr %[[B:.*]], ptr %[[C:.*]], ptr %[[D:.*]], ptr %[[E:.*]]) personality
// ARC: %[[V0:.*]] = tail call ptr @llvm.objc.autorelease(ptr %[[A]])
// ARC-NEXT: tail call void @llvm.objc.autoreleasePoolPop(ptr %[[A]])
// ARC-NEXT: %[[V1:.*]] = tail call ptr @llvm.objc.autoreleasePoolPush()
// ARC-NEXT: %[[V2:.*]] = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %[[A]])
// ARC-NEXT: tail call void @llvm.objc.copyWeak(ptr %[[B]], ptr %[[C]])
// ARC-NEXT: tail call void @llvm.objc.destroyWeak(ptr %[[B]])
// ARC-NEXT: %[[V102:.*]] = tail call ptr @llvm.objc.initWeak(ptr %[[E]], ptr %[[D]])
// ARC-NEXT: %[[V4:.*]] = tail call ptr @llvm.objc.loadWeak(ptr %[[B]])
// ARC-NEXT: %[[V5:.*]] = tail call ptr @llvm.objc.loadWeakRetained(ptr %[[B]])
// ARC-NEXT: tail call void @llvm.objc.moveWeak(ptr %[[B]], ptr %[[C]])
// ARC-NEXT: tail call void @llvm.objc.release(ptr %[[A]])
// ARC-NEXT: %[[V6:.*]] = tail call ptr @llvm.objc.retain(ptr %[[A]])
// ARC-NEXT: %[[V7:.*]] = tail call ptr @llvm.objc.retainAutorelease(ptr %[[A]])
// ARC-NEXT: %[[V8:.*]] = tail call ptr @llvm.objc.retainAutoreleaseReturnValue(ptr %[[A]])
// ARC-NEXT: %[[V9:.*]] = tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %[[A]])
// ARC-NEXT: %[[V10:.*]] = tail call ptr @llvm.objc.retainBlock(ptr %[[A]])
// ARC-NEXT: tail call void @llvm.objc.storeStrong(ptr %[[B]], ptr %[[A]])
// ARC-NEXT: %[[V11:.*]] = tail call ptr @llvm.objc.storeWeak(ptr %[[B]], ptr %[[A]])
// ARC-NEXT: tail call void (...) @llvm.objc.clang.arc.use(ptr %[[A]])
// ARC-NEXT: %[[V12:.*]] = tail call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %[[A]])
// ARC-NEXT: %[[V13:.*]] = tail call ptr @llvm.objc.retainedObject(ptr %[[A]])
// ARC-NEXT: %[[V14:.*]] = tail call ptr @llvm.objc.unretainedObject(ptr %[[A]])
// ARC-NEXT: %[[V15:.*]] = tail call ptr @llvm.objc.unretainedPointer(ptr %[[A]])
// ARC-NEXT: %[[V16:.*]] = tail call ptr @objc_retain.autorelease(ptr %[[A]])
// ARC-NEXT: %[[V17:.*]] = tail call i32 @objc_sync.enter(ptr %[[A]])
// ARC-NEXT: %[[V18:.*]] = tail call i32 @objc_sync.exit(ptr %[[A]])
// ARC-NEXT: tail call void @llvm.objc.arc.annotation.topdown.bbstart(ptr %[[B]], ptr %[[C]])
// ARC-NEXT: tail call void @llvm.objc.arc.annotation.topdown.bbend(ptr %[[B]], ptr %[[C]])
// ARC-NEXT: tail call void @llvm.objc.arc.annotation.bottomup.bbstart(ptr %[[B]], ptr %[[C]])
// ARC-NEXT: tail call void @llvm.objc.arc.annotation.bottomup.bbend(ptr %[[B]], ptr %[[C]])
// ARC-NEXT: invoke void @objc_autoreleasePoolPop(ptr %[[A]])

// NOUPGRADE: define void @testRuntimeCalls(ptr %[[A:.*]], ptr %[[B:.*]], ptr %[[C:.*]], ptr %[[D:.*]], ptr %[[E:.*]]) personality
// NOUPGRADE: %[[V0:.*]] = tail call ptr @objc_autorelease(ptr %[[A]])
// NOUPGRADE-NEXT: tail call void @objc_autoreleasePoolPop(ptr %[[A]])
// NOUPGRADE-NEXT: %[[V1:.*]] = tail call ptr @objc_autoreleasePoolPush()
// NOUPGRADE-NEXT: %[[V2:.*]] = tail call ptr @objc_autoreleaseReturnValue(ptr %[[A]])
// NOUPGRADE-NEXT: tail call void @objc_copyWeak(ptr %[[B]], ptr %[[C]])
// NOUPGRADE-NEXT: tail call void @objc_destroyWeak(ptr %[[B]])
// NOUPGRADE-NEXT: %[[V3:.*]] = tail call ptr @objc_initWeak(ptr %[[E]], ptr %[[D]])
// NOUPGRADE-NEXT: %[[V4:.*]] = tail call ptr @objc_loadWeak(ptr %[[B]])
// NOUPGRADE-NEXT: %[[V5:.*]] = tail call ptr @objc_loadWeakRetained(ptr %[[B]])
// NOUPGRADE-NEXT: tail call void @objc_moveWeak(ptr %[[B]], ptr %[[C]])
// NOUPGRADE-NEXT: tail call void @objc_release(ptr %[[A]])
// NOUPGRADE-NEXT: %[[V6:.*]] = tail call ptr @objc_retain(ptr %[[A]])
// NOUPGRADE-NEXT: %[[V7:.*]] = tail call ptr @objc_retainAutorelease(ptr %[[A]])
// NOUPGRADE-NEXT: %[[V8:.*]] = tail call ptr @objc_retainAutoreleaseReturnValue(ptr %[[A]])
// NOUPGRADE-NEXT: %[[V9:.*]] = tail call ptr @objc_retainAutoreleasedReturnValue(ptr %[[A]])
// NOUPGRADE-NEXT: %[[V10:.*]] = tail call ptr @objc_retainBlock(ptr %[[A]])
// NOUPGRADE-NEXT: tail call void @objc_storeStrong(ptr %[[B]], ptr %[[A]])
// NOUPGRADE-NEXT: %[[V11:.*]] = tail call ptr @objc_storeWeak(ptr %[[B]], ptr %[[A]])
// NOUPGRADE-NEXT: tail call void (...) @llvm.objc.clang.arc.use(ptr %[[A]])
// NOUPGRADE-NEXT: %[[V12:.*]] = tail call ptr @objc_unsafeClaimAutoreleasedReturnValue(ptr %[[A]])
// NOUPGRADE-NEXT: %[[V13:.*]] = tail call ptr @objc_retainedObject(ptr %[[A]])
// NOUPGRADE-NEXT: %[[V14:.*]] = tail call ptr @objc_unretainedObject(ptr %[[A]])
// NOUPGRADE-NEXT: %[[V15:.*]] = tail call ptr @objc_unretainedPointer(ptr %[[A]])
// NOUPGRADE-NEXT: %[[V16:.*]] = tail call ptr @objc_retain.autorelease(ptr %[[A]])
// NOUPGRADE-NEXT: %[[V17:.*]] = tail call i32 @objc_sync.enter(ptr %[[A]])
// NOUPGRADE-NEXT: %[[V18:.*]] = tail call i32 @objc_sync.exit(ptr %[[A]])
// NOUPGRADE-NEXT: tail call void @objc_arc_annotation_topdown_bbstart(ptr %[[B]], ptr %[[C]])
// NOUPGRADE-NEXT: tail call void @objc_arc_annotation_topdown_bbend(ptr %[[B]], ptr %[[C]])
// NOUPGRADE-NEXT: tail call void @objc_arc_annotation_bottomup_bbstart(ptr %[[B]], ptr %[[C]])
// NOUPGRADE-NEXT: tail call void @objc_arc_annotation_bottomup_bbend(ptr %[[B]], ptr %[[C]])
// NOUPGRADE-NEXT: invoke void @objc_autoreleasePoolPop(ptr %[[A]])
