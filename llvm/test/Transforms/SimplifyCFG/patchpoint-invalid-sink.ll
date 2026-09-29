; RUN: opt -passes='simplifycfg<sink-common-insts>' -S %s | FileCheck %s

declare void @personalityFn()

define void @test(i1 %c) personality ptr @personalityFn {
; CHECK-LABEL: define void @test
; CHECK-LABEL: entry:
; CHECK-NEXT:    br i1 %c, label %taken, label %untaken
; CHECK-LABEL: taken:
; CHECK-NEXT:    invoke void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 1, i32 0, ptr null, i32 0)
; CHECK-LABEL: untaken:
; CHECK-NEXT:    invoke void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 2, i32 0, ptr null, i32 0)
; CHECK-LABEL: end:
; CHECK-NEXT:    ret void
entry:
  br i1 %c, label %taken, label %untaken

taken:
  invoke void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 1, i32 0, ptr null, i32 0)
          to label %end unwind label %unwind

untaken:
  invoke void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 2, i32 0, ptr null, i32 0)
          to label %end unwind label %unwind

end:
  ret void

unwind:
  %0 = landingpad { ptr, i32 }
          cleanup
  br label %end
}

declare void @llvm.experimental.patchpoint.void(i64 immarg, i32 immarg, ptr, i32 immarg, ...)
