; RUN: opt -S -hotcoldsplit -hotcoldsplit-threshold=0 < %s 2>&1 | FileCheck %s

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)

declare void @llvm.lifetime.end.p0(i64, ptr nocapture)

declare void @use(ptr)

declare void @cold_use2(ptr, ptr) cold

; CHECK-LABEL: define {{.*}}@foo(
define void @foo() {
entry:
  %local1 = alloca i256
  %local2 = alloca i256
  br i1 undef, label %normalPath, label %outlinedPath

normalPath:
  ; These two uses of stack slots are non-overlapping. Based on this alone,
  ; the stack slots could be merged.
  call void @llvm.lifetime.start.p0(i64 1, ptr %local1)
  call void @use(ptr %local1)
  call void @llvm.lifetime.end.p0(i64 1, ptr %local1)
  call void @llvm.lifetime.start.p0(i64 1, ptr %local2)
  call void @use(ptr %local2)
  call void @llvm.lifetime.end.p0(i64 1, ptr %local2)
  ret void

; CHECK-LABEL: codeRepl:
; CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 -1, ptr %local1)
; CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 -1, ptr %local2)
; CHECK-NEXT: call i1 @foo.cold.1(ptr %local1, ptr %local2)
; CHECK-NEXT: br i1

outlinedPath:
  ; These two uses of stack slots are overlapping. This should prevent
  ; merging of stack slots. CodeExtractor must replicate the effects of
  ; these markers in the caller to inhibit stack coloring.
  %gep1 = getelementptr inbounds i8, ptr %local1, i64 1
  call void @llvm.lifetime.start.p0(i64 1, ptr %gep1)
  call void @llvm.lifetime.start.p0(i64 1, ptr %local2)
  call void @cold_use2(ptr %local1, ptr %local2)
  call void @llvm.lifetime.end.p0(i64 1, ptr %gep1)
  call void @llvm.lifetime.end.p0(i64 1, ptr %local2)
  br i1 undef, label %outlinedPath2, label %outlinedPathExit

outlinedPath2:
  ; These extra lifetime markers are used to test that we emit only one
  ; pair of guard markers in the caller per memory object.
  call void @llvm.lifetime.start.p0(i64 1, ptr %local2)
  call void @use(ptr %local2)
  call void @llvm.lifetime.end.p0(i64 1, ptr %local2)
  ret void

outlinedPathExit:
  ret void
}

; CHECK-LABEL: define {{.*}}@foo.cold.1(
; CHECK-NOT: @llvm.lifetime
