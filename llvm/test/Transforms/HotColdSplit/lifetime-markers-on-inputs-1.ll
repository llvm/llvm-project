; RUN: opt -S -hotcoldsplit -hotcoldsplit-threshold=0 < %s 2>&1 | FileCheck %s

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

declare void @use(i8*)

declare void @cold_use2(i8*, i8*) cold

; CHECK-LABEL: define {{.*}}@foo(
define void @foo() {
entry:
  %local1 = alloca i256
  %local2 = alloca i256
  %local1_cast = bitcast i256* %local1 to i8*
  %local2_cast = bitcast i256* %local2 to i8*
  br i1 undef, label %normalPath, label %outlinedPath

normalPath:
  ; These two uses of stack slots are non-overlapping. Based on this alone,
  ; the stack slots could be merged.
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %local1_cast)
  call void @use(i8* %local1_cast)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %local1_cast)
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %local2_cast)
  call void @use(i8* %local2_cast)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %local2_cast)
  ret void

; CHECK-LABEL: codeRepl:
; CHECK: [[local1_cast:%.*]] = bitcast i256* %local1 to i8*
; CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 -1, i8* [[local1_cast]])
; CHECK-NEXT: [[local2_cast:%.*]] = bitcast i256* %local2 to i8*
; CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 -1, i8* [[local2_cast]])
; CHECK-NEXT: call i1 @foo.cold.1(i8* %local1_cast, i8* %local2_cast)
; CHECK-NEXT: br i1

outlinedPath:
  ; These two uses of stack slots are overlapping. This should prevent
  ; merging of stack slots. CodeExtractor must replicate the effects of
  ; these markers in the caller to inhibit stack coloring.
  %gep1 = getelementptr inbounds i8, i8* %local1_cast, i64 1
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %gep1)
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %local2_cast)
  call void @cold_use2(i8* %local1_cast, i8* %local2_cast)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %gep1)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %local2_cast)
  br i1 undef, label %outlinedPath2, label %outlinedPathExit

outlinedPath2:
  ; These extra lifetime markers are used to test that we emit only one
  ; pair of guard markers in the caller per memory object.
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %local2_cast)
  call void @use(i8* %local2_cast)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %local2_cast)
  ret void

outlinedPathExit:
  ret void
}

; CHECK-LABEL: define {{.*}}@foo.cold.1(
; CHECK-NOT: @llvm.lifetime
