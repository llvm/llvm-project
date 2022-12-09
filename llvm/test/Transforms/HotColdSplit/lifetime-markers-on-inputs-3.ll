; RUN: opt -S -hotcoldsplit -hotcoldsplit-threshold=0 < %s 2>&1 | FileCheck %s

%type1 = type opaque
%type2 = type opaque

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)

declare void @llvm.lifetime.end.p0(i64, ptr nocapture)

declare void @use(ptr, ptr)

declare void @use2(ptr, ptr) cold

; CHECK-LABEL: define {{.*}}@foo(
define void @foo() {
entry:
  %local1 = alloca ptr
  %local2 = alloca ptr
  br i1 undef, label %normalPath, label %outlinedPath

normalPath:
  call void @use(ptr %local1, ptr %local2)
  ret void

; CHECK-LABEL: codeRepl:
; CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 -1, ptr %local1)
; CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 -1, ptr %local2)
; CHECK-NEXT: call void @foo.cold.1(ptr %local1, ptr %local2

outlinedPath:
  call void @llvm.lifetime.start.p0(i64 1, ptr %local1)
  call void @llvm.lifetime.start.p0(i64 1, ptr %local2)
  call void @use2(ptr %local1, ptr %local2)
  call void @llvm.lifetime.end.p0(i64 1, ptr %local1)
  call void @llvm.lifetime.end.p0(i64 1, ptr %local2)
  br label %outlinedPathExit

outlinedPathExit:
  ret void
}

; CHECK-LABEL: define {{.*}}@foo.cold.1(
; CHECK-NOT: @llvm.lifetime
