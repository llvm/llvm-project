; RUN: opt -S -hotcoldsplit -hotcoldsplit-threshold=0 < %s 2>&1 | FileCheck %s

%type1 = type opaque
%type2 = type opaque

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

declare void @use(%type1**, %type2**)

declare void @use2(%type1**, %type2**) cold

; CHECK-LABEL: define {{.*}}@foo(
define void @foo() {
entry:
  %local1 = alloca %type1*
  %local2 = alloca %type2*
  %local1_cast = bitcast %type1** %local1 to i8*
  %local2_cast = bitcast %type2** %local2 to i8*
  br i1 undef, label %normalPath, label %outlinedPath

normalPath:
  call void @use(%type1** %local1, %type2** %local2)
  ret void

; CHECK-LABEL: codeRepl:
; CHECK: [[local1_cast:%.*]] = bitcast %type1** %local1 to i8*
; CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 -1, i8* [[local1_cast]])
; CHECK-NEXT: [[local2_cast:%.*]] = bitcast %type2** %local2 to i8*
; CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 -1, i8* [[local2_cast]])
; CHECK-NEXT: call void @foo.cold.1(i8* %local1_cast, i8* %local2_cast

outlinedPath:
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %local1_cast)
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %local2_cast)
  call void @use2(%type1** %local1, %type2** %local2)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %local1_cast)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %local2_cast)
  br label %outlinedPathExit

outlinedPathExit:
  ret void
}

; CHECK-LABEL: define {{.*}}@foo.cold.1(
; CHECK-NOT: @llvm.lifetime
