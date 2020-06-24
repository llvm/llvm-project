; RUN: llc -march=hexagon -mcpu=hexagonv65 < %s | FileCheck %s
; REQUIRES: hexagon

; C file was:
; struct S { int a[3];};
; void foo(int x, struct S arg1, struct S arg2);
; void bar() {
;  struct S s;
;  s.a[0] = 9;
;  foo(42, s, s);
; }

; Test that while passing a 12-byte struct on the stack, the
; struct is aligned to 4 bytes since its largest member is of type int.
; Previously, the struct was being aligned to 8 bytes

; CHECK: memw(r{{[0-9]+}}+#12) = #9

; Check that the flag hexagon-disable-args-min-alignment works and the struct
; is aligned to 8 bytes.
; RUN: llc -march=hexagon -mcpu=hexagonv65 -hexagon-disable-args-min-alignment < %s | FileCheck -check-prefix=HEXAGON_LEGACY %s

; HEXAGON_LEGACY: memw(r{{[0-9]+}}+#16) = #9

%struct.S = type { [3 x i32] }

; Function Attrs: nounwind
define dso_local void @bar() local_unnamed_addr #0 {
entry:
  %s = alloca %struct.S, align 4
  %0 = bitcast %struct.S* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %0) #3
  %arrayidx = getelementptr inbounds %struct.S, %struct.S* %s, i32 0, i32 0, i32 0
  store i32 9, i32* %arrayidx, align 4
  tail call void @foo(i32 42, %struct.S* nonnull byval(%struct.S) align 4 %s, %struct.S* nonnull byval(%struct.S) align 4 %s) #3
  call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %0) #3
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

declare dso_local void @foo(i32, %struct.S* byval(%struct.S) align 4, %struct.S* byval(%struct.S) align 4) local_unnamed_addr #2

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1
