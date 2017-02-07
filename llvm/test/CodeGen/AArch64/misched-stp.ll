; REQUIRES: asserts
; RUN: llc < %s -mtriple=aarch64 -mcpu=cyclone -mattr=+use-aa -enable-misched -verify-misched -debug-only=misched -o - 2>&1 > /dev/null | FileCheck %s

; Tests to check that the scheduler dependencies derived from alias analysis are
; correct when we have loads that have been split up so that they can later be
; merged into STP.

; CHECK: ********** MI Scheduling **********
; CHECK: test_splat:BB#0 entry
; CHECK: SU({{[0-9]+}}):   STRWui %vreg{{[0-9]+}}, %vreg{{[0-9]+}}, 3; mem:ST4[%3+8]
; CHECK: Successors:
; CHECK-NEXT: ord  [[SU1:SU\([0-9]+\)]]
; CHECK: SU({{[0-9]+}}):   STRWui %vreg{{[0-9]+}}, %vreg{{[0-9]+}}, 2; mem:ST4[%3+4]
; CHECK: Successors:
; CHECK-NEXT: ord  [[SU2:SU\([0-9]+\)]]
; CHECK: [[SU1]]:   STRWui %vreg{{[0-9]+}}, %vreg{{[0-9]+}}, 3; mem:ST4[%2]
; CHECK: [[SU2]]:   STRWui %vreg{{[0-9]+}}, %vreg{{[0-9]+}}, 2; mem:ST4[%1]
define void @test_splat(i32 %x, i32 %y, i32* %p) {
entry:
  %val = load i32, i32* %p, align 4
  %0 = getelementptr inbounds i32, i32* %p, i64 1
  %1 = getelementptr inbounds i32, i32* %p, i64 2
  %2 = getelementptr inbounds i32, i32* %p, i64 3
  %vec0 = insertelement <4 x i32> undef, i32 %val, i32 0
  %vec1 = insertelement <4 x i32> %vec0, i32 %val, i32 1
  %vec2 = insertelement <4 x i32> %vec1, i32 %val, i32 2
  %vec3 = insertelement <4 x i32> %vec2, i32 %val, i32 3
  %3 = bitcast i32* %0 to <4 x i32>*
  store <4 x i32> %vec3, <4 x i32>* %3, align 4
  store i32 %x, i32* %2, align 4
  store i32 %y, i32* %1, align 4
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1)
%struct.tree_common = type { i8*, i8*, i32 }

; CHECK: ********** MI Scheduling **********
; CHECK: test_zero:BB#0 entry
; CHECK: SU({{[0-9]+}}):   STRXui %XZR, %vreg{{[0-9]+}}, 2; mem:ST8[%0+16]
; CHECK: Successors:
; CHECK-NEXT: ord  [[SU3:SU\([0-9]+\)]]
; CHECK: SU({{[0-9]+}}):   STRXui %XZR, %vreg{{[0-9]+}}, 1; mem:ST8[%0+8]
; CHECK: Successors:
; CHECK-NEXT: ord  [[SU4:SU\([0-9]+\)]]
; CHECK: [[SU3]]:   STRWui %vreg{{[0-9]+}}, %vreg{{[0-9]+}}, 4; mem:ST4[%code1]
; CHECK: [[SU4]]:   STRXui %vreg{{[0-9]+}}, %vreg{{[0-9]+}}, 1; mem:ST8[%type2]
define void @test_zero(%struct.tree_common* %t, i32 %code, i8* %type) {
entry:
  %0 = bitcast %struct.tree_common* %t to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 24, i32 8, i1 false)
  %code1 = getelementptr inbounds %struct.tree_common, %struct.tree_common* %t, i64 0, i32 2
  store i32 %code, i32* %code1, align 8
  %type2 = getelementptr inbounds %struct.tree_common, %struct.tree_common* %t, i64 0, i32 1
  store i8* %type, i8** %type2, align 8
  ret void
}
