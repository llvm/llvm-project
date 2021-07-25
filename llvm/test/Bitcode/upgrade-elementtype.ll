; RUN: opt -S < %s.bc | FileCheck %s

%struct.s = type { i32, i32 }

define void @test(%struct.s* %arg) {
; CHECK-LABEL: define void @test
; CHECK: %x = call %struct.s* @llvm.preserve.array.access.index.p0s_struct.ss.p0s_struct.ss(%struct.s* elementtype(%struct.s) %arg, i32 0, i32 2)
; CHECK: %1 = call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.ss(%struct.s* elementtype(%struct.s) %x, i32 1, i32 1)
  %x = call %struct.s* @llvm.preserve.array.access.index.p0s_struct.ss.p0s_struct.ss(%struct.s* %arg, i32 0, i32 2)
  call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.ss(%struct.s* %x, i32 1, i32 1)
  ret void
}

declare %struct.s* @llvm.preserve.array.access.index.p0s_struct.ss.p0s_struct.ss(%struct.s*, i32, i32)
declare i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.ss(%struct.s*, i32, i32)
