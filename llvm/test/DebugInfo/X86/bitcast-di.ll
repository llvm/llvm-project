; RUN: opt -mtriple=x86_64-unknown-linux-gnu -S -debugify -codegenprepare < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

@x = external global [1 x [2 x <4 x float>]]

; Is DI maitained when a GEP with all zero indices gets converted to bitcast?
define void @test2() {
; CHECK-LABEL: @test2
load.i145:
; CHECK: bitcast ptr @x to ptr, !dbg ![[castLoc:[0-9]+]]
  %x_offset = getelementptr [1 x [2 x <4 x float>]], ptr @x, i32 0, i64 0
  ret void
}

; CHECK: ![[castLoc]] = !DILocation(line: 1
