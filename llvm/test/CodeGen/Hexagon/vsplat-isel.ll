; RUN: llc -mtriple=hexagon -O0 < %s | FileCheck %s
; CHECK: vsplatb

declare i32 @llvm.hexagon.S2.vsplatrb(i32) #0

define i32 @foo(i8 %x) {
  %p0 = zext i8 %x to i32
  %p1 = tail call i32 @llvm.hexagon.S2.vsplatrb(i32 %p0)
  ret i32 %p1
}
