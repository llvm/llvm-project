; RUN: opt -passes=instcombine -S < %s | FileCheck %s

define i1 @src(i64 %arg0, ptr %arg1) {
; CHECK: %[[SH:.*]] = shl i32 5, %[[SHIFT:.*]]
; CHECK: %[[TR:.*]] = trunc i32 %[[SH]] to i8
; CHECK: %[[AND:.*]] = and i8 %{{.*}}, %[[TR]]
; CHECK: icmp ne i8 %[[AND]], 0

  %v0 = load i8, ptr %arg1, align 1
  %v1 = trunc nuw nsw i64 %arg0 to i32
  %v2 = shl nuw nsw i32 1, %v1
  %v3 = trunc nuw nsw i32 %v2 to i8
  %v4 = and i8 %v0, %v3
  %v5 = shl nuw nsw i32 4, %v1
  %v6 = trunc nuw nsw i32 %v5 to i8
  %v7 = and i8 %v0, %v6
  %v8 = icmp ne i8 %v4, 0
  %v9 = icmp ne i8 %v7, 0
  %v10 = select i1 %v8, i1 true, i1 %v9
  ret i1 %v10
}
