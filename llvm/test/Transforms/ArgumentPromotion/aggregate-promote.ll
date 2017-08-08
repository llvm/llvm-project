; RUN: opt < %s -argpromotion -S | FileCheck %s
; RUN: opt < %s -passes=argpromotion -S | FileCheck %s

%T = type { i32, i32, i32, i32 }
@G = constant %T { i32 0, i32 0, i32 17, i32 25 }

define internal i32 @test(%T* %p) {
; CHECK-LABEL: define internal i32 @test(
; CHECK: i32 %{{.*}}, i32 %{{.*}})
entry:
  %a.gep = getelementptr %T, %T* %p, i64 0, i32 3
  %b.gep = getelementptr %T, %T* %p, i64 0, i32 2
  %a = load i32, i32* %a.gep
  %b = load i32, i32* %b.gep
; CHECK-NOT: load
  %v = add i32 %a, %b
  ret i32 %v
; CHECK: ret i32
}

define i32 @caller() {
; CHECK-LABEL: define i32 @caller(
entry:
  %v = call i32 @test(%T* @G)
; CHECK: %[[B_GEP:.*]] = getelementptr %T, %T* @G, i64 0, i32 2
; CHECK: %[[B:.*]] = load i32, i32* %[[B_GEP]]
; CHECK: %[[A_GEP:.*]] = getelementptr %T, %T* @G, i64 0, i32 3
; CHECK: %[[A:.*]] = load i32, i32* %[[A_GEP]]
; CHECK: call i32 @test(i32 %[[B]], i32 %[[A]])
  ret i32 %v
}
