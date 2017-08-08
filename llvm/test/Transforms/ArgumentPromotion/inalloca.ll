; RUN: opt %s -argpromotion -sroa -S | FileCheck %s
; RUN: opt %s -passes='argpromotion,function(sroa)' -S | FileCheck %s

target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

%struct.ss = type { i32, i32 }

; Argpromote + sroa should change this to passing the two integers by value.
define internal i32 @f(%struct.ss* inalloca  %s) {
entry:
  %f0 = getelementptr %struct.ss, %struct.ss* %s, i32 0, i32 0
  %f1 = getelementptr %struct.ss, %struct.ss* %s, i32 0, i32 1
  %a = load i32, i32* %f0, align 4
  %b = load i32, i32* %f1, align 4
  %r = add i32 %a, %b
  ret i32 %r
}
; CHECK-LABEL: define internal i32 @f
; CHECK-NOT: load
; CHECK: ret

define i32 @main() {
entry:
  %S = alloca inalloca %struct.ss
  %f0 = getelementptr %struct.ss, %struct.ss* %S, i32 0, i32 0
  %f1 = getelementptr %struct.ss, %struct.ss* %S, i32 0, i32 1
  store i32 1, i32* %f0, align 4
  store i32 2, i32* %f1, align 4
  %r = call i32 @f(%struct.ss* inalloca %S)
  ret i32 %r
}
; CHECK-LABEL: define i32 @main
; CHECK-NOT: load
; CHECK: ret

; Argpromote can't promote %a because of the icmp use.
define internal i1 @g(%struct.ss* %a, %struct.ss* inalloca %b) nounwind  {
; CHECK: define internal i1 @g(%struct.ss* %a, %struct.ss* inalloca %b)
entry:
  %c = icmp eq %struct.ss* %a, %b
  ret i1 %c
}

define i32 @test() {
entry:
  %S = alloca inalloca %struct.ss
  %c = call i1 @g(%struct.ss* %S, %struct.ss* inalloca %S)
; CHECK: call i1 @g(%struct.ss* %S, %struct.ss* inalloca %S)
  ret i32 0
}
