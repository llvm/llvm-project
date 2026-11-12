; RUN: opt < %s -disable-output -passes='print<assumptions>' 2>&1 | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"

declare void @llvm.assume(i1)

define void @test1(i32 %a) {
; CHECK-LABEL: Cached assumptions for function: test1
; CHECK-NEXT: icmp ne i32 %{{.*}}, 0
; CHECK-NEXT: icmp slt i32 %{{.*}}, 0
; CHECK-NEXT: icmp sgt i32 %{{.*}}, 0

entry:
  %cond1 = icmp ne i32 %a, 0
  call void @llvm.assume(i1 %cond1)
  %cond2 = icmp slt i32 %a, 0
  call void @llvm.assume(i1 %cond2)
  %cond3 = icmp sgt i32 %a, 0
  call void @llvm.assume(i1 %cond3)

  ret void
}

@G = external global i32
define void @test2() {
; CHECK-LABEL: Cached assumptions for function: test2
; CHECK-NEXT: icmp ne ptr @G, null

entry:
  %cond1 = icmp ne ptr @G, null
  call void @llvm.assume(i1 %cond1)
  ret void
}

define void @test_bundles(ptr %A, i32 %x) {
; CHECK-LABEL: Cached assumptions for function: test_bundles
; CHECK-NEXT: [ "dereferenceable"(ptr %A, i64 1024) ]
; CHECK-NEXT: [ "align"(ptr %A, i64 8), "nonnull"(ptr %A) ]
; CHECK-NEXT: icmp ne i32 %{{.*}}, 0
; CHECK-NEXT: [ "separate_storage"(ptr %A, ptr %A) ]

entry:
  call void @llvm.assume(i1 true) [ "dereferenceable"(ptr %A, i64 1024) ]
  call void @llvm.assume(i1 true) [ "align"(ptr %A, i64 8), "nonnull"(ptr %A) ]
  %cond = icmp ne i32 %x, 0
  call void @llvm.assume(i1 %cond)
  call void @llvm.assume(i1 true) [ "separate_storage"(ptr %A, ptr %A) ]
  ret void
}
