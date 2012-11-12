; RUN: llc < %s -O0 -fast-isel-abort -march=x86 | FileCheck %s

%struct.s = type {i32, i32, i32}

define i32 @test1() nounwind {
tak:
	%tmp = call i1 @foo()
	br i1 %tmp, label %BB1, label %BB2
BB1:
	ret i32 1
BB2:
	ret i32 0
; CHECK: test1:
; CHECK: calll
; CHECK-NEXT: testb	$1
}
declare zeroext i1 @foo()  nounwind

declare void @foo2(%struct.s* byval)

define void @test2(%struct.s* %d) nounwind {
  call void @foo2(%struct.s* byval %d )
  ret void
; CHECK: test2:
; CHECK: movl	(%eax)
; CHECK: movl {{.*}}, (%esp)
; CHECK: movl	4(%eax)
; CHECK: movl {{.*}}, 4(%esp)
; CHECK: movl	8(%eax)
; CHECK: movl {{.*}}, 8(%esp)
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind

define void @test3(i8* %a) {
  call void @llvm.memset.p0i8.i32(i8* %a, i8 0, i32 100, i32 1, i1 false)
  ret void
; CHECK: test3:
; CHECK:   movl	{{.*}}, (%esp)
; CHECK:   movl	$0, 4(%esp)
; CHECK:   movl	$100, 8(%esp)
; CHECK:   calll {{.*}}memset
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

define void @test4(i8* %a, i8* %b) {
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a, i8* %b, i32 100, i32 1, i1 false)
  ret void
; CHECK: test4:
; CHECK:   movl	{{.*}}, (%esp)
; CHECK:   movl	{{.*}}, 4(%esp)
; CHECK:   movl	$100, 8(%esp)
; CHECK:   calll {{.*}}memcpy
}
