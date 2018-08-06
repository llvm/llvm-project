; RUN: llc < %s -O0 -fast-isel-abort=1 -mtriple=i686-apple-darwin8 2>/dev/null | FileCheck %s
; RUN: llc < %s -O0 -fast-isel-abort=1 -mtriple=i686-apple-darwin8 2>&1 >/dev/null | FileCheck -check-prefix=STDERR -allow-empty %s

%struct.s = type {i32, i32, i32}

define i32 @test1() nounwind {
tak:
	%tmp = call i1 @foo()
	br i1 %tmp, label %BB1, label %BB2
BB1:
	ret i32 1
BB2:
	ret i32 0
; CHECK-LABEL: test1:
; CHECK: calll
; CHECK-NEXT: testb	$1
}
declare zeroext i1 @foo()  nounwind

declare void @foo2(%struct.s* byval)

define void @test2(%struct.s* %d) nounwind {
  call void @foo2(%struct.s* byval %d )
  ret void
; CHECK-LABEL: test2:
; CHECK: movl	(%eax), %ecx
; CHECK: movl	%ecx, (%esp)
; CHECK: movl	4(%eax), %ecx
; CHECK: movl	%ecx, 4(%esp)
; CHECK: movl	8(%eax), %eax
; CHECK: movl	%eax, 8(%esp)
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1) nounwind

define void @test3(i8* %a) {
  call void @llvm.memset.p0i8.i32(i8* %a, i8 0, i32 100, i1 false)
  ret void
; CHECK-LABEL: test3:
; CHECK:   movl	{{.*}}, (%esp)
; CHECK:   movl	$0, 4(%esp)
; CHECK:   movl	$100, 8(%esp)
; CHECK:   calll {{.*}}memset
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind

define void @test4(i8* %a, i8* %b) {
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a, i8* %b, i32 100, i1 false)
  ret void
; CHECK-LABEL: test4:
; CHECK:   movl	{{.*}}, (%esp)
; CHECK:   movl	{{.*}}, 4(%esp)
; CHECK:   movl	$100, 8(%esp)
; CHECK:   calll {{.*}}memcpy
}

; STDERR-NOT: FastISel missed call:   call x86_thiscallcc void @thiscallfun
%struct.S = type { i8 }
define void @test5() {
entry:
  %s = alloca %struct.S, align 1
; CHECK-LABEL: test5:
; CHECK: subl $12, %esp
; CHECK: leal 8(%esp), %ecx
; CHECK: movl $43, (%esp)
; CHECK: calll {{.*}}thiscallfun
; CHECK: addl $8, %esp
  call x86_thiscallcc void @thiscallfun(%struct.S* %s, i32 43)
  ret void
}
declare x86_thiscallcc void @thiscallfun(%struct.S*, i32) #1

; STDERR-NOT: FastISel missed call:   call x86_stdcallcc void @stdcallfun
define void @test6() {
entry:
; CHECK-LABEL: test6:
; CHECK: subl $12, %esp
; CHECK: movl $43, (%esp)
; CHECK: calll {{.*}}stdcallfun
; CHECK: addl $8, %esp
  call x86_stdcallcc void @stdcallfun(i32 43)
  ret void
}
declare x86_stdcallcc void @stdcallfun(i32) #1
