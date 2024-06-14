; RUN: llc < %s -O0 -fast-isel-abort=1 -mtriple=i686-apple-darwin8 2>/dev/null | FileCheck %s
; RUN: llc < %s -O0 -fast-isel-abort=1 -mtriple=i686-apple-darwin8 2>&1 >/dev/null | FileCheck -check-prefix=STDERR -allow-empty %s
; RUN: llc < %s -O0 -fast-isel-abort=1 -mtriple=i686 2>/dev/null | FileCheck %s --check-prefix=ELF

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

declare void @foo2(ptr byval(%struct.s))

define void @test2(ptr %d) nounwind {
  call void @foo2(ptr byval(%struct.s) %d )
  ret void
; CHECK-LABEL: test2:
; CHECK: movl	(%eax), %ecx
; CHECK: movl	%ecx, (%esp)
; CHECK: movl	4(%eax), %ecx
; CHECK: movl	%ecx, 4(%esp)
; CHECK: movl	8(%eax), %eax
; CHECK: movl	%eax, 8(%esp)
}

declare void @llvm.memset.p0.i32(ptr nocapture, i8, i32, i1) nounwind

define void @test3(ptr %a) {
  call void @llvm.memset.p0.i32(ptr %a, i8 0, i32 100, i1 false)
  ret void
; CHECK-LABEL: test3:
; CHECK:   movl	{{.*}}, (%esp)
; CHECK:   movl	$0, 4(%esp)
; CHECK:   movl	$100, 8(%esp)
; CHECK:   calll {{.*}}memset

; ELF-LABEL: test3:
; ELF:         calll memset{{$}}
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind

define void @test4(ptr %a, ptr %b) {
  call void @llvm.memcpy.p0.p0.i32(ptr %a, ptr %b, i32 100, i1 false)
  ret void
; CHECK-LABEL: test4:
; CHECK:   movl	{{.*}}, (%esp)
; CHECK:   movl	{{.*}}, 4(%esp)
; CHECK:   movl	$100, 8(%esp)
; CHECK:   calll {{.*}}memcpy

; ELF-LABEL: test4:
; ELF:         calll memcpy{{$}}
}

; STDERR-NOT: FastISel missed call:   call x86_thiscallcc void @thiscallfun
%struct.S = type { i8 }
define void @test5() {
entry:
  %s = alloca %struct.S, align 8
; CHECK-LABEL: test5:
; CHECK: subl $12, %esp
; CHECK: leal 8(%esp), %ecx
; CHECK: movl $43, (%esp)
; CHECK: calll {{.*}}thiscallfun
; CHECK: addl $8, %esp
  call x86_thiscallcc void @thiscallfun(ptr %s, i32 43)
  ret void
}
declare x86_thiscallcc void @thiscallfun(ptr, i32) #1

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
