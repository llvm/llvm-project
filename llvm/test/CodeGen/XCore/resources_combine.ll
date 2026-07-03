; RUN: llc -mtriple=xcore < %s | FileCheck %s

declare i32 @llvm.xcore.int.p1(ptr addrspace(1) %r)
declare i32 @llvm.xcore.inct.p1(ptr addrspace(1) %r)
declare i32 @llvm.xcore.testct.p1(ptr addrspace(1) %r)
declare i32 @llvm.xcore.testwct.p1(ptr addrspace(1) %r)
declare i32 @llvm.xcore.getts.p1(ptr addrspace(1) %r)
declare void @llvm.xcore.outt.p1(ptr addrspace(1) %r, i32 %value)
declare void @llvm.xcore.outct.p1(ptr addrspace(1) %r, i32 %value)
declare void @llvm.xcore.chkct.p1(ptr addrspace(1) %r, i32 %value)
declare void @llvm.xcore.setpt.p1(ptr addrspace(1) %r, i32 %value)

define i32 @int(ptr addrspace(1) %r) nounwind {
; CHECK-LABEL: int:
; CHECK: int r0, res[r0]
; CHECK-NEXT: retsp 0
	%result = call i32 @llvm.xcore.int.p1(ptr addrspace(1) %r)
	%trunc = and i32 %result, 255
	ret i32 %trunc
}

define i32 @inct(ptr addrspace(1) %r) nounwind {
; CHECK-LABEL: inct:
; CHECK: inct r0, res[r0]
; CHECK-NEXT: retsp 0
	%result = call i32 @llvm.xcore.inct.p1(ptr addrspace(1) %r)
	%trunc = and i32 %result, 255
	ret i32 %trunc
}

define i32 @testct(ptr addrspace(1) %r) nounwind {
; CHECK-LABEL: testct:
; CHECK: testct r0, res[r0]
; CHECK-NEXT: retsp 0
	%result = call i32 @llvm.xcore.testct.p1(ptr addrspace(1) %r)
	%trunc = and i32 %result, 1
	ret i32 %trunc
}

define i32 @testwct(ptr addrspace(1) %r) nounwind {
; CHECK-LABEL: testwct:
; CHECK: testwct r0, res[r0]
; CHECK-NEXT: retsp 0
	%result = call i32 @llvm.xcore.testwct.p1(ptr addrspace(1) %r)
	%trunc = and i32 %result, 7
	ret i32 %trunc
}

define i32 @getts(ptr addrspace(1) %r) nounwind {
; CHECK-LABEL: getts:
; CHECK: getts r0, res[r0]
; CHECK-NEXT: retsp 0
	%result = call i32 @llvm.xcore.getts.p1(ptr addrspace(1) %r)
	%trunc = and i32 %result, 65535
	ret i32 %result
}

define void @outt(ptr addrspace(1) %r, i32 %value) nounwind {
; CHECK-LABEL: outt:
; CHECK-NOT: zext
; CHECK: outt res[r0], r1
; CHECK-NEXT: retsp 0
	%trunc = and i32 %value, 255
	call void @llvm.xcore.outt.p1(ptr addrspace(1) %r, i32 %trunc)
	ret void
}

define void @outct(ptr addrspace(1) %r, i32 %value) nounwind {
; CHECK-LABEL: outct:
; CHECK-NOT: zext
; CHECK: outct res[r0], r1
	%trunc = and i32 %value, 255
	call void @llvm.xcore.outct.p1(ptr addrspace(1) %r, i32 %trunc)
	ret void
}

define void @chkct(ptr addrspace(1) %r, i32 %value) nounwind {
; CHECK-LABEL: chkct:
; CHECK-NOT: zext
; CHECK: chkct res[r0], r1
	%trunc = and i32 %value, 255
	call void @llvm.xcore.chkct.p1(ptr addrspace(1) %r, i32 %trunc)
	ret void
}

define void @setpt(ptr addrspace(1) %r, i32 %value) nounwind {
; CHECK-LABEL: setpt:
; CHECK-NOT: zext
; CHECK: setpt res[r0], r1
	%trunc = and i32 %value, 65535
	call void @llvm.xcore.setpt.p1(ptr addrspace(1) %r, i32 %trunc)
	ret void
}
