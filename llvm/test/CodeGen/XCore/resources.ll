; RUN: llc -mtriple=xcore < %s | FileCheck %s

declare ptr addrspace(1) @llvm.xcore.getr.p1(i32 %type)
declare void @llvm.xcore.freer.p1(ptr addrspace(1) %r)
declare i32 @llvm.xcore.in.p1(ptr addrspace(1) %r)
declare i32 @llvm.xcore.int.p1(ptr addrspace(1) %r)
declare i32 @llvm.xcore.inct.p1(ptr addrspace(1) %r)
declare void @llvm.xcore.out.p1(ptr addrspace(1) %r, i32 %value)
declare void @llvm.xcore.outt.p1(ptr addrspace(1) %r, i32 %value)
declare void @llvm.xcore.outct.p1(ptr addrspace(1) %r, i32 %value)
declare void @llvm.xcore.chkct.p1(ptr addrspace(1) %r, i32 %value)
declare i32 @llvm.xcore.testct.p1(ptr addrspace(1) %r)
declare i32 @llvm.xcore.testwct.p1(ptr addrspace(1) %r)
declare void @llvm.xcore.setd.p1(ptr addrspace(1) %r, i32 %value)
declare void @llvm.xcore.setc.p1(ptr addrspace(1) %r, i32 %value)
declare i32 @llvm.xcore.inshr.p1(ptr addrspace(1) %r, i32 %value)
declare i32 @llvm.xcore.outshr.p1(ptr addrspace(1) %r, i32 %value)
declare void @llvm.xcore.clrpt.p1(ptr addrspace(1) %r)
declare void @llvm.xcore.setpt.p1(ptr addrspace(1) %r, i32 %value)
declare i32 @llvm.xcore.getts.p1(ptr addrspace(1) %r)
declare void @llvm.xcore.syncr.p1(ptr addrspace(1) %r)
declare void @llvm.xcore.settw.p1(ptr addrspace(1) %r, i32 %value)
declare void @llvm.xcore.setv.p1(ptr addrspace(1) %r, ptr %p)
declare void @llvm.xcore.setev.p1(ptr addrspace(1) %r, ptr %p)
declare void @llvm.xcore.edu.p1(ptr addrspace(1) %r)
declare void @llvm.xcore.eeu.p1(ptr addrspace(1) %r)
declare void @llvm.xcore.setclk.p1.p1(ptr addrspace(1) %a, ptr addrspace(1) %b)
declare void @llvm.xcore.setrdy.p1.p1(ptr addrspace(1) %a, ptr addrspace(1) %b)
declare void @llvm.xcore.setpsc.p1(ptr addrspace(1) %r, i32 %value)
declare i32 @llvm.xcore.peek.p1(ptr addrspace(1) %r)
declare i32 @llvm.xcore.endin.p1(ptr addrspace(1) %r)

define ptr addrspace(1) @getr() {
; CHECK-LABEL: getr:
; CHECK: getr r0, 5
	%result = call ptr addrspace(1) @llvm.xcore.getr.p1(i32 5)
	ret ptr addrspace(1) %result
}

define void @freer(ptr addrspace(1) %r) {
; CHECK-LABEL: freer:
; CHECK: freer res[r0]
	call void @llvm.xcore.freer.p1(ptr addrspace(1) %r)
	ret void
}

define i32 @in(ptr addrspace(1) %r) {
; CHECK-LABEL: in:
; CHECK: in r0, res[r0]
	%result = call i32 @llvm.xcore.in.p1(ptr addrspace(1) %r)
	ret i32 %result
}

define i32 @int(ptr addrspace(1) %r) {
; CHECK-LABEL: int:
; CHECK: int r0, res[r0]
	%result = call i32 @llvm.xcore.int.p1(ptr addrspace(1) %r)
	ret i32 %result
}

define i32 @inct(ptr addrspace(1) %r) {
; CHECK-LABEL: inct:
; CHECK: inct r0, res[r0]
	%result = call i32 @llvm.xcore.inct.p1(ptr addrspace(1) %r)
	ret i32 %result
}

define void @out(ptr addrspace(1) %r, i32 %value) {
; CHECK-LABEL: out:
; CHECK: out res[r0], r1
	call void @llvm.xcore.out.p1(ptr addrspace(1) %r, i32 %value)
	ret void
}

define void @outt(ptr addrspace(1) %r, i32 %value) {
; CHECK-LABEL: outt:
; CHECK: outt res[r0], r1
	call void @llvm.xcore.outt.p1(ptr addrspace(1) %r, i32 %value)
	ret void
}

define void @outct(ptr addrspace(1) %r, i32 %value) {
; CHECK-LABEL: outct:
; CHECK: outct res[r0], r1
	call void @llvm.xcore.outct.p1(ptr addrspace(1) %r, i32 %value)
	ret void
}

define void @outcti(ptr addrspace(1) %r) {
; CHECK-LABEL: outcti:
; CHECK: outct res[r0], 11
	call void @llvm.xcore.outct.p1(ptr addrspace(1) %r, i32 11)
	ret void
}

define void @chkct(ptr addrspace(1) %r, i32 %value) {
; CHECK-LABEL: chkct:
; CHECK: chkct res[r0], r1
	call void @llvm.xcore.chkct.p1(ptr addrspace(1) %r, i32 %value)
	ret void
}

define void @chkcti(ptr addrspace(1) %r) {
; CHECK-LABEL: chkcti:
; CHECK: chkct res[r0], 11
	call void @llvm.xcore.chkct.p1(ptr addrspace(1) %r, i32 11)
	ret void
}

define void @setd(ptr addrspace(1) %r, i32 %value) {
; CHECK-LABEL: setd:
; CHECK: setd res[r0], r1
	call void @llvm.xcore.setd.p1(ptr addrspace(1) %r, i32 %value)
	ret void
}

define void @setc(ptr addrspace(1) %r, i32 %value) {
; CHECK-LABEL: setc:
; CHECK: setc res[r0], r1
	call void @llvm.xcore.setc.p1(ptr addrspace(1) %r, i32 %value)
	ret void
}

define void @setci(ptr addrspace(1) %r) {
; CHECK-LABEL: setci:
; CHECK: setc res[r0], 2
	call void @llvm.xcore.setc.p1(ptr addrspace(1) %r, i32 2)
	ret void
}

define i32 @inshr(i32 %value, ptr addrspace(1) %r) {
; CHECK-LABEL: inshr:
; CHECK: inshr r0, res[r1]
	%result = call i32 @llvm.xcore.inshr.p1(ptr addrspace(1) %r, i32 %value)
	ret i32 %result
}

define i32 @outshr(i32 %value, ptr addrspace(1) %r) {
; CHECK-LABEL: outshr:
; CHECK: outshr res[r1], r0
	%result = call i32 @llvm.xcore.outshr.p1(ptr addrspace(1) %r, i32 %value)
	ret i32 %result
}

define void @clrpt(ptr addrspace(1) %r) {
; CHECK-LABEL: clrpt:
; CHECK: clrpt res[r0]
	call void @llvm.xcore.clrpt.p1(ptr addrspace(1) %r)
	ret void
}

define void @setpt(ptr addrspace(1) %r, i32 %value) {
; CHECK-LABEL: setpt:
; CHECK: setpt res[r0], r1
	call void @llvm.xcore.setpt.p1(ptr addrspace(1) %r, i32 %value)
	ret void
}

define i32 @getts(ptr addrspace(1) %r) {
; CHECK-LABEL: getts:
; CHECK: getts r0, res[r0]
	%result = call i32 @llvm.xcore.getts.p1(ptr addrspace(1) %r)
	ret i32 %result
}

define void @syncr(ptr addrspace(1) %r) {
; CHECK-LABEL: syncr:
; CHECK: syncr res[r0]
	call void @llvm.xcore.syncr.p1(ptr addrspace(1) %r)
	ret void
}

define void @settw(ptr addrspace(1) %r, i32 %value) {
; CHECK-LABEL: settw:
; CHECK: settw res[r0], r1
	call void @llvm.xcore.settw.p1(ptr addrspace(1) %r, i32 %value)
	ret void
}

define void @setv(ptr addrspace(1) %r, ptr %p) {
; CHECK-LABEL: setv:
; CHECK: mov r11, r1
; CHECK-NEXT: setv res[r0], r11
	call void @llvm.xcore.setv.p1(ptr addrspace(1) %r, ptr %p)
	ret void
}

define void @setev(ptr addrspace(1) %r, ptr %p) {
; CHECK-LABEL: setev:
; CHECK: mov r11, r1
; CHECK-NEXT: setev res[r0], r11
	call void @llvm.xcore.setev.p1(ptr addrspace(1) %r, ptr %p)
	ret void
}

define void @edu(ptr addrspace(1) %r) {
; CHECK-LABEL: edu:
; CHECK: edu res[r0]
	call void @llvm.xcore.edu.p1(ptr addrspace(1) %r)
	ret void
}

define void @eeu(ptr addrspace(1) %r) {
; CHECK-LABEL: eeu:
; CHECK: eeu res[r0]
	call void @llvm.xcore.eeu.p1(ptr addrspace(1) %r)
	ret void
}

define void @setclk(ptr addrspace(1) %a, ptr addrspace(1) %b) {
; CHECK: setclk
; CHECK: setclk res[r0], r1
	call void @llvm.xcore.setclk.p1.p1(ptr addrspace(1) %a, ptr addrspace(1) %b)
	ret void
}

define void @setrdy(ptr addrspace(1) %a, ptr addrspace(1) %b) {
; CHECK: setrdy
; CHECK: setrdy res[r0], r1
	call void @llvm.xcore.setrdy.p1.p1(ptr addrspace(1) %a, ptr addrspace(1) %b)
	ret void
}

define void @setpsc(ptr addrspace(1) %r, i32 %value) {
; CHECK: setpsc
; CHECK: setpsc res[r0], r1
	call void @llvm.xcore.setpsc.p1(ptr addrspace(1) %r, i32 %value)
	ret void
}

define i32 @peek(ptr addrspace(1) %r) {
; CHECK-LABEL: peek:
; CHECK: peek r0, res[r0]
	%result = call i32 @llvm.xcore.peek.p1(ptr addrspace(1) %r)
	ret i32 %result
}

define i32 @endin(ptr addrspace(1) %r) {
; CHECK-LABEL: endin:
; CHECK: endin r0, res[r0]
	%result = call i32 @llvm.xcore.endin.p1(ptr addrspace(1) %r)
	ret i32 %result
}

define i32 @testct(ptr addrspace(1) %r) {
; CHECK-LABEL: testct:
; CHECK: testct r0, res[r0]
	%result = call i32 @llvm.xcore.testct.p1(ptr addrspace(1) %r)
	ret i32 %result
}

define i32 @testwct(ptr addrspace(1) %r) {
; CHECK-LABEL: testwct:
; CHECK: testwct r0, res[r0]
	%result = call i32 @llvm.xcore.testwct.p1(ptr addrspace(1) %r)
	ret i32 %result
}
