; RUN: llc -march=xcore < %s | FileCheck %s

declare i8 addrspace(1)* @llvm.xcore.getst.p1i8.p1i8(i8 addrspace(1)* %r)
declare void @llvm.xcore.msync.p1i8(i8 addrspace(1)* %r)
declare void @llvm.xcore.ssync()
declare void @llvm.xcore.mjoin.p1i8(i8 addrspace(1)* %r)
declare void @llvm.xcore.initsp.p1i8(i8 addrspace(1)* %r, i8* %value)
declare void @llvm.xcore.initpc.p1i8(i8 addrspace(1)* %r, i8* %value)
declare void @llvm.xcore.initlr.p1i8(i8 addrspace(1)* %r, i8* %value)
declare void @llvm.xcore.initcp.p1i8(i8 addrspace(1)* %r, i8* %value)
declare void @llvm.xcore.initdp.p1i8(i8 addrspace(1)* %r, i8* %value)

define i8 addrspace(1)* @getst(i8 addrspace(1)* %r) {
; CHECK: getst:
; CHECK: getst r0, res[r0]
        %result = call i8 addrspace(1)* @llvm.xcore.getst.p1i8.p1i8(i8 addrspace(1)* %r)
        ret i8 addrspace(1)* %result
}

define void @ssync() {
; CHECK: ssync:
; CHECK: ssync
	call void @llvm.xcore.ssync()
	ret void
}

define void @mjoin(i8 addrspace(1)* %r) {
; CHECK: mjoin:
; CHECK: mjoin res[r0]
	call void @llvm.xcore.mjoin.p1i8(i8 addrspace(1)* %r)
	ret void
}

define void @initsp(i8 addrspace(1)* %t, i8* %src) {
; CHECK: initsp:
; CHECK: init t[r0]:sp, r1
        call void @llvm.xcore.initsp.p1i8(i8 addrspace(1)* %t, i8* %src)
        ret void
}

define void @initpc(i8 addrspace(1)* %t, i8* %src) {
; CHECK: initpc:
; CHECK: init t[r0]:pc, r1
        call void @llvm.xcore.initpc.p1i8(i8 addrspace(1)* %t, i8* %src)
        ret void
}

define void @initlr(i8 addrspace(1)* %t, i8* %src) {
; CHECK: initlr:
; CHECK: init t[r0]:lr, r1
        call void @llvm.xcore.initlr.p1i8(i8 addrspace(1)* %t, i8* %src)
        ret void
}

define void @initcp(i8 addrspace(1)* %t, i8* %src) {
; CHECK: initcp:
; CHECK: init t[r0]:cp, r1
        call void @llvm.xcore.initcp.p1i8(i8 addrspace(1)* %t, i8* %src)
        ret void
}

define void @initdp(i8 addrspace(1)* %t, i8* %src) {
; CHECK: initdp:
; CHECK: init t[r0]:dp, r1
        call void @llvm.xcore.initdp.p1i8(i8 addrspace(1)* %t, i8* %src)
        ret void
}
