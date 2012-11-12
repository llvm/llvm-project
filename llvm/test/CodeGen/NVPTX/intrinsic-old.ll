; RUN: llc < %s -march=nvptx -mcpu=sm_10 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_10 | FileCheck %s
; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s

define ptx_device i32 @test_tid_x() {
; CHECK: mov.u32 %r0, %tid.x;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.tid.x()
	ret i32 %x
}

define ptx_device i32 @test_tid_y() {
; CHECK: mov.u32 %r0, %tid.y;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.tid.y()
	ret i32 %x
}

define ptx_device i32 @test_tid_z() {
; CHECK: mov.u32 %r0, %tid.z;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.tid.z()
	ret i32 %x
}

define ptx_device i32 @test_tid_w() {
; CHECK: mov.u32 %r0, %tid.w;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.tid.w()
	ret i32 %x
}

define ptx_device i32 @test_ntid_x() {
; CHECK: mov.u32 %r0, %ntid.x;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.ntid.x()
	ret i32 %x
}

define ptx_device i32 @test_ntid_y() {
; CHECK: mov.u32 %r0, %ntid.y;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.ntid.y()
	ret i32 %x
}

define ptx_device i32 @test_ntid_z() {
; CHECK: mov.u32 %r0, %ntid.z;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.ntid.z()
	ret i32 %x
}

define ptx_device i32 @test_ntid_w() {
; CHECK: mov.u32 %r0, %ntid.w;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.ntid.w()
	ret i32 %x
}

define ptx_device i32 @test_laneid() {
; CHECK: mov.u32 %r0, %laneid;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.laneid()
	ret i32 %x
}

define ptx_device i32 @test_warpid() {
; CHECK: mov.u32 %r0, %warpid;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.warpid()
	ret i32 %x
}

define ptx_device i32 @test_nwarpid() {
; CHECK: mov.u32 %r0, %nwarpid;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.nwarpid()
	ret i32 %x
}

define ptx_device i32 @test_ctaid_x() {
; CHECK: mov.u32 %r0, %ctaid.x;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.ctaid.x()
	ret i32 %x
}

define ptx_device i32 @test_ctaid_y() {
; CHECK: mov.u32 %r0, %ctaid.y;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.ctaid.y()
	ret i32 %x
}

define ptx_device i32 @test_ctaid_z() {
; CHECK: mov.u32 %r0, %ctaid.z;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.ctaid.z()
	ret i32 %x
}

define ptx_device i32 @test_ctaid_w() {
; CHECK: mov.u32 %r0, %ctaid.w;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.ctaid.w()
	ret i32 %x
}

define ptx_device i32 @test_nctaid_x() {
; CHECK: mov.u32 %r0, %nctaid.x;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.nctaid.x()
	ret i32 %x
}

define ptx_device i32 @test_nctaid_y() {
; CHECK: mov.u32 %r0, %nctaid.y;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.nctaid.y()
	ret i32 %x
}

define ptx_device i32 @test_nctaid_z() {
; CHECK: mov.u32 %r0, %nctaid.z;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.nctaid.z()
	ret i32 %x
}

define ptx_device i32 @test_nctaid_w() {
; CHECK: mov.u32 %r0, %nctaid.w;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.nctaid.w()
	ret i32 %x
}

define ptx_device i32 @test_smid() {
; CHECK: mov.u32 %r0, %smid;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.smid()
	ret i32 %x
}

define ptx_device i32 @test_nsmid() {
; CHECK: mov.u32 %r0, %nsmid;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.nsmid()
	ret i32 %x
}

define ptx_device i32 @test_gridid() {
; CHECK: mov.u32 %r0, %gridid;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.gridid()
	ret i32 %x
}

define ptx_device i32 @test_lanemask_eq() {
; CHECK: mov.u32 %r0, %lanemask_eq;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.lanemask.eq()
	ret i32 %x
}

define ptx_device i32 @test_lanemask_le() {
; CHECK: mov.u32 %r0, %lanemask_le;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.lanemask.le()
	ret i32 %x
}

define ptx_device i32 @test_lanemask_lt() {
; CHECK: mov.u32 %r0, %lanemask_lt;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.lanemask.lt()
	ret i32 %x
}

define ptx_device i32 @test_lanemask_ge() {
; CHECK: mov.u32 %r0, %lanemask_ge;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.lanemask.ge()
	ret i32 %x
}

define ptx_device i32 @test_lanemask_gt() {
; CHECK: mov.u32 %r0, %lanemask_gt;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.lanemask.gt()
	ret i32 %x
}

define ptx_device i32 @test_clock() {
; CHECK: mov.u32 %r0, %clock;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.clock()
	ret i32 %x
}

define ptx_device i64 @test_clock64() {
; CHECK: mov.u64 %rl0, %clock64;
; CHECK: ret;
	%x = call i64 @llvm.ptx.read.clock64()
	ret i64 %x
}

define ptx_device i32 @test_pm0() {
; CHECK: mov.u32 %r0, %pm0;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.pm0()
	ret i32 %x
}

define ptx_device i32 @test_pm1() {
; CHECK: mov.u32 %r0, %pm1;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.pm1()
	ret i32 %x
}

define ptx_device i32 @test_pm2() {
; CHECK: mov.u32 %r0, %pm2;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.pm2()
	ret i32 %x
}

define ptx_device i32 @test_pm3() {
; CHECK: mov.u32 %r0, %pm3;
; CHECK: ret;
	%x = call i32 @llvm.ptx.read.pm3()
	ret i32 %x
}

define ptx_device void @test_bar_sync() {
; CHECK: bar.sync 0
; CHECK: ret;
	call void @llvm.ptx.bar.sync(i32 0)
	ret void
}

declare i32 @llvm.ptx.read.tid.x()
declare i32 @llvm.ptx.read.tid.y()
declare i32 @llvm.ptx.read.tid.z()
declare i32 @llvm.ptx.read.tid.w()
declare i32 @llvm.ptx.read.ntid.x()
declare i32 @llvm.ptx.read.ntid.y()
declare i32 @llvm.ptx.read.ntid.z()
declare i32 @llvm.ptx.read.ntid.w()

declare i32 @llvm.ptx.read.laneid()
declare i32 @llvm.ptx.read.warpid()
declare i32 @llvm.ptx.read.nwarpid()

declare i32 @llvm.ptx.read.ctaid.x()
declare i32 @llvm.ptx.read.ctaid.y()
declare i32 @llvm.ptx.read.ctaid.z()
declare i32 @llvm.ptx.read.ctaid.w()
declare i32 @llvm.ptx.read.nctaid.x()
declare i32 @llvm.ptx.read.nctaid.y()
declare i32 @llvm.ptx.read.nctaid.z()
declare i32 @llvm.ptx.read.nctaid.w()

declare i32 @llvm.ptx.read.smid()
declare i32 @llvm.ptx.read.nsmid()
declare i32 @llvm.ptx.read.gridid()

declare i32 @llvm.ptx.read.lanemask.eq()
declare i32 @llvm.ptx.read.lanemask.le()
declare i32 @llvm.ptx.read.lanemask.lt()
declare i32 @llvm.ptx.read.lanemask.ge()
declare i32 @llvm.ptx.read.lanemask.gt()

declare i32 @llvm.ptx.read.clock()
declare i64 @llvm.ptx.read.clock64()

declare i32 @llvm.ptx.read.pm0()
declare i32 @llvm.ptx.read.pm1()
declare i32 @llvm.ptx.read.pm2()
declare i32 @llvm.ptx.read.pm3()

declare void @llvm.ptx.bar.sync(i32 %i)
