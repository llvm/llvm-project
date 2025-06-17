; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck -allow-deprecated-dag-overlap %s
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck -allow-deprecated-dag-overlap %s
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -mtriple=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

define ptx_device i32 @test_tid_x() {
; CHECK: mov.u32 %r{{[0-9]+}}, %tid.x;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
	ret i32 %x
}

define ptx_device i32 @test_tid_y() {
; CHECK: mov.u32 %r{{[0-9]+}}, %tid.y;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
	ret i32 %x
}

define ptx_device i32 @test_tid_z() {
; CHECK: mov.u32 %r{{[0-9]+}}, %tid.z;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
	ret i32 %x
}

define ptx_device i32 @test_tid_w() {
; CHECK: mov.u32 %r{{[0-9]+}}, %tid.w;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.tid.w()
	ret i32 %x
}

define ptx_device i32 @test_ntid_x() {
; CHECK: mov.u32 %r{{[0-9]+}}, %ntid.x;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
	ret i32 %x
}

define ptx_device i32 @test_ntid_y() {
; CHECK: mov.u32 %r{{[0-9]+}}, %ntid.y;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
	ret i32 %x
}

define ptx_device i32 @test_ntid_z() {
; CHECK: mov.u32 %r{{[0-9]+}}, %ntid.z;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
	ret i32 %x
}

define ptx_device i32 @test_ntid_w() {
; CHECK: mov.u32 %r{{[0-9]+}}, %ntid.w;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.ntid.w()
	ret i32 %x
}

define ptx_device i32 @test_laneid() {
; CHECK: mov.u32 %r{{[0-9]+}}, %laneid;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.laneid()
	ret i32 %x
}

define ptx_device i32 @test_warpsize() {
; CHECK: mov.u32 %r{{[0-9]+}}, WARP_SZ;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
	ret i32 %x
}

define ptx_device i32 @test_warpid() {
; CHECK: mov.u32 %r{{[0-9]+}}, %warpid;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.warpid()
	ret i32 %x
}

define ptx_device i32 @test_nwarpid() {
; CHECK: mov.u32 %r{{[0-9]+}}, %nwarpid;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.nwarpid()
	ret i32 %x
}

define ptx_device i32 @test_ctaid_y() {
; CHECK: mov.u32 %r{{[0-9]+}}, %ctaid.y;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
	ret i32 %x
}

define ptx_device i32 @test_ctaid_z() {
; CHECK: mov.u32 %r{{[0-9]+}}, %ctaid.z;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
	ret i32 %x
}

define ptx_device i32 @test_ctaid_x() {
; CHECK: mov.u32 %r{{[0-9]+}}, %ctaid.x;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
	ret i32 %x
}

define ptx_device i32 @test_ctaid_w() {
; CHECK: mov.u32 %r{{[0-9]+}}, %ctaid.w;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.w()
	ret i32 %x
}

define ptx_device i32 @test_nctaid_y() {
; CHECK: mov.u32 %r{{[0-9]+}}, %nctaid.y;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
	ret i32 %x
}

define ptx_device i32 @test_nctaid_z() {
; CHECK: mov.u32 %r{{[0-9]+}}, %nctaid.z;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
	ret i32 %x
}

define ptx_device i32 @test_nctaid_x() {
; CHECK: mov.u32 %r{{[0-9]+}}, %nctaid.x;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
	ret i32 %x
}

define ptx_device i32 @test_already_has_range_md() {
; CHECK: mov.u32 %r{{[0-9]+}}, %nctaid.x;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x(), !range !0
	ret i32 %x
}


define ptx_device i32 @test_nctaid_w() {
; CHECK: mov.u32 %r{{[0-9]+}}, %nctaid.w;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.w()
	ret i32 %x
}

define ptx_device i32 @test_smid() {
; CHECK: mov.u32 %r{{[0-9]+}}, %smid;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.smid()
	ret i32 %x
}

define ptx_device i32 @test_nsmid() {
; CHECK: mov.u32 %r{{[0-9]+}}, %nsmid;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.nsmid()
	ret i32 %x
}

define ptx_device i32 @test_gridid() {
; CHECK: mov.u32 %r{{[0-9]+}}, %gridid;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.gridid()
	ret i32 %x
}

define ptx_device i32 @test_lanemask_eq() {
; CHECK: mov.u32 %r{{[0-9]+}}, %lanemask_eq;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.lanemask.eq()
	ret i32 %x
}

define ptx_device i32 @test_lanemask_le() {
; CHECK: mov.u32 %r{{[0-9]+}}, %lanemask_le;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.lanemask.le()
	ret i32 %x
}

define ptx_device i32 @test_lanemask_lt() {
; CHECK: mov.u32 %r{{[0-9]+}}, %lanemask_lt;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.lanemask.lt()
	ret i32 %x
}

define ptx_device i32 @test_lanemask_ge() {
; CHECK: mov.u32 %r{{[0-9]+}}, %lanemask_ge;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.lanemask.ge()
	ret i32 %x
}

define ptx_device i32 @test_lanemask_gt() {
; CHECK: mov.u32 %r{{[0-9]+}}, %lanemask_gt;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.lanemask.gt()
	ret i32 %x
}

define ptx_device i32 @test_clock() {
; CHECK: mov.u32 %r{{[0-9]+}}, %clock;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.clock()
	ret i32 %x
}

define ptx_device i64 @test_clock64() {
; CHECK: mov.u64 %rd{{[0-9]+}}, %clock64;
; CHECK: ret;
	%x = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
	ret i64 %x
}

define ptx_device i32 @test_pm0() {
; CHECK: mov.u32 %r{{[0-9]+}}, %pm0;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.pm0()
	ret i32 %x
}

define ptx_device i32 @test_pm1() {
; CHECK: mov.u32 %r{{[0-9]+}}, %pm1;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.pm1()
	ret i32 %x
}

define ptx_device i32 @test_pm2() {
; CHECK: mov.u32 %r{{[0-9]+}}, %pm2;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.pm2()
	ret i32 %x
}

define ptx_device i32 @test_pm3() {
; CHECK: mov.u32 %r{{[0-9]+}}, %pm3;
; CHECK: ret;
	%x = call i32 @llvm.nvvm.read.ptx.sreg.pm3()
	ret i32 %x
}

define ptx_device void @test_bar_sync() {
; CHECK: bar.sync 0
; CHECK: ret;
	call void @llvm.nvvm.bar.sync(i32 0)
	ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.w()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.w()

declare i32 @llvm.nvvm.read.ptx.sreg.warpsize()
declare i32 @llvm.nvvm.read.ptx.sreg.laneid()
declare i32 @llvm.nvvm.read.ptx.sreg.warpid()
declare i32 @llvm.nvvm.read.ptx.sreg.nwarpid()

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.w()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.w()

declare i32 @llvm.nvvm.read.ptx.sreg.smid()
declare i32 @llvm.nvvm.read.ptx.sreg.nsmid()
declare i32 @llvm.nvvm.read.ptx.sreg.gridid()

declare i32 @llvm.nvvm.read.ptx.sreg.lanemask.eq()
declare i32 @llvm.nvvm.read.ptx.sreg.lanemask.le()
declare i32 @llvm.nvvm.read.ptx.sreg.lanemask.lt()
declare i32 @llvm.nvvm.read.ptx.sreg.lanemask.ge()
declare i32 @llvm.nvvm.read.ptx.sreg.lanemask.gt()

declare i32 @llvm.nvvm.read.ptx.sreg.clock()
declare i64 @llvm.nvvm.read.ptx.sreg.clock64()

declare i32 @llvm.nvvm.read.ptx.sreg.pm0()
declare i32 @llvm.nvvm.read.ptx.sreg.pm1()
declare i32 @llvm.nvvm.read.ptx.sreg.pm2()
declare i32 @llvm.nvvm.read.ptx.sreg.pm3()

declare void @llvm.nvvm.bar.sync(i32 %i)

!0 = !{i32 0, i32 19}
