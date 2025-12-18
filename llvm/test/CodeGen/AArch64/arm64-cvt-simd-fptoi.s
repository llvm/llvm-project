	.file	"arm64-cvt-simd-fptoi.ll"
	.text
	.globl	test_fptosi_f16_i32_simd        // -- Begin function test_fptosi_f16_i32_simd
	.p2align	2
	.type	test_fptosi_f16_i32_simd,@function
test_fptosi_f16_i32_simd:               // @test_fptosi_f16_i32_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	s0, h0
	ret
.Lfunc_end0:
	.size	test_fptosi_f16_i32_simd, .Lfunc_end0-test_fptosi_f16_i32_simd
	.cfi_endproc
                                        // -- End function
	.globl	test_fptosi_f16_i64_simd        // -- Begin function test_fptosi_f16_i64_simd
	.p2align	2
	.type	test_fptosi_f16_i64_simd,@function
test_fptosi_f16_i64_simd:               // @test_fptosi_f16_i64_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, h0
	ret
.Lfunc_end1:
	.size	test_fptosi_f16_i64_simd, .Lfunc_end1-test_fptosi_f16_i64_simd
	.cfi_endproc
                                        // -- End function
	.globl	test_fptosi_f64_i32_simd        // -- Begin function test_fptosi_f64_i32_simd
	.p2align	2
	.type	test_fptosi_f64_i32_simd,@function
test_fptosi_f64_i32_simd:               // @test_fptosi_f64_i32_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	s0, d0
	ret
.Lfunc_end2:
	.size	test_fptosi_f64_i32_simd, .Lfunc_end2-test_fptosi_f64_i32_simd
	.cfi_endproc
                                        // -- End function
	.globl	test_fptosi_f32_i64_simd        // -- Begin function test_fptosi_f32_i64_simd
	.p2align	2
	.type	test_fptosi_f32_i64_simd,@function
test_fptosi_f32_i64_simd:               // @test_fptosi_f32_i64_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, s0
	ret
.Lfunc_end3:
	.size	test_fptosi_f32_i64_simd, .Lfunc_end3-test_fptosi_f32_i64_simd
	.cfi_endproc
                                        // -- End function
	.globl	test_fptosi_f64_i64_simd        // -- Begin function test_fptosi_f64_i64_simd
	.p2align	2
	.type	test_fptosi_f64_i64_simd,@function
test_fptosi_f64_i64_simd:               // @test_fptosi_f64_i64_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, d0
	ret
.Lfunc_end4:
	.size	test_fptosi_f64_i64_simd, .Lfunc_end4-test_fptosi_f64_i64_simd
	.cfi_endproc
                                        // -- End function
	.globl	test_fptosi_f32_i32_simd        // -- Begin function test_fptosi_f32_i32_simd
	.p2align	2
	.type	test_fptosi_f32_i32_simd,@function
test_fptosi_f32_i32_simd:               // @test_fptosi_f32_i32_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	s0, s0
	ret
.Lfunc_end5:
	.size	test_fptosi_f32_i32_simd, .Lfunc_end5-test_fptosi_f32_i32_simd
	.cfi_endproc
                                        // -- End function
	.globl	test_fptoui_f16_i32_simd        // -- Begin function test_fptoui_f16_i32_simd
	.p2align	2
	.type	test_fptoui_f16_i32_simd,@function
test_fptoui_f16_i32_simd:               // @test_fptoui_f16_i32_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	s0, h0
	ret
.Lfunc_end6:
	.size	test_fptoui_f16_i32_simd, .Lfunc_end6-test_fptoui_f16_i32_simd
	.cfi_endproc
                                        // -- End function
	.globl	test_fptoui_f16_i64_simd        // -- Begin function test_fptoui_f16_i64_simd
	.p2align	2
	.type	test_fptoui_f16_i64_simd,@function
test_fptoui_f16_i64_simd:               // @test_fptoui_f16_i64_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	d0, h0
	ret
.Lfunc_end7:
	.size	test_fptoui_f16_i64_simd, .Lfunc_end7-test_fptoui_f16_i64_simd
	.cfi_endproc
                                        // -- End function
	.globl	test_fptoui_f64_i32_simd        // -- Begin function test_fptoui_f64_i32_simd
	.p2align	2
	.type	test_fptoui_f64_i32_simd,@function
test_fptoui_f64_i32_simd:               // @test_fptoui_f64_i32_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	s0, d0
	ret
.Lfunc_end8:
	.size	test_fptoui_f64_i32_simd, .Lfunc_end8-test_fptoui_f64_i32_simd
	.cfi_endproc
                                        // -- End function
	.globl	test_fptoui_f32_i64_simd        // -- Begin function test_fptoui_f32_i64_simd
	.p2align	2
	.type	test_fptoui_f32_i64_simd,@function
test_fptoui_f32_i64_simd:               // @test_fptoui_f32_i64_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	d0, s0
	ret
.Lfunc_end9:
	.size	test_fptoui_f32_i64_simd, .Lfunc_end9-test_fptoui_f32_i64_simd
	.cfi_endproc
                                        // -- End function
	.globl	test_fptoui_f64_i64_simd        // -- Begin function test_fptoui_f64_i64_simd
	.p2align	2
	.type	test_fptoui_f64_i64_simd,@function
test_fptoui_f64_i64_simd:               // @test_fptoui_f64_i64_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	d0, d0
	ret
.Lfunc_end10:
	.size	test_fptoui_f64_i64_simd, .Lfunc_end10-test_fptoui_f64_i64_simd
	.cfi_endproc
                                        // -- End function
	.globl	test_fptoui_f32_i32_simd        // -- Begin function test_fptoui_f32_i32_simd
	.p2align	2
	.type	test_fptoui_f32_i32_simd,@function
test_fptoui_f32_i32_simd:               // @test_fptoui_f32_i32_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	s0, s0
	ret
.Lfunc_end11:
	.size	test_fptoui_f32_i32_simd, .Lfunc_end11-test_fptoui_f32_i32_simd
	.cfi_endproc
                                        // -- End function
	.globl	fptosi_i32_f16_simd             // -- Begin function fptosi_i32_f16_simd
	.p2align	2
	.type	fptosi_i32_f16_simd,@function
fptosi_i32_f16_simd:                    // @fptosi_i32_f16_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	s0, h0
	ret
.Lfunc_end12:
	.size	fptosi_i32_f16_simd, .Lfunc_end12-fptosi_i32_f16_simd
	.cfi_endproc
                                        // -- End function
	.globl	fptosi_i64_f16_simd             // -- Begin function fptosi_i64_f16_simd
	.p2align	2
	.type	fptosi_i64_f16_simd,@function
fptosi_i64_f16_simd:                    // @fptosi_i64_f16_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, h0
	ret
.Lfunc_end13:
	.size	fptosi_i64_f16_simd, .Lfunc_end13-fptosi_i64_f16_simd
	.cfi_endproc
                                        // -- End function
	.globl	fptosi_i64_f32_simd             // -- Begin function fptosi_i64_f32_simd
	.p2align	2
	.type	fptosi_i64_f32_simd,@function
fptosi_i64_f32_simd:                    // @fptosi_i64_f32_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, s0
	ret
.Lfunc_end14:
	.size	fptosi_i64_f32_simd, .Lfunc_end14-fptosi_i64_f32_simd
	.cfi_endproc
                                        // -- End function
	.globl	fptosi_i32_f64_simd             // -- Begin function fptosi_i32_f64_simd
	.p2align	2
	.type	fptosi_i32_f64_simd,@function
fptosi_i32_f64_simd:                    // @fptosi_i32_f64_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	s0, d0
	ret
.Lfunc_end15:
	.size	fptosi_i32_f64_simd, .Lfunc_end15-fptosi_i32_f64_simd
	.cfi_endproc
                                        // -- End function
	.globl	fptosi_i64_f64_simd             // -- Begin function fptosi_i64_f64_simd
	.p2align	2
	.type	fptosi_i64_f64_simd,@function
fptosi_i64_f64_simd:                    // @fptosi_i64_f64_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, d0
	ret
.Lfunc_end16:
	.size	fptosi_i64_f64_simd, .Lfunc_end16-fptosi_i64_f64_simd
	.cfi_endproc
                                        // -- End function
	.globl	fptosi_i32_f32_simd             // -- Begin function fptosi_i32_f32_simd
	.p2align	2
	.type	fptosi_i32_f32_simd,@function
fptosi_i32_f32_simd:                    // @fptosi_i32_f32_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	s0, s0
	ret
.Lfunc_end17:
	.size	fptosi_i32_f32_simd, .Lfunc_end17-fptosi_i32_f32_simd
	.cfi_endproc
                                        // -- End function
	.globl	fptoui_i32_f16_simd             // -- Begin function fptoui_i32_f16_simd
	.p2align	2
	.type	fptoui_i32_f16_simd,@function
fptoui_i32_f16_simd:                    // @fptoui_i32_f16_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	s0, h0
	ret
.Lfunc_end18:
	.size	fptoui_i32_f16_simd, .Lfunc_end18-fptoui_i32_f16_simd
	.cfi_endproc
                                        // -- End function
	.globl	fptoui_i64_f16_simd             // -- Begin function fptoui_i64_f16_simd
	.p2align	2
	.type	fptoui_i64_f16_simd,@function
fptoui_i64_f16_simd:                    // @fptoui_i64_f16_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	d0, h0
	ret
.Lfunc_end19:
	.size	fptoui_i64_f16_simd, .Lfunc_end19-fptoui_i64_f16_simd
	.cfi_endproc
                                        // -- End function
	.globl	fptoui_i64_f32_simd             // -- Begin function fptoui_i64_f32_simd
	.p2align	2
	.type	fptoui_i64_f32_simd,@function
fptoui_i64_f32_simd:                    // @fptoui_i64_f32_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	d0, s0
	ret
.Lfunc_end20:
	.size	fptoui_i64_f32_simd, .Lfunc_end20-fptoui_i64_f32_simd
	.cfi_endproc
                                        // -- End function
	.globl	fptoui_i32_f64_simd             // -- Begin function fptoui_i32_f64_simd
	.p2align	2
	.type	fptoui_i32_f64_simd,@function
fptoui_i32_f64_simd:                    // @fptoui_i32_f64_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	s0, d0
	ret
.Lfunc_end21:
	.size	fptoui_i32_f64_simd, .Lfunc_end21-fptoui_i32_f64_simd
	.cfi_endproc
                                        // -- End function
	.globl	fptoui_i64_f64_simd             // -- Begin function fptoui_i64_f64_simd
	.p2align	2
	.type	fptoui_i64_f64_simd,@function
fptoui_i64_f64_simd:                    // @fptoui_i64_f64_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	d0, d0
	ret
.Lfunc_end22:
	.size	fptoui_i64_f64_simd, .Lfunc_end22-fptoui_i64_f64_simd
	.cfi_endproc
                                        // -- End function
	.globl	fptoui_i32_f32_simd             // -- Begin function fptoui_i32_f32_simd
	.p2align	2
	.type	fptoui_i32_f32_simd,@function
fptoui_i32_f32_simd:                    // @fptoui_i32_f32_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	s0, s0
	ret
.Lfunc_end23:
	.size	fptoui_i32_f32_simd, .Lfunc_end23-fptoui_i32_f32_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtas_ds_round_simd            // -- Begin function fcvtas_ds_round_simd
	.p2align	2
	.type	fcvtas_ds_round_simd,@function
fcvtas_ds_round_simd:                   // @fcvtas_ds_round_simd
	.cfi_startproc
// %bb.0:
	fcvtas	d0, s0
	ret
.Lfunc_end24:
	.size	fcvtas_ds_round_simd, .Lfunc_end24-fcvtas_ds_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtas_sd_round_simd            // -- Begin function fcvtas_sd_round_simd
	.p2align	2
	.type	fcvtas_sd_round_simd,@function
fcvtas_sd_round_simd:                   // @fcvtas_sd_round_simd
	.cfi_startproc
// %bb.0:
	fcvtas	s0, d0
	ret
.Lfunc_end25:
	.size	fcvtas_sd_round_simd, .Lfunc_end25-fcvtas_sd_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtas_ss_round_simd            // -- Begin function fcvtas_ss_round_simd
	.p2align	2
	.type	fcvtas_ss_round_simd,@function
fcvtas_ss_round_simd:                   // @fcvtas_ss_round_simd
	.cfi_startproc
// %bb.0:
	fcvtas	s0, s0
	ret
.Lfunc_end26:
	.size	fcvtas_ss_round_simd, .Lfunc_end26-fcvtas_ss_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtas_dd_round_simd            // -- Begin function fcvtas_dd_round_simd
	.p2align	2
	.type	fcvtas_dd_round_simd,@function
fcvtas_dd_round_simd:                   // @fcvtas_dd_round_simd
	.cfi_startproc
// %bb.0:
	fcvtas	d0, d0
	ret
.Lfunc_end27:
	.size	fcvtas_dd_round_simd, .Lfunc_end27-fcvtas_dd_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtau_ds_round_simd            // -- Begin function fcvtau_ds_round_simd
	.p2align	2
	.type	fcvtau_ds_round_simd,@function
fcvtau_ds_round_simd:                   // @fcvtau_ds_round_simd
	.cfi_startproc
// %bb.0:
	fcvtau	d0, s0
	ret
.Lfunc_end28:
	.size	fcvtau_ds_round_simd, .Lfunc_end28-fcvtau_ds_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtau_sd_round_simd            // -- Begin function fcvtau_sd_round_simd
	.p2align	2
	.type	fcvtau_sd_round_simd,@function
fcvtau_sd_round_simd:                   // @fcvtau_sd_round_simd
	.cfi_startproc
// %bb.0:
	fcvtau	s0, d0
	ret
.Lfunc_end29:
	.size	fcvtau_sd_round_simd, .Lfunc_end29-fcvtau_sd_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtau_ss_round_simd            // -- Begin function fcvtau_ss_round_simd
	.p2align	2
	.type	fcvtau_ss_round_simd,@function
fcvtau_ss_round_simd:                   // @fcvtau_ss_round_simd
	.cfi_startproc
// %bb.0:
	fcvtas	s0, s0
	ret
.Lfunc_end30:
	.size	fcvtau_ss_round_simd, .Lfunc_end30-fcvtau_ss_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtau_dd_round_simd            // -- Begin function fcvtau_dd_round_simd
	.p2align	2
	.type	fcvtau_dd_round_simd,@function
fcvtau_dd_round_simd:                   // @fcvtau_dd_round_simd
	.cfi_startproc
// %bb.0:
	fcvtas	d0, d0
	ret
.Lfunc_end31:
	.size	fcvtau_dd_round_simd, .Lfunc_end31-fcvtau_dd_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtms_ds_round_simd            // -- Begin function fcvtms_ds_round_simd
	.p2align	2
	.type	fcvtms_ds_round_simd,@function
fcvtms_ds_round_simd:                   // @fcvtms_ds_round_simd
	.cfi_startproc
// %bb.0:
	fcvtms	d0, s0
	ret
.Lfunc_end32:
	.size	fcvtms_ds_round_simd, .Lfunc_end32-fcvtms_ds_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtms_sd_round_simd            // -- Begin function fcvtms_sd_round_simd
	.p2align	2
	.type	fcvtms_sd_round_simd,@function
fcvtms_sd_round_simd:                   // @fcvtms_sd_round_simd
	.cfi_startproc
// %bb.0:
	fcvtms	s0, d0
	ret
.Lfunc_end33:
	.size	fcvtms_sd_round_simd, .Lfunc_end33-fcvtms_sd_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtms_ss_round_simd            // -- Begin function fcvtms_ss_round_simd
	.p2align	2
	.type	fcvtms_ss_round_simd,@function
fcvtms_ss_round_simd:                   // @fcvtms_ss_round_simd
	.cfi_startproc
// %bb.0:
	fcvtms	s0, s0
	ret
.Lfunc_end34:
	.size	fcvtms_ss_round_simd, .Lfunc_end34-fcvtms_ss_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtms_dd_round_simd            // -- Begin function fcvtms_dd_round_simd
	.p2align	2
	.type	fcvtms_dd_round_simd,@function
fcvtms_dd_round_simd:                   // @fcvtms_dd_round_simd
	.cfi_startproc
// %bb.0:
	fcvtms	d0, d0
	ret
.Lfunc_end35:
	.size	fcvtms_dd_round_simd, .Lfunc_end35-fcvtms_dd_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtmu_ds_round_simd            // -- Begin function fcvtmu_ds_round_simd
	.p2align	2
	.type	fcvtmu_ds_round_simd,@function
fcvtmu_ds_round_simd:                   // @fcvtmu_ds_round_simd
	.cfi_startproc
// %bb.0:
	fcvtmu	d0, s0
	ret
.Lfunc_end36:
	.size	fcvtmu_ds_round_simd, .Lfunc_end36-fcvtmu_ds_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtmu_sd_round_simd            // -- Begin function fcvtmu_sd_round_simd
	.p2align	2
	.type	fcvtmu_sd_round_simd,@function
fcvtmu_sd_round_simd:                   // @fcvtmu_sd_round_simd
	.cfi_startproc
// %bb.0:
	fcvtmu	s0, d0
	ret
.Lfunc_end37:
	.size	fcvtmu_sd_round_simd, .Lfunc_end37-fcvtmu_sd_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtmu_ss_round_simd            // -- Begin function fcvtmu_ss_round_simd
	.p2align	2
	.type	fcvtmu_ss_round_simd,@function
fcvtmu_ss_round_simd:                   // @fcvtmu_ss_round_simd
	.cfi_startproc
// %bb.0:
	fcvtms	s0, s0
	ret
.Lfunc_end38:
	.size	fcvtmu_ss_round_simd, .Lfunc_end38-fcvtmu_ss_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtmu_dd_round_simd            // -- Begin function fcvtmu_dd_round_simd
	.p2align	2
	.type	fcvtmu_dd_round_simd,@function
fcvtmu_dd_round_simd:                   // @fcvtmu_dd_round_simd
	.cfi_startproc
// %bb.0:
	fcvtms	d0, d0
	ret
.Lfunc_end39:
	.size	fcvtmu_dd_round_simd, .Lfunc_end39-fcvtmu_dd_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtps_ds_round_simd            // -- Begin function fcvtps_ds_round_simd
	.p2align	2
	.type	fcvtps_ds_round_simd,@function
fcvtps_ds_round_simd:                   // @fcvtps_ds_round_simd
	.cfi_startproc
// %bb.0:
	fcvtps	d0, s0
	ret
.Lfunc_end40:
	.size	fcvtps_ds_round_simd, .Lfunc_end40-fcvtps_ds_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtps_sd_round_simd            // -- Begin function fcvtps_sd_round_simd
	.p2align	2
	.type	fcvtps_sd_round_simd,@function
fcvtps_sd_round_simd:                   // @fcvtps_sd_round_simd
	.cfi_startproc
// %bb.0:
	fcvtps	s0, d0
	ret
.Lfunc_end41:
	.size	fcvtps_sd_round_simd, .Lfunc_end41-fcvtps_sd_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtps_ss_round_simd            // -- Begin function fcvtps_ss_round_simd
	.p2align	2
	.type	fcvtps_ss_round_simd,@function
fcvtps_ss_round_simd:                   // @fcvtps_ss_round_simd
	.cfi_startproc
// %bb.0:
	fcvtps	s0, s0
	ret
.Lfunc_end42:
	.size	fcvtps_ss_round_simd, .Lfunc_end42-fcvtps_ss_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtps_dd_round_simd            // -- Begin function fcvtps_dd_round_simd
	.p2align	2
	.type	fcvtps_dd_round_simd,@function
fcvtps_dd_round_simd:                   // @fcvtps_dd_round_simd
	.cfi_startproc
// %bb.0:
	fcvtps	d0, d0
	ret
.Lfunc_end43:
	.size	fcvtps_dd_round_simd, .Lfunc_end43-fcvtps_dd_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtpu_ds_round_simd            // -- Begin function fcvtpu_ds_round_simd
	.p2align	2
	.type	fcvtpu_ds_round_simd,@function
fcvtpu_ds_round_simd:                   // @fcvtpu_ds_round_simd
	.cfi_startproc
// %bb.0:
	fcvtpu	d0, s0
	ret
.Lfunc_end44:
	.size	fcvtpu_ds_round_simd, .Lfunc_end44-fcvtpu_ds_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtpu_sd_round_simd            // -- Begin function fcvtpu_sd_round_simd
	.p2align	2
	.type	fcvtpu_sd_round_simd,@function
fcvtpu_sd_round_simd:                   // @fcvtpu_sd_round_simd
	.cfi_startproc
// %bb.0:
	fcvtpu	s0, d0
	ret
.Lfunc_end45:
	.size	fcvtpu_sd_round_simd, .Lfunc_end45-fcvtpu_sd_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtpu_ss_round_simd            // -- Begin function fcvtpu_ss_round_simd
	.p2align	2
	.type	fcvtpu_ss_round_simd,@function
fcvtpu_ss_round_simd:                   // @fcvtpu_ss_round_simd
	.cfi_startproc
// %bb.0:
	fcvtps	s0, s0
	ret
.Lfunc_end46:
	.size	fcvtpu_ss_round_simd, .Lfunc_end46-fcvtpu_ss_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtpu_dd_round_simd            // -- Begin function fcvtpu_dd_round_simd
	.p2align	2
	.type	fcvtpu_dd_round_simd,@function
fcvtpu_dd_round_simd:                   // @fcvtpu_dd_round_simd
	.cfi_startproc
// %bb.0:
	fcvtps	d0, d0
	ret
.Lfunc_end47:
	.size	fcvtpu_dd_round_simd, .Lfunc_end47-fcvtpu_dd_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_ds_round_simd            // -- Begin function fcvtzs_ds_round_simd
	.p2align	2
	.type	fcvtzs_ds_round_simd,@function
fcvtzs_ds_round_simd:                   // @fcvtzs_ds_round_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, s0
	ret
.Lfunc_end48:
	.size	fcvtzs_ds_round_simd, .Lfunc_end48-fcvtzs_ds_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_sd_round_simd            // -- Begin function fcvtzs_sd_round_simd
	.p2align	2
	.type	fcvtzs_sd_round_simd,@function
fcvtzs_sd_round_simd:                   // @fcvtzs_sd_round_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	s0, d0
	ret
.Lfunc_end49:
	.size	fcvtzs_sd_round_simd, .Lfunc_end49-fcvtzs_sd_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_ss_round_simd            // -- Begin function fcvtzs_ss_round_simd
	.p2align	2
	.type	fcvtzs_ss_round_simd,@function
fcvtzs_ss_round_simd:                   // @fcvtzs_ss_round_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	s0, s0
	ret
.Lfunc_end50:
	.size	fcvtzs_ss_round_simd, .Lfunc_end50-fcvtzs_ss_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_dd_round_simd            // -- Begin function fcvtzs_dd_round_simd
	.p2align	2
	.type	fcvtzs_dd_round_simd,@function
fcvtzs_dd_round_simd:                   // @fcvtzs_dd_round_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, d0
	ret
.Lfunc_end51:
	.size	fcvtzs_dd_round_simd, .Lfunc_end51-fcvtzs_dd_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_ds_round_simd            // -- Begin function fcvtzu_ds_round_simd
	.p2align	2
	.type	fcvtzu_ds_round_simd,@function
fcvtzu_ds_round_simd:                   // @fcvtzu_ds_round_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	d0, s0
	ret
.Lfunc_end52:
	.size	fcvtzu_ds_round_simd, .Lfunc_end52-fcvtzu_ds_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_sd_round_simd            // -- Begin function fcvtzu_sd_round_simd
	.p2align	2
	.type	fcvtzu_sd_round_simd,@function
fcvtzu_sd_round_simd:                   // @fcvtzu_sd_round_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	s0, d0
	ret
.Lfunc_end53:
	.size	fcvtzu_sd_round_simd, .Lfunc_end53-fcvtzu_sd_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_ss_round_simd            // -- Begin function fcvtzu_ss_round_simd
	.p2align	2
	.type	fcvtzu_ss_round_simd,@function
fcvtzu_ss_round_simd:                   // @fcvtzu_ss_round_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	s0, s0
	ret
.Lfunc_end54:
	.size	fcvtzu_ss_round_simd, .Lfunc_end54-fcvtzu_ss_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_dd_round_simd            // -- Begin function fcvtzu_dd_round_simd
	.p2align	2
	.type	fcvtzu_dd_round_simd,@function
fcvtzu_dd_round_simd:                   // @fcvtzu_dd_round_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, d0
	ret
.Lfunc_end55:
	.size	fcvtzu_dd_round_simd, .Lfunc_end55-fcvtzu_dd_round_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_sh_sat_simd              // -- Begin function fcvtzs_sh_sat_simd
	.p2align	2
	.type	fcvtzs_sh_sat_simd,@function
fcvtzs_sh_sat_simd:                     // @fcvtzs_sh_sat_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	s0, h0
	ret
.Lfunc_end56:
	.size	fcvtzs_sh_sat_simd, .Lfunc_end56-fcvtzs_sh_sat_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_dh_sat_simd              // -- Begin function fcvtzs_dh_sat_simd
	.p2align	2
	.type	fcvtzs_dh_sat_simd,@function
fcvtzs_dh_sat_simd:                     // @fcvtzs_dh_sat_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, h0
	ret
.Lfunc_end57:
	.size	fcvtzs_dh_sat_simd, .Lfunc_end57-fcvtzs_dh_sat_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_ds_sat_simd              // -- Begin function fcvtzs_ds_sat_simd
	.p2align	2
	.type	fcvtzs_ds_sat_simd,@function
fcvtzs_ds_sat_simd:                     // @fcvtzs_ds_sat_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, s0
	ret
.Lfunc_end58:
	.size	fcvtzs_ds_sat_simd, .Lfunc_end58-fcvtzs_ds_sat_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_sd_sat_simd              // -- Begin function fcvtzs_sd_sat_simd
	.p2align	2
	.type	fcvtzs_sd_sat_simd,@function
fcvtzs_sd_sat_simd:                     // @fcvtzs_sd_sat_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	s0, d0
	ret
.Lfunc_end59:
	.size	fcvtzs_sd_sat_simd, .Lfunc_end59-fcvtzs_sd_sat_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_ss_sat_simd              // -- Begin function fcvtzs_ss_sat_simd
	.p2align	2
	.type	fcvtzs_ss_sat_simd,@function
fcvtzs_ss_sat_simd:                     // @fcvtzs_ss_sat_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	s0, s0
	ret
.Lfunc_end60:
	.size	fcvtzs_ss_sat_simd, .Lfunc_end60-fcvtzs_ss_sat_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_dd_sat_simd              // -- Begin function fcvtzs_dd_sat_simd
	.p2align	2
	.type	fcvtzs_dd_sat_simd,@function
fcvtzs_dd_sat_simd:                     // @fcvtzs_dd_sat_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, d0
	ret
.Lfunc_end61:
	.size	fcvtzs_dd_sat_simd, .Lfunc_end61-fcvtzs_dd_sat_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_sh_sat_simd              // -- Begin function fcvtzu_sh_sat_simd
	.p2align	2
	.type	fcvtzu_sh_sat_simd,@function
fcvtzu_sh_sat_simd:                     // @fcvtzu_sh_sat_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	s0, h0
	ret
.Lfunc_end62:
	.size	fcvtzu_sh_sat_simd, .Lfunc_end62-fcvtzu_sh_sat_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_dh_sat_simd              // -- Begin function fcvtzu_dh_sat_simd
	.p2align	2
	.type	fcvtzu_dh_sat_simd,@function
fcvtzu_dh_sat_simd:                     // @fcvtzu_dh_sat_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	d0, h0
	ret
.Lfunc_end63:
	.size	fcvtzu_dh_sat_simd, .Lfunc_end63-fcvtzu_dh_sat_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_ds_sat_simd              // -- Begin function fcvtzu_ds_sat_simd
	.p2align	2
	.type	fcvtzu_ds_sat_simd,@function
fcvtzu_ds_sat_simd:                     // @fcvtzu_ds_sat_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	d0, s0
	ret
.Lfunc_end64:
	.size	fcvtzu_ds_sat_simd, .Lfunc_end64-fcvtzu_ds_sat_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_sd_sat_simd              // -- Begin function fcvtzu_sd_sat_simd
	.p2align	2
	.type	fcvtzu_sd_sat_simd,@function
fcvtzu_sd_sat_simd:                     // @fcvtzu_sd_sat_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	s0, d0
	ret
.Lfunc_end65:
	.size	fcvtzu_sd_sat_simd, .Lfunc_end65-fcvtzu_sd_sat_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_ss_sat_simd              // -- Begin function fcvtzu_ss_sat_simd
	.p2align	2
	.type	fcvtzu_ss_sat_simd,@function
fcvtzu_ss_sat_simd:                     // @fcvtzu_ss_sat_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	s0, s0
	ret
.Lfunc_end66:
	.size	fcvtzu_ss_sat_simd, .Lfunc_end66-fcvtzu_ss_sat_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_dd_sat_simd              // -- Begin function fcvtzu_dd_sat_simd
	.p2align	2
	.type	fcvtzu_dd_sat_simd,@function
fcvtzu_dd_sat_simd:                     // @fcvtzu_dd_sat_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, d0
	ret
.Lfunc_end67:
	.size	fcvtzu_dd_sat_simd, .Lfunc_end67-fcvtzu_dd_sat_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtas_sh_simd                  // -- Begin function fcvtas_sh_simd
	.p2align	2
	.type	fcvtas_sh_simd,@function
fcvtas_sh_simd:                         // @fcvtas_sh_simd
	.cfi_startproc
// %bb.0:
	fcvtas	s0, h0
	ret
.Lfunc_end68:
	.size	fcvtas_sh_simd, .Lfunc_end68-fcvtas_sh_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtas_dh_simd                  // -- Begin function fcvtas_dh_simd
	.p2align	2
	.type	fcvtas_dh_simd,@function
fcvtas_dh_simd:                         // @fcvtas_dh_simd
	.cfi_startproc
// %bb.0:
	fcvtas	d0, h0
	ret
.Lfunc_end69:
	.size	fcvtas_dh_simd, .Lfunc_end69-fcvtas_dh_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtas_ds_simd                  // -- Begin function fcvtas_ds_simd
	.p2align	2
	.type	fcvtas_ds_simd,@function
fcvtas_ds_simd:                         // @fcvtas_ds_simd
	.cfi_startproc
// %bb.0:
	fcvtas	d0, s0
	ret
.Lfunc_end70:
	.size	fcvtas_ds_simd, .Lfunc_end70-fcvtas_ds_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtas_sd_simd                  // -- Begin function fcvtas_sd_simd
	.p2align	2
	.type	fcvtas_sd_simd,@function
fcvtas_sd_simd:                         // @fcvtas_sd_simd
	.cfi_startproc
// %bb.0:
	fcvtas	s0, d0
	ret
.Lfunc_end71:
	.size	fcvtas_sd_simd, .Lfunc_end71-fcvtas_sd_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtas_ss_simd                  // -- Begin function fcvtas_ss_simd
	.p2align	2
	.type	fcvtas_ss_simd,@function
fcvtas_ss_simd:                         // @fcvtas_ss_simd
	.cfi_startproc
// %bb.0:
	fcvtas	s0, s0
	ret
.Lfunc_end72:
	.size	fcvtas_ss_simd, .Lfunc_end72-fcvtas_ss_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtas_dd_simd                  // -- Begin function fcvtas_dd_simd
	.p2align	2
	.type	fcvtas_dd_simd,@function
fcvtas_dd_simd:                         // @fcvtas_dd_simd
	.cfi_startproc
// %bb.0:
	fcvtas	d0, d0
	ret
.Lfunc_end73:
	.size	fcvtas_dd_simd, .Lfunc_end73-fcvtas_dd_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtau_sh_simd                  // -- Begin function fcvtau_sh_simd
	.p2align	2
	.type	fcvtau_sh_simd,@function
fcvtau_sh_simd:                         // @fcvtau_sh_simd
	.cfi_startproc
// %bb.0:
	fcvtau	s0, h0
	ret
.Lfunc_end74:
	.size	fcvtau_sh_simd, .Lfunc_end74-fcvtau_sh_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtau_dh_simd                  // -- Begin function fcvtau_dh_simd
	.p2align	2
	.type	fcvtau_dh_simd,@function
fcvtau_dh_simd:                         // @fcvtau_dh_simd
	.cfi_startproc
// %bb.0:
	fcvtau	d0, h0
	ret
.Lfunc_end75:
	.size	fcvtau_dh_simd, .Lfunc_end75-fcvtau_dh_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtau_ds_simd                  // -- Begin function fcvtau_ds_simd
	.p2align	2
	.type	fcvtau_ds_simd,@function
fcvtau_ds_simd:                         // @fcvtau_ds_simd
	.cfi_startproc
// %bb.0:
	fcvtau	d0, s0
	ret
.Lfunc_end76:
	.size	fcvtau_ds_simd, .Lfunc_end76-fcvtau_ds_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtau_sd_simd                  // -- Begin function fcvtau_sd_simd
	.p2align	2
	.type	fcvtau_sd_simd,@function
fcvtau_sd_simd:                         // @fcvtau_sd_simd
	.cfi_startproc
// %bb.0:
	fcvtau	s0, d0
	ret
.Lfunc_end77:
	.size	fcvtau_sd_simd, .Lfunc_end77-fcvtau_sd_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtau_ss_simd                  // -- Begin function fcvtau_ss_simd
	.p2align	2
	.type	fcvtau_ss_simd,@function
fcvtau_ss_simd:                         // @fcvtau_ss_simd
	.cfi_startproc
// %bb.0:
	fcvtas	s0, s0
	ret
.Lfunc_end78:
	.size	fcvtau_ss_simd, .Lfunc_end78-fcvtau_ss_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtau_dd_simd                  // -- Begin function fcvtau_dd_simd
	.p2align	2
	.type	fcvtau_dd_simd,@function
fcvtau_dd_simd:                         // @fcvtau_dd_simd
	.cfi_startproc
// %bb.0:
	fcvtas	d0, d0
	ret
.Lfunc_end79:
	.size	fcvtau_dd_simd, .Lfunc_end79-fcvtau_dd_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtms_sh_simd                  // -- Begin function fcvtms_sh_simd
	.p2align	2
	.type	fcvtms_sh_simd,@function
fcvtms_sh_simd:                         // @fcvtms_sh_simd
	.cfi_startproc
// %bb.0:
	fcvtms	s0, h0
	ret
.Lfunc_end80:
	.size	fcvtms_sh_simd, .Lfunc_end80-fcvtms_sh_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtms_dh_simd                  // -- Begin function fcvtms_dh_simd
	.p2align	2
	.type	fcvtms_dh_simd,@function
fcvtms_dh_simd:                         // @fcvtms_dh_simd
	.cfi_startproc
// %bb.0:
	fcvtms	d0, h0
	ret
.Lfunc_end81:
	.size	fcvtms_dh_simd, .Lfunc_end81-fcvtms_dh_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtms_ds_simd                  // -- Begin function fcvtms_ds_simd
	.p2align	2
	.type	fcvtms_ds_simd,@function
fcvtms_ds_simd:                         // @fcvtms_ds_simd
	.cfi_startproc
// %bb.0:
	fcvtms	d0, s0
	ret
.Lfunc_end82:
	.size	fcvtms_ds_simd, .Lfunc_end82-fcvtms_ds_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtms_sd_simd                  // -- Begin function fcvtms_sd_simd
	.p2align	2
	.type	fcvtms_sd_simd,@function
fcvtms_sd_simd:                         // @fcvtms_sd_simd
	.cfi_startproc
// %bb.0:
	fcvtms	s0, d0
	ret
.Lfunc_end83:
	.size	fcvtms_sd_simd, .Lfunc_end83-fcvtms_sd_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtms_ss_simd                  // -- Begin function fcvtms_ss_simd
	.p2align	2
	.type	fcvtms_ss_simd,@function
fcvtms_ss_simd:                         // @fcvtms_ss_simd
	.cfi_startproc
// %bb.0:
	fcvtms	s0, s0
	ret
.Lfunc_end84:
	.size	fcvtms_ss_simd, .Lfunc_end84-fcvtms_ss_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtms_dd_simd                  // -- Begin function fcvtms_dd_simd
	.p2align	2
	.type	fcvtms_dd_simd,@function
fcvtms_dd_simd:                         // @fcvtms_dd_simd
	.cfi_startproc
// %bb.0:
	fcvtms	d0, d0
	ret
.Lfunc_end85:
	.size	fcvtms_dd_simd, .Lfunc_end85-fcvtms_dd_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtmu_sh_simd                  // -- Begin function fcvtmu_sh_simd
	.p2align	2
	.type	fcvtmu_sh_simd,@function
fcvtmu_sh_simd:                         // @fcvtmu_sh_simd
	.cfi_startproc
// %bb.0:
	fcvtmu	s0, h0
	ret
.Lfunc_end86:
	.size	fcvtmu_sh_simd, .Lfunc_end86-fcvtmu_sh_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtmu_dh_simd                  // -- Begin function fcvtmu_dh_simd
	.p2align	2
	.type	fcvtmu_dh_simd,@function
fcvtmu_dh_simd:                         // @fcvtmu_dh_simd
	.cfi_startproc
// %bb.0:
	fcvtmu	d0, h0
	ret
.Lfunc_end87:
	.size	fcvtmu_dh_simd, .Lfunc_end87-fcvtmu_dh_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtmu_ds_simd                  // -- Begin function fcvtmu_ds_simd
	.p2align	2
	.type	fcvtmu_ds_simd,@function
fcvtmu_ds_simd:                         // @fcvtmu_ds_simd
	.cfi_startproc
// %bb.0:
	fcvtmu	d0, s0
	ret
.Lfunc_end88:
	.size	fcvtmu_ds_simd, .Lfunc_end88-fcvtmu_ds_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtmu_sd_simd                  // -- Begin function fcvtmu_sd_simd
	.p2align	2
	.type	fcvtmu_sd_simd,@function
fcvtmu_sd_simd:                         // @fcvtmu_sd_simd
	.cfi_startproc
// %bb.0:
	fcvtmu	s0, d0
	ret
.Lfunc_end89:
	.size	fcvtmu_sd_simd, .Lfunc_end89-fcvtmu_sd_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtmu_ss_simd                  // -- Begin function fcvtmu_ss_simd
	.p2align	2
	.type	fcvtmu_ss_simd,@function
fcvtmu_ss_simd:                         // @fcvtmu_ss_simd
	.cfi_startproc
// %bb.0:
	fcvtms	s0, s0
	ret
.Lfunc_end90:
	.size	fcvtmu_ss_simd, .Lfunc_end90-fcvtmu_ss_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtmu_dd_simd                  // -- Begin function fcvtmu_dd_simd
	.p2align	2
	.type	fcvtmu_dd_simd,@function
fcvtmu_dd_simd:                         // @fcvtmu_dd_simd
	.cfi_startproc
// %bb.0:
	fcvtms	d0, d0
	ret
.Lfunc_end91:
	.size	fcvtmu_dd_simd, .Lfunc_end91-fcvtmu_dd_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtps_sh_simd                  // -- Begin function fcvtps_sh_simd
	.p2align	2
	.type	fcvtps_sh_simd,@function
fcvtps_sh_simd:                         // @fcvtps_sh_simd
	.cfi_startproc
// %bb.0:
	fcvtps	s0, h0
	ret
.Lfunc_end92:
	.size	fcvtps_sh_simd, .Lfunc_end92-fcvtps_sh_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtps_dh_simd                  // -- Begin function fcvtps_dh_simd
	.p2align	2
	.type	fcvtps_dh_simd,@function
fcvtps_dh_simd:                         // @fcvtps_dh_simd
	.cfi_startproc
// %bb.0:
	fcvtps	d0, h0
	ret
.Lfunc_end93:
	.size	fcvtps_dh_simd, .Lfunc_end93-fcvtps_dh_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtps_ds_simd                  // -- Begin function fcvtps_ds_simd
	.p2align	2
	.type	fcvtps_ds_simd,@function
fcvtps_ds_simd:                         // @fcvtps_ds_simd
	.cfi_startproc
// %bb.0:
	fcvtps	d0, s0
	ret
.Lfunc_end94:
	.size	fcvtps_ds_simd, .Lfunc_end94-fcvtps_ds_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtps_sd_simd                  // -- Begin function fcvtps_sd_simd
	.p2align	2
	.type	fcvtps_sd_simd,@function
fcvtps_sd_simd:                         // @fcvtps_sd_simd
	.cfi_startproc
// %bb.0:
	fcvtps	s0, d0
	ret
.Lfunc_end95:
	.size	fcvtps_sd_simd, .Lfunc_end95-fcvtps_sd_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtps_ss_simd                  // -- Begin function fcvtps_ss_simd
	.p2align	2
	.type	fcvtps_ss_simd,@function
fcvtps_ss_simd:                         // @fcvtps_ss_simd
	.cfi_startproc
// %bb.0:
	fcvtps	s0, s0
	ret
.Lfunc_end96:
	.size	fcvtps_ss_simd, .Lfunc_end96-fcvtps_ss_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtps_dd_simd                  // -- Begin function fcvtps_dd_simd
	.p2align	2
	.type	fcvtps_dd_simd,@function
fcvtps_dd_simd:                         // @fcvtps_dd_simd
	.cfi_startproc
// %bb.0:
	fcvtps	d0, d0
	ret
.Lfunc_end97:
	.size	fcvtps_dd_simd, .Lfunc_end97-fcvtps_dd_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtpu_sh_simd                  // -- Begin function fcvtpu_sh_simd
	.p2align	2
	.type	fcvtpu_sh_simd,@function
fcvtpu_sh_simd:                         // @fcvtpu_sh_simd
	.cfi_startproc
// %bb.0:
	fcvtpu	s0, h0
	ret
.Lfunc_end98:
	.size	fcvtpu_sh_simd, .Lfunc_end98-fcvtpu_sh_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtpu_dh_simd                  // -- Begin function fcvtpu_dh_simd
	.p2align	2
	.type	fcvtpu_dh_simd,@function
fcvtpu_dh_simd:                         // @fcvtpu_dh_simd
	.cfi_startproc
// %bb.0:
	fcvtpu	d0, h0
	ret
.Lfunc_end99:
	.size	fcvtpu_dh_simd, .Lfunc_end99-fcvtpu_dh_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtpu_ds_simd                  // -- Begin function fcvtpu_ds_simd
	.p2align	2
	.type	fcvtpu_ds_simd,@function
fcvtpu_ds_simd:                         // @fcvtpu_ds_simd
	.cfi_startproc
// %bb.0:
	fcvtpu	d0, s0
	ret
.Lfunc_end100:
	.size	fcvtpu_ds_simd, .Lfunc_end100-fcvtpu_ds_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtpu_sd_simd                  // -- Begin function fcvtpu_sd_simd
	.p2align	2
	.type	fcvtpu_sd_simd,@function
fcvtpu_sd_simd:                         // @fcvtpu_sd_simd
	.cfi_startproc
// %bb.0:
	fcvtpu	s0, d0
	ret
.Lfunc_end101:
	.size	fcvtpu_sd_simd, .Lfunc_end101-fcvtpu_sd_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtpu_ss_simd                  // -- Begin function fcvtpu_ss_simd
	.p2align	2
	.type	fcvtpu_ss_simd,@function
fcvtpu_ss_simd:                         // @fcvtpu_ss_simd
	.cfi_startproc
// %bb.0:
	fcvtps	s0, s0
	ret
.Lfunc_end102:
	.size	fcvtpu_ss_simd, .Lfunc_end102-fcvtpu_ss_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtpu_dd_simd                  // -- Begin function fcvtpu_dd_simd
	.p2align	2
	.type	fcvtpu_dd_simd,@function
fcvtpu_dd_simd:                         // @fcvtpu_dd_simd
	.cfi_startproc
// %bb.0:
	fcvtps	d0, d0
	ret
.Lfunc_end103:
	.size	fcvtpu_dd_simd, .Lfunc_end103-fcvtpu_dd_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_sh_simd                  // -- Begin function fcvtzs_sh_simd
	.p2align	2
	.type	fcvtzs_sh_simd,@function
fcvtzs_sh_simd:                         // @fcvtzs_sh_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	s0, h0
	ret
.Lfunc_end104:
	.size	fcvtzs_sh_simd, .Lfunc_end104-fcvtzs_sh_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_dh_simd                  // -- Begin function fcvtzs_dh_simd
	.p2align	2
	.type	fcvtzs_dh_simd,@function
fcvtzs_dh_simd:                         // @fcvtzs_dh_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, h0
	ret
.Lfunc_end105:
	.size	fcvtzs_dh_simd, .Lfunc_end105-fcvtzs_dh_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_ds_simd                  // -- Begin function fcvtzs_ds_simd
	.p2align	2
	.type	fcvtzs_ds_simd,@function
fcvtzs_ds_simd:                         // @fcvtzs_ds_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, s0
	ret
.Lfunc_end106:
	.size	fcvtzs_ds_simd, .Lfunc_end106-fcvtzs_ds_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_sd_simd                  // -- Begin function fcvtzs_sd_simd
	.p2align	2
	.type	fcvtzs_sd_simd,@function
fcvtzs_sd_simd:                         // @fcvtzs_sd_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	s0, d0
	ret
.Lfunc_end107:
	.size	fcvtzs_sd_simd, .Lfunc_end107-fcvtzs_sd_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_ss_simd                  // -- Begin function fcvtzs_ss_simd
	.p2align	2
	.type	fcvtzs_ss_simd,@function
fcvtzs_ss_simd:                         // @fcvtzs_ss_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	s0, s0
	ret
.Lfunc_end108:
	.size	fcvtzs_ss_simd, .Lfunc_end108-fcvtzs_ss_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_dd_simd                  // -- Begin function fcvtzs_dd_simd
	.p2align	2
	.type	fcvtzs_dd_simd,@function
fcvtzs_dd_simd:                         // @fcvtzs_dd_simd
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, d0
	ret
.Lfunc_end109:
	.size	fcvtzs_dd_simd, .Lfunc_end109-fcvtzs_dd_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_sh_simd                  // -- Begin function fcvtzu_sh_simd
	.p2align	2
	.type	fcvtzu_sh_simd,@function
fcvtzu_sh_simd:                         // @fcvtzu_sh_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	s0, h0
	ret
.Lfunc_end110:
	.size	fcvtzu_sh_simd, .Lfunc_end110-fcvtzu_sh_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_dh_simd                  // -- Begin function fcvtzu_dh_simd
	.p2align	2
	.type	fcvtzu_dh_simd,@function
fcvtzu_dh_simd:                         // @fcvtzu_dh_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	d0, h0
	ret
.Lfunc_end111:
	.size	fcvtzu_dh_simd, .Lfunc_end111-fcvtzu_dh_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_ds_simd                  // -- Begin function fcvtzu_ds_simd
	.p2align	2
	.type	fcvtzu_ds_simd,@function
fcvtzu_ds_simd:                         // @fcvtzu_ds_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	d0, s0
	ret
.Lfunc_end112:
	.size	fcvtzu_ds_simd, .Lfunc_end112-fcvtzu_ds_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_sd_simd                  // -- Begin function fcvtzu_sd_simd
	.p2align	2
	.type	fcvtzu_sd_simd,@function
fcvtzu_sd_simd:                         // @fcvtzu_sd_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	s0, d0
	ret
.Lfunc_end113:
	.size	fcvtzu_sd_simd, .Lfunc_end113-fcvtzu_sd_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_ss_simd                  // -- Begin function fcvtzu_ss_simd
	.p2align	2
	.type	fcvtzu_ss_simd,@function
fcvtzu_ss_simd:                         // @fcvtzu_ss_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	s0, s0
	ret
.Lfunc_end114:
	.size	fcvtzu_ss_simd, .Lfunc_end114-fcvtzu_ss_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_dd_simd                  // -- Begin function fcvtzu_dd_simd
	.p2align	2
	.type	fcvtzu_dd_simd,@function
fcvtzu_dd_simd:                         // @fcvtzu_dd_simd
	.cfi_startproc
// %bb.0:
	fcvtzu	d0, d0
	ret
.Lfunc_end115:
	.size	fcvtzu_dd_simd, .Lfunc_end115-fcvtzu_dd_simd
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_scalar_to_vector_h       // -- Begin function fcvtzs_scalar_to_vector_h
	.p2align	2
	.type	fcvtzs_scalar_to_vector_h,@function
fcvtzs_scalar_to_vector_h:              // @fcvtzs_scalar_to_vector_h
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, h0
	ret
.Lfunc_end116:
	.size	fcvtzs_scalar_to_vector_h, .Lfunc_end116-fcvtzs_scalar_to_vector_h
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_scalar_to_vector_s       // -- Begin function fcvtzs_scalar_to_vector_s
	.p2align	2
	.type	fcvtzs_scalar_to_vector_s,@function
fcvtzs_scalar_to_vector_s:              // @fcvtzs_scalar_to_vector_s
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, s0
	ret
.Lfunc_end117:
	.size	fcvtzs_scalar_to_vector_s, .Lfunc_end117-fcvtzs_scalar_to_vector_s
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_scalar_to_vector_d       // -- Begin function fcvtzs_scalar_to_vector_d
	.p2align	2
	.type	fcvtzs_scalar_to_vector_d,@function
fcvtzs_scalar_to_vector_d:              // @fcvtzs_scalar_to_vector_d
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, d0
	ret
.Lfunc_end118:
	.size	fcvtzs_scalar_to_vector_d, .Lfunc_end118-fcvtzs_scalar_to_vector_d
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_scalar_to_vector_h       // -- Begin function fcvtzu_scalar_to_vector_h
	.p2align	2
	.type	fcvtzu_scalar_to_vector_h,@function
fcvtzu_scalar_to_vector_h:              // @fcvtzu_scalar_to_vector_h
	.cfi_startproc
// %bb.0:
	fcvtzu	d0, h0
	ret
.Lfunc_end119:
	.size	fcvtzu_scalar_to_vector_h, .Lfunc_end119-fcvtzu_scalar_to_vector_h
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_scalar_to_vector_s       // -- Begin function fcvtzu_scalar_to_vector_s
	.p2align	2
	.type	fcvtzu_scalar_to_vector_s,@function
fcvtzu_scalar_to_vector_s:              // @fcvtzu_scalar_to_vector_s
	.cfi_startproc
// %bb.0:
	fcvtzu	d0, s0
	ret
.Lfunc_end120:
	.size	fcvtzu_scalar_to_vector_s, .Lfunc_end120-fcvtzu_scalar_to_vector_s
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_scalar_to_vector_d       // -- Begin function fcvtzu_scalar_to_vector_d
	.p2align	2
	.type	fcvtzu_scalar_to_vector_d,@function
fcvtzu_scalar_to_vector_d:              // @fcvtzu_scalar_to_vector_d
	.cfi_startproc
// %bb.0:
	fcvtzu	d0, d0
	ret
.Lfunc_end121:
	.size	fcvtzu_scalar_to_vector_d, .Lfunc_end121-fcvtzu_scalar_to_vector_d
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_scalar_to_vector_h_strict // -- Begin function fcvtzs_scalar_to_vector_h_strict
	.p2align	2
	.type	fcvtzs_scalar_to_vector_h_strict,@function
fcvtzs_scalar_to_vector_h_strict:       // @fcvtzs_scalar_to_vector_h_strict
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, h0
	ret
.Lfunc_end122:
	.size	fcvtzs_scalar_to_vector_h_strict, .Lfunc_end122-fcvtzs_scalar_to_vector_h_strict
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzs_scalar_to_vector_s_strict // -- Begin function fcvtzs_scalar_to_vector_s_strict
	.p2align	2
	.type	fcvtzs_scalar_to_vector_s_strict,@function
fcvtzs_scalar_to_vector_s_strict:       // @fcvtzs_scalar_to_vector_s_strict
	.cfi_startproc
// %bb.0:
	fcvtzs	d0, s0
	ret
.Lfunc_end123:
	.size	fcvtzs_scalar_to_vector_s_strict, .Lfunc_end123-fcvtzs_scalar_to_vector_s_strict
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_scalar_to_vector_h_strict // -- Begin function fcvtzu_scalar_to_vector_h_strict
	.p2align	2
	.type	fcvtzu_scalar_to_vector_h_strict,@function
fcvtzu_scalar_to_vector_h_strict:       // @fcvtzu_scalar_to_vector_h_strict
	.cfi_startproc
// %bb.0:
	fcvtzu	d0, h0
	ret
.Lfunc_end124:
	.size	fcvtzu_scalar_to_vector_h_strict, .Lfunc_end124-fcvtzu_scalar_to_vector_h_strict
	.cfi_endproc
                                        // -- End function
	.globl	fcvtzu_scalar_to_vector_s_strict // -- Begin function fcvtzu_scalar_to_vector_s_strict
	.p2align	2
	.type	fcvtzu_scalar_to_vector_s_strict,@function
fcvtzu_scalar_to_vector_s_strict:       // @fcvtzu_scalar_to_vector_s_strict
	.cfi_startproc
// %bb.0:
	fcvtzu	d0, s0
	ret
.Lfunc_end125:
	.size	fcvtzu_scalar_to_vector_s_strict, .Lfunc_end125-fcvtzu_scalar_to_vector_s_strict
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
