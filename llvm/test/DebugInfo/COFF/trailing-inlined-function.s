# REQUIRES: x86-registered-target
# RUN: llvm-mc -filetype=obj --triple=x86_64-pc-windows-msvc %s | llvm-readobj - --codeview --codeview-subsection-bytes | FileCheck %s

# Rust source to regenerate:
# #[no_mangle]
# extern "C" fn add_numbers(x: &Option<i32>, y: &Option<i32>) -> i32 {
#     let x1 = x.unwrap();
#     let y1 = y.unwrap();
#     x1 + y1
# }
# $ rustc trailing-inlined-function.rs --crate-type cdylib --emit=asm -Copt-level=3 -Cpanic=abort -Cdebuginfo=1

# Validate that unwrap() was inlined.
# CHECK:       InlineSiteSym {
# CHECK-NEXT:    Kind: S_INLINESITE (0x114D)
# CHECK-NEXT:    PtrParent: 0x0
# CHECK-NEXT:    PtrEnd: 0x0
# CHECK-NEXT:    Inlinee: unwrap
# CHECK-NEXT:    BinaryAnnotations [
# CHECK-NEXT:      ChangeCodeOffsetAndLineOffset: {CodeOffset: [[#%#x,Offset1_1:]], LineOffset: 1}
# CHECK-NEXT:      ChangeCodeLength: [[#%#x,Length1_1:]]
# CHECK-NEXT:      ChangeLineOffset: 2
# CHECK-NEXT:      ChangeCodeOffset: [[#%#x,Offset1_2:]]
# CHECK-NEXT:      ChangeCodeLength: [[#%#x,]]
# CHECK-NEXT:      (Annotation Padding)
# CHECK:      InlineSiteSym {
# CHECK-NEXT:    Kind: S_INLINESITE (0x114D)
# CHECK-NEXT:    PtrParent: 0x0
# CHECK-NEXT:    PtrEnd: 0x0
# CHECK-NEXT:    Inlinee: unwrap
# CHECK-NEXT:    BinaryAnnotations [
# CHECK-NEXT:      ChangeCodeOffsetAndLineOffset: {CodeOffset: [[#%#x,Offset2_1:]], LineOffset: 1}
# CHECK-NEXT:      ChangeCodeLength: [[#%#x,Length2_1:]]
# CHECK-NEXT:      ChangeLineOffset: 2
# CHECK-NEXT:      ChangeCodeOffset: [[#%#x,Offset2_2:]]
# CHECK-NEXT:      ChangeCodeLength: [[#%#x,]]
# CHECK-NEXT:      (Annotation Padding)

# Validate that basic blocks from an inlined function that are sunk below the rest of the function
# (namely bb1 and bb4 in this test) get the correct debug info.
# CHECK:       SubSectionType: Lines (0xF2)
# CHECK-NEXT:   SubSectionSize: [[#%#x,]]
# CHECK-NEXT:   SubSectionContents (
# CHECK-NEXT:     0000: [[#%.8X,]] [[#%.8X,]] [[#%.8X,]] [[#%.8X,]]
#                       Code starts at line 2
# CHECK-NEXT:     0010: [[#%.8X,]] [[#%.8X,]] [[#%.8X,]] 02000000
#                       The success paths for unwrap() (lines 3 & 4) are next.
# CHECK-NEXT:     0020: [[#%.2X,Offset1_1]]000000 03000000 [[#%.2X,Offset2_1]]000000 04000000
#                       Then the addition (line 5) and the end of the function (end-brace on line 6).
# CHECK-NEXT:     0030: [[#%.8X,]] 05000000 [[#%.8X,]] 06000000
#                       The failure paths for unwrap() (lines 3 & 4) are placed after the `ret` instruction.
# CHECK-NEXT:     0040: [[#%.2X,Offset1_1 + Length1_1 + Offset1_2]]000000 03000000 [[#%.2X,Offset2_1 + Length2_1 + Offset2_2]]000000 04000000
# CHECK-NOT:    SubSectionType: Lines (0xF2)

	.text
	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.file	"trailing_inlined_function.3a6e73a087a7434a-cgu.0"
	.def	add_numbers;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",one_only,add_numbers
	.globl	add_numbers
	.p2align	4, 0x90
add_numbers:
.Lfunc_begin0:
	.cv_func_id 0
	.cv_file	1 "C:\\llvm\\trailing-inlined-function.rs" "A63E3A719BDF505386FDB73BF86EC58591BDAC588181F0E423E724AEEC3E4852" 3
	.cv_loc	0 1 2 0
.seh_proc add_numbers
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
.Ltmp0:
	.cv_file	2 "/rustc/bc28abf92efc32f8f9312851bf8af38fbd23be42\\library\\core\\src\\option.rs" "7B702FA8D5AAEDC0CCA1EE32F30D5922BC11516B54D592279493A30457F918D9" 3
	.cv_inline_site_id 1 within 0 inlined_at 1 3 0
	.cv_loc	1 2 933 0
	cmpl	$0, (%rcx)
	je	.LBB0_1
.Ltmp1:
	.cv_inline_site_id 2 within 0 inlined_at 1 4 0
	.cv_loc	2 2 933 0
	cmpl	$0, (%rdx)
	je	.LBB0_4
.Ltmp2:
	.cv_loc	0 1 5 0
	movl	4(%rcx), %eax
.Ltmp3:
	addl	4(%rdx), %eax
.Ltmp4:
	.cv_loc	0 1 6 0
	.seh_startepilogue
	addq	$40, %rsp
	.seh_endepilogue
	retq
.LBB0_1:
.Ltmp5:
	.cv_loc	1 2 935 0
	leaq	__unnamed_1(%rip), %rcx
	leaq	__unnamed_2(%rip), %r8
.Ltmp6:
	movl	$43, %edx
	callq	_ZN4core9panicking5panic17hd083df7b722701afE
	ud2
.LBB0_4:
.Ltmp7:
	.cv_loc	2 2 935 0
	leaq	__unnamed_1(%rip), %rcx
	leaq	__unnamed_3(%rip), %r8
.Ltmp8:
	movl	$43, %edx
	callq	_ZN4core9panicking5panic17hd083df7b722701afE
	ud2
.Ltmp9:
.Lfunc_end0:
	.seh_endproc

	.section	.rdata,"dr",one_only,__unnamed_1
__unnamed_1:
	.ascii	"called `Option::unwrap()` on a `None` value"

	.section	.rdata,"dr",one_only,__unnamed_4
__unnamed_4:
	.ascii	"trailing-inlined-function.rs"

	.section	.rdata,"dr",one_only,__unnamed_2
	.p2align	3, 0x0
__unnamed_2:
	.quad	__unnamed_4
	.asciz	"\034\000\000\000\000\000\000\000\003\000\000\000\020\000\000"

	.section	.rdata,"dr",one_only,__unnamed_3
	.p2align	3, 0x0
__unnamed_3:
	.quad	__unnamed_4
	.asciz	"\034\000\000\000\000\000\000\000\004\000\000\000\020\000\000"

	.section	.debug$S,"dr"
	.p2align	2, 0x0
	.long	4
	.long	241
	.long	.Ltmp11-.Ltmp10
.Ltmp10:
	.short	.Ltmp13-.Ltmp12
.Ltmp12:
	.short	4353
	.long	0
	.byte	0
	.p2align	2, 0x0
.Ltmp13:
	.short	.Ltmp15-.Ltmp14
.Ltmp14:
	.short	4412
	.long	21
	.short	208
	.short	1
	.short	73
	.short	0
	.short	0
	.short	17000
	.short	0
	.short	0
	.short	0
	.asciz	"clang LLVM (rustc version 1.73.0-beta.3 (bc28abf92 2023-08-27))"
	.p2align	2, 0x0
.Ltmp15:
.Ltmp11:
	.p2align	2, 0x0
	.long	246
	.long	.Ltmp17-.Ltmp16
.Ltmp16:
	.long	0


	.long	4099
	.cv_filechecksumoffset	2
	.long	932


	.long	4099
	.cv_filechecksumoffset	2
	.long	932
.Ltmp17:
	.p2align	2, 0x0
	.section	.debug$S,"dr",associative,add_numbers
	.p2align	2, 0x0
	.long	4
	.long	241
	.long	.Ltmp19-.Ltmp18
.Ltmp18:
	.short	.Ltmp21-.Ltmp20
.Ltmp20:
	.short	4423
	.long	0
	.long	0
	.long	0
	.long	.Lfunc_end0-add_numbers
	.long	0
	.long	0
	.long	4101
	.secrel32	add_numbers
	.secidx	add_numbers
	.byte	128
	.asciz	"trailing_inlined_function::add_numbers"
	.p2align	2, 0x0
.Ltmp21:
	.short	.Ltmp23-.Ltmp22
.Ltmp22:
	.short	4114
	.long	40
	.long	0
	.long	0
	.long	0
	.long	0
	.short	0
	.long	1138688
	.p2align	2, 0x0
.Ltmp23:
	.short	.Ltmp25-.Ltmp24
.Ltmp24:
	.short	4429
	.long	0
	.long	0
	.long	4099
	.cv_inline_linetable	1 2 932 .Lfunc_begin0 .Lfunc_end0
	.p2align	2, 0x0
.Ltmp25:
	.short	2
	.short	4430
	.short	.Ltmp27-.Ltmp26
.Ltmp26:
	.short	4429
	.long	0
	.long	0
	.long	4099
	.cv_inline_linetable	2 2 932 .Lfunc_begin0 .Lfunc_end0
	.p2align	2, 0x0
.Ltmp27:
	.short	2
	.short	4430
	.short	2
	.short	4431
.Ltmp19:
	.p2align	2, 0x0
	.cv_linetable	0, add_numbers, .Lfunc_end0
	.section	.debug$S,"dr"
	.cv_filechecksums
	.cv_stringtable
	.long	241
	.long	.Ltmp29-.Ltmp28
.Ltmp28:
	.short	.Ltmp31-.Ltmp30
.Ltmp30:
	.short	4428
	.long	4105
	.p2align	2, 0x0
.Ltmp31:
.Ltmp29:
	.p2align	2, 0x0
	.section	.debug$T,"dr"
	.p2align	2, 0x0
	.long	4
	.short	0x1e
	.short	0x1605
	.long	0x0
	.asciz	"core::option::Option"
	.byte	243
	.byte	242
	.byte	241
	.short	0x6
	.short	0x1201
	.long	0x0
	.short	0xe
	.short	0x1008
	.long	0x3
	.byte	0x0
	.byte	0x0
	.short	0x0
	.long	0x1001
	.short	0x12
	.short	0x1601
	.long	0x1000
	.long	0x1002
	.asciz	"unwrap"
	.byte	241
	.short	0x22
	.short	0x1605
	.long	0x0
	.asciz	"trailing_inlined_function"
	.byte	242
	.byte	241
	.short	0x16
	.short	0x1601
	.long	0x1004
	.long	0x1002
	.asciz	"add_numbers"
	.short	0xe
	.short	0x1605
	.long	0x0
	.asciz	"C:\\llvm"
	.short	0x56
	.short	0x1605
	.long	0x0
	.asciz	"trailing-inlined-function.rs\\@\\trailing_inlined_function.3a6e73a087a7434a-cgu.0"
	.short	0xa
	.short	0x1605
	.long	0x0
	.byte	0
	.byte	243
	.byte	242
	.byte	241
	.short	0x1a
	.short	0x1603
	.short	0x5
	.long	0x1006
	.long	0x0
	.long	0x1007
	.long	0x1008
	.long	0x0
	.byte	242
	.byte	241
