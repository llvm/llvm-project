; Test whether the following functions, with vectors featuring negative or values larger than the element
; bit size have their results of operations generated correctly when placed into constant pools

; RUN: llc -mtriple=mips64-elf -mattr=+msa,+fp64 -relocation-model=pic < %s \
; RUN:   | FileCheck -check-prefixes=ALL,MIPS64 %s
; RUN: llc -mtriple=mips-elf -mattr=+msa,+fp64,+mips32r2 -relocation-model=pic < %s \
; RUN:   | FileCheck -check-prefixes=ALL,MIPS32 %s
; RUN: llc -mtriple=mips64el-elf -mattr=+msa,+fp64 -relocation-model=pic < %s \
; RUN:   | FileCheck -check-prefixes=ALL,MIPS64 %s
; RUN: llc -mtriple=mipsel-elf -mattr=+msa,+fp64,mips32r2 -relocation-model=pic < %s \
; RUN:   | FileCheck -check-prefixes=ALL,MIPS32 %s

@llvm_mips_bclr_w_test_const_vec_res = global <4 x i32> zeroinitializer, align 16

define void @llvm_mips_bclr_w_test_const_vec() nounwind {
entry:
  %0 = tail call <4 x i32> @llvm.mips.bclr.w(<4 x i32> <i32 2147483649, i32 2147483649, i32 7, i32 7>, <4 x i32> <i32 -1, i32 31, i32 2, i32 34>)
  store <4 x i32> %0, ptr @llvm_mips_bclr_w_test_const_vec_res
  ret void
}

declare <4 x i32> @llvm.mips.bclr.w(<4 x i32>, <4 x i32>) nounwind

; MIPS32: [[LABEL:\$CPI[0-9]+_[0-9]+]]:
; MIPS64: [[LABEL:\.LCPI[0-9]+_[0-9]+]]:
; ALL:	.4byte	1                       # 0x1
; ALL:	.4byte	1                       # 0x1
; ALL:	.4byte	3                       # 0x3
; ALL:	.4byte	3                       # 0x3
; ALL-LABEL: llvm_mips_bclr_w_test_const_vec:
; MIPS32: lw $[[R2:[0-9]+]], %got([[LABEL]])($[[R1:[0-9]+]])
; MIPS32: addiu $[[R2]], $[[R2]], %lo([[LABEL]])
; MIPS32: lw $[[R3:[0-9]+]], %got(llvm_mips_bclr_w_test_const_vec_res)($[[R1]])
; MIPS64: ld $[[R2:[0-9]+]], %got_page([[LABEL]])($[[R1:[0-9]+]])
; MIPS64: daddiu $[[R2]], $[[R2]], %got_ofst([[LABEL]])
; MIPS64: ld $[[R3:[0-9]+]], %got_disp(llvm_mips_bclr_w_test_const_vec_res)($[[R1]])
; ALL: ld.w $w0, 0($[[R2]])
; ALL: st.w $w0, 0($[[R3]])


@llvm_mips_bneg_w_test_const_vec_res = global <4 x i32> zeroinitializer, align 16

define void @llvm_mips_bneg_w_test_const_vec() nounwind {
entry:
  %0 = tail call <4 x i32> @llvm.mips.bneg.w(<4 x i32> <i32 2147483649, i32 2147483649, i32 7, i32 7>, <4 x i32> <i32 -1, i32 31, i32 2, i32 34>)
  store <4 x i32> %0, ptr @llvm_mips_bneg_w_test_const_vec_res
  ret void
}

declare <4 x i32> @llvm.mips.bneg.w(<4 x i32>, <4 x i32>) nounwind

; MIPS32: [[LABEL:\$CPI[0-9]+_[0-9]+]]:
; MIPS64: [[LABEL:\.LCPI[0-9]+_[0-9]+]]:
; ALL:	.4byte	1                       # 0x1
; ALL:	.4byte	1                       # 0x1
; ALL:	.4byte	3                       # 0x3
; ALL:	.4byte	3                       # 0x3
; ALL-LABEL: llvm_mips_bneg_w_test_const_vec:
; MIPS32: lw $[[R2:[0-9]+]], %got([[LABEL]])($[[R1:[0-9]+]])
; MIPS32: addiu $[[R2]], $[[R2]], %lo([[LABEL]])
; MIPS32: lw $[[R3:[0-9]+]], %got(llvm_mips_bneg_w_test_const_vec_res)($[[R1]])
; MIPS64: ld $[[R2:[0-9]+]], %got_page([[LABEL]])($[[R1:[0-9]+]])
; MIPS64: daddiu $[[R2]], $[[R2]], %got_ofst([[LABEL]])
; MIPS64: ld $[[R3:[0-9]+]], %got_disp(llvm_mips_bneg_w_test_const_vec_res)($[[R1]])
; ALL: ld.w $w0, 0($[[R2]])
; ALL: st.w $w0, 0($[[R3]])


@llvm_mips_bset_w_test_const_vec_res = global <4 x i32> zeroinitializer, align 16

define void @llvm_mips_bset_w_test_const_vec() nounwind {
entry:
  %0 = tail call <4 x i32> @llvm.mips.bset.w(<4 x i32> <i32 0, i32 0, i32 0, i32 0>, <4 x i32> <i32 -1, i32 31, i32 2, i32 34>)
  store <4 x i32> %0, ptr @llvm_mips_bset_w_test_const_vec_res
  ret void
}

declare <4 x i32> @llvm.mips.bset.w(<4 x i32>, <4 x i32>) nounwind

; MIPS32: [[LABEL:\$CPI[0-9]+_[0-9]+]]:
; MIPS64: [[LABEL:\.LCPI[0-9]+_[0-9]+]]:
; ALL:	.4byte	2147483648              # 0x80000000
; ALL:	.4byte	2147483648              # 0x80000000
; ALL:	.4byte	4                       # 0x4
; ALL:	.4byte	4                       # 0x4
; ALL-LABEL: llvm_mips_bset_w_test_const_vec:
; MIPS32: lw $[[R2:[0-9]+]], %got([[LABEL]])($[[R1:[0-9]+]])
; MIPS32: addiu $[[R2]], $[[R2]], %lo([[LABEL]])
; MIPS32: lw $[[R3:[0-9]+]], %got(llvm_mips_bset_w_test_const_vec_res)($[[R1]])
; MIPS64: ld $[[R2:[0-9]+]], %got_page([[LABEL]])($[[R1:[0-9]+]])
; MIPS64: daddiu $[[R2]], $[[R2]], %got_ofst([[LABEL]])
; MIPS64: ld $[[R3:[0-9]+]], %got_disp(llvm_mips_bset_w_test_const_vec_res)($[[R1]])
; ALL: ld.w $w0, 0($[[R2]])
; ALL: st.w $w0, 0($[[R3]])

@llvm_mips_sll_w_test_const_vec_res = global <4 x i32> zeroinitializer, align 16

define void @llvm_mips_sll_w_test_const_vec() nounwind {
entry:
  %0 = tail call <4 x i32> @llvm.mips.sll.w(<4 x i32> <i32 1, i32 1, i32 1, i32 1>, <4 x i32> <i32 -1, i32 31, i32 2, i32 34>)
  store <4 x i32> %0, ptr @llvm_mips_sll_w_test_const_vec_res
  ret void
}

declare <4 x i32> @llvm.mips.sll.w(<4 x i32>, <4 x i32>) nounwind

; MIPS32: [[LABEL:\$CPI[0-9]+_[0-9]+]]:
; MIPS64: [[LABEL:\.LCPI[0-9]+_[0-9]+]]:
; ALL: .4byte 2147483648              # 0x80000000
; ALL: .4byte 2147483648              # 0x80000000
; ALL: .4byte 4                       # 0x4
; ALL: .4byte 4                       # 0x4
; ALL-LABEL: llvm_mips_sll_w_test_const_vec:
; MIPS32: lw $[[R2:[0-9]+]], %got([[LABEL]])($[[R1:[0-9]+]])
; MIPS32: addiu $[[R2]], $[[R2]], %lo([[LABEL]])
; MIPS32: lw $[[R3:[0-9]+]], %got(llvm_mips_sll_w_test_const_vec_res)($[[R1]])
; MIPS64: ld $[[R2:[0-9]+]], %got_page([[LABEL]])($[[R1:[0-9]+]])
; MIPS64: daddiu $[[R2]], $[[R2]], %got_ofst([[LABEL]])
; MIPS64: ld $[[R3:[0-9]+]], %got_disp(llvm_mips_sll_w_test_const_vec_res)($[[R1]])
; ALL: ld.w $w0, 0($[[R2]])
; ALL: st.w $w0, 0($[[R3]])

@llvm_mips_sra_w_test_const_vec_res = global <4 x i32> zeroinitializer, align 16

define void @llvm_mips_sra_w_test_const_vec() nounwind {
entry:
  %0 = tail call <4 x i32> @llvm.mips.sra.w(<4 x i32> <i32 -16, i32 16, i32 16, i32 16>, <4 x i32> <i32 2, i32 -30, i32 33, i32 1>)
  store <4 x i32> %0, ptr @llvm_mips_sra_w_test_const_vec_res
  ret void
}

declare <4 x i32> @llvm.mips.sra.w(<4 x i32>, <4 x i32>) nounwind

; MIPS32: [[LABEL:\$CPI[0-9]+_[0-9]+]]:
; MIPS64: [[LABEL:\.LCPI[0-9]+_[0-9]+]]:
; ALL: .4byte 4294967292              # 0xfffffffc
; ALL: .4byte 4                       # 0x4
; ALL: .4byte 8                       # 0x8
; ALL: .4byte 8                       # 0x8
; ALL-LABEL: llvm_mips_sra_w_test_const_vec:
; MIPS32: lw $[[R2:[0-9]+]], %got([[LABEL]])($[[R1:[0-9]+]])
; MIPS32: addiu $[[R2]], $[[R2]], %lo([[LABEL]])
; MIPS32: lw $[[R3:[0-9]+]], %got(llvm_mips_sra_w_test_const_vec_res)($[[R1]])
; MIPS64: ld $[[R2:[0-9]+]], %got_page([[LABEL]])($[[R1:[0-9]+]])
; MIPS64: daddiu $[[R2]], $[[R2]], %got_ofst([[LABEL]])
; MIPS64: ld $[[R3:[0-9]+]], %got_disp(llvm_mips_sra_w_test_const_vec_res)($[[R1]])
; ALL: ld.w $w0, 0($[[R2]])
; ALL: st.w $w0, 0($[[R3]])

@llvm_mips_srl_w_test_const_vec_res = global <4 x i32> zeroinitializer, align 16

define void @llvm_mips_srl_w_test_const_vec() nounwind {
entry:
  %0 = tail call <4 x i32> @llvm.mips.srl.w(<4 x i32> <i32 -16, i32 16, i32 16, i32 16>, <4 x i32> <i32 2, i32 -30, i32 33, i32 1>)
  store <4 x i32> %0, ptr @llvm_mips_srl_w_test_const_vec_res
  ret void
}

declare <4 x i32> @llvm.mips.srl.w(<4 x i32>, <4 x i32>) nounwind

; MIPS32: [[LABEL:\$CPI[0-9]+_[0-9]+]]:
; MIPS64: [[LABEL:\.LCPI[0-9]+_[0-9]+]]:
; ALL: .4byte 1073741820              # 0x3ffffffc
; ALL: .4byte 4                       # 0x4
; ALL: .4byte 8                       # 0x8
; ALL: .4byte 8                       # 0x8
; ALL-LABEL: llvm_mips_srl_w_test_const_vec:
; MIPS32: lw $[[R2:[0-9]+]], %got([[LABEL]])($[[R1:[0-9]+]])
; MIPS32: addiu $[[R2]], $[[R2]], %lo([[LABEL]])
; MIPS32: lw $[[R3:[0-9]+]], %got(llvm_mips_srl_w_test_const_vec_res)($[[R1]])
; MIPS64: ld $[[R2:[0-9]+]], %got_page([[LABEL]])($[[R1:[0-9]+]])
; MIPS64: daddiu $[[R2]], $[[R2]], %got_ofst([[LABEL]])
; MIPS64: ld $[[R3:[0-9]+]], %got_disp(llvm_mips_srl_w_test_const_vec_res)($[[R1]])
; ALL: ld.w $w0, 0($[[R2]])
; ALL: st.w $w0, 0($[[R3]])
