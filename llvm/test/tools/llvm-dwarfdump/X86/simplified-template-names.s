# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o - \
# RUN:   | llvm-dwarfdump --verify - | FileCheck %s

# CHECK: No errors.

# To regenerate this test, run from llvm-project root:
#   LLVM_SRC_ROOT=/path/to/llvm_src PATH=/path/to/clang_build/bin:$PATH llvm/utils/update_test_body.py \
#     llvm/test/tools/llvm-dwarfdump/X86/simplified-template-names.s

.ifdef GEN
#--- gen
clang --target=x86_64-linux -g -Xclang -gsimple-template-names=mangled -Xclang -debug-forward-template-params -S -std=c++20 -fdebug-prefix-map="$LLVM_SRC_ROOT"=/proc/self/cwd "$LLVM_SRC_ROOT"/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp -o -
.endif
	.file	"simplified_template_names.cpp"
	.file	0 "/proc/self/cwd" "/proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp" md5 0xaf88d5278ad7b2df17933c22083c1f2e
	.file	1 "cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs" "simplified_template_names.cpp" md5 0xaf88d5278ad7b2df17933c22083c1f2e
	.text
	.globl	_Zli5_suffy                     # -- Begin function _Zli5_suffy
	.p2align	4
	.type	_Zli5_suffy,@function
_Zli5_suffy:                            # @_Zli5_suffy
.Lfunc_begin0:
	.loc	1 69 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:69:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp0:
	.loc	1 69 44 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:69:44
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	_Zli5_suffy, .Lfunc_end0-_Zli5_suffy
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4
	.type	main,@function
main:                                   # @main
.Lfunc_begin1:
	.loc	1 100 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:100:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
.Ltmp2:
	.loc	1 103 8 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:103:8
	movb	.L__const.main.L(%rip), %al
	movb	%al, -2(%rbp)
	.loc	1 104 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:104:3
	callq	_Z2f1IJiEEvv
	.loc	1 105 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:105:3
	callq	_Z2f1IJfEEvv
	.loc	1 106 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:106:3
	callq	_Z2f1IJbEEvv
	.loc	1 107 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:107:3
	callq	_Z2f1IJdEEvv
	.loc	1 108 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:108:3
	callq	_Z2f1IJlEEvv
	.loc	1 109 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:109:3
	callq	_Z2f1IJsEEvv
	.loc	1 110 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:110:3
	callq	_Z2f1IJjEEvv
	.loc	1 111 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:111:3
	callq	_Z2f1IJyEEvv
	.loc	1 112 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:112:3
	callq	_Z2f1IJxEEvv
	.loc	1 113 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:113:3
	callq	_Z2f1IJ3udtEEvv
	.loc	1 114 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:114:3
	callq	_Z2f1IJN2ns3udtEEEvv
	.loc	1 115 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:115:3
	callq	_Z2f1IJPN2ns3udtEEEvv
	.loc	1 116 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:116:3
	callq	_Z2f1IJN2ns5inner3udtEEEvv
	.loc	1 117 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:117:3
	callq	_Z2f1IJ2t1IJiEEEEvv
	.loc	1 118 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:118:3
	callq	_Z2f1IJifEEvv
	.loc	1 119 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:119:3
	callq	_Z2f1IJPiEEvv
	.loc	1 120 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:120:3
	callq	_Z2f1IJRiEEvv
	.loc	1 121 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:121:3
	callq	_Z2f1IJOiEEvv
	.loc	1 122 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:122:3
	callq	_Z2f1IJKiEEvv
	.loc	1 123 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:123:3
	callq	_Z2f1IJA3_iEEvv
	.loc	1 124 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:124:3
	callq	_Z2f1IJvEEvv
	.loc	1 125 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:125:3
	callq	_Z2f1IJN11outer_class11inner_classEEEvv
	.loc	1 126 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:126:3
	callq	_Z2f1IJmEEvv
	.loc	1 127 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:127:3
	callq	_Z2f2ILb1ELi3EEvv
	.loc	1 128 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:128:3
	callq	_Z2f3IN2ns11EnumerationETpTnT_JLS1_1ELS1_2EEEvv
	.loc	1 129 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:129:3
	callq	_Z2f3IN2ns16EnumerationClassETpTnT_JLS1_1ELS1_2EEEvv
	.loc	1 131 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:131:3
	callq	_Z2f3IN2ns16EnumerationSmallETpTnT_JLS1_255EEEvv
	.loc	1 132 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:132:3
	callq	_Z2f3IN2ns3$_0ETpTnT_JLS1_1ELS1_2EEEvv
	.loc	1 133 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:133:3
	callq	_Z2f3IN12_GLOBAL__N_19LocalEnumETpTnT_JLS1_0EEEvv
	.loc	1 134 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:134:3
	callq	_Z2f3IPiTpTnT_JXadL_Z1iEEEEvv
	.loc	1 135 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:135:3
	callq	_Z2f3IPiTpTnT_JLS0_0EEEvv
	.loc	1 137 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:137:3
	callq	_Z2f3ImTpTnT_JLm1EEEvv
	.loc	1 138 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:138:3
	callq	_Z2f3IyTpTnT_JLy1EEEvv
	.loc	1 139 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:139:3
	callq	_Z2f3IlTpTnT_JLl1EEEvv
	.loc	1 140 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:140:3
	callq	_Z2f3IjTpTnT_JLj1EEEvv
	.loc	1 141 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:141:3
	callq	_Z2f3IsTpTnT_JLs1EEEvv
	.loc	1 142 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:142:3
	callq	_Z2f3IhTpTnT_JLh0EEEvv
	.loc	1 143 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:143:3
	callq	_Z2f3IaTpTnT_JLa0EEEvv
	.loc	1 144 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:144:3
	callq	_Z2f3ItTpTnT_JLt1ELt2EEEvv
	.loc	1 145 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:145:3
	callq	_Z2f3IcTpTnT_JLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv
	.loc	1 146 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:146:3
	callq	_Z2f3InTpTnT_JLn18446744073709551614EEEvv
	.loc	1 147 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:147:3
	callq	_Z2f4IjLj3EEvv
	.loc	1 148 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:148:3
	callq	_Z2f1IJ2t3IiLb0EEEEvv
	.loc	1 149 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:149:3
	callq	_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv
	.loc	1 150 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:150:3
	callq	_Z2f1IJZ4mainE3$_0EEvv
	.loc	1 152 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:152:3
	callq	_Z2f1IJ2t3IS0_IZ4mainE3$_0Lb0EELb0EEEEvv
	.loc	1 153 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:153:3
	callq	_Z2f1IJFifEEEvv
	.loc	1 154 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:154:3
	callq	_Z2f1IJFvzEEEvv
	.loc	1 155 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:155:3
	callq	_Z2f1IJFvizEEEvv
	.loc	1 156 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:156:3
	callq	_Z2f1IJRKiEEvv
	.loc	1 157 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:157:3
	callq	_Z2f1IJRPKiEEvv
	.loc	1 158 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:158:3
	callq	_Z2f1IJN12_GLOBAL__N_12t5EEEvv
	.loc	1 159 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:159:3
	callq	_Z2f1IJDnEEvv
	.loc	1 160 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:160:3
	callq	_Z2f1IJPlS0_EEvv
	.loc	1 161 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:161:3
	callq	_Z2f1IJPlP3udtEEvv
	.loc	1 162 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:162:3
	callq	_Z2f1IJKPvEEvv
	.loc	1 163 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:163:3
	callq	_Z2f1IJPKPKvEEvv
	.loc	1 164 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:164:3
	callq	_Z2f1IJFvvEEEvv
	.loc	1 165 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:165:3
	callq	_Z2f1IJPFvvEEEvv
	.loc	1 166 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:166:3
	callq	_Z2f1IJPZ4mainE3$_0EEvv
	.loc	1 167 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:167:3
	callq	_Z2f1IJZ4mainE3$_1EEvv
	.loc	1 168 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:168:3
	callq	_Z2f1IJPZ4mainE3$_1EEvv
	.loc	1 169 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:169:3
	callq	_Z2f5IJ2t1IJiEEEiEvv
	.loc	1 170 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:170:3
	callq	_Z2f5IJEiEvv
	.loc	1 171 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:171:3
	callq	_Z2f6I2t1IJiEEJEEvv
	.loc	1 172 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:172:3
	callq	_Z2f1IJEEvv
	.loc	1 173 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:173:3
	callq	_Z2f1IJPKvS1_EEvv
	.loc	1 174 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:174:3
	callq	_Z2f1IJP2t1IJPiEEEEvv
	.loc	1 175 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:175:3
	callq	_Z2f1IJA_PiEEvv
	.loc	1 177 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:177:6
	leaq	-5(%rbp), %rdi
	movl	$1, %esi
	callq	_ZN2t6lsIiEEvi
	.loc	1 178 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:178:6
	leaq	-5(%rbp), %rdi
	movl	$1, %esi
	callq	_ZN2t6ltIiEEvi
	.loc	1 179 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:179:6
	leaq	-5(%rbp), %rdi
	movl	$1, %esi
	callq	_ZN2t6leIiEEvi
	.loc	1 180 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:180:6
	leaq	-5(%rbp), %rdi
	callq	_ZN2t6cvP2t1IJfEEIiEEv
	.loc	1 181 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:181:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6miIiEEvi
	.loc	1 182 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:182:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6mlIiEEvi
	.loc	1 183 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:183:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6dvIiEEvi
	.loc	1 184 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:184:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6rmIiEEvi
	.loc	1 185 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:185:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6eoIiEEvi
	.loc	1 186 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:186:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6anIiEEvi
	.loc	1 187 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:187:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6orIiEEvi
	.loc	1 188 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:188:6
	leaq	-5(%rbp), %rdi
	callq	_ZN2t6coIiEEvv
	.loc	1 189 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:189:6
	leaq	-5(%rbp), %rdi
	callq	_ZN2t6ntIiEEvv
	.loc	1 190 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:190:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6aSIiEEvi
	.loc	1 191 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:191:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6gtIiEEvi
	.loc	1 192 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:192:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6cmIiEEvi
	.loc	1 193 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:193:6
	leaq	-5(%rbp), %rdi
	callq	_ZN2t6clIiEEvv
	.loc	1 194 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:194:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6ixIiEEvi
	.loc	1 195 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:195:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6ssIiEEvi
	.loc	1 196 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:196:3
	xorl	%eax, %eax
	movl	%eax, %edi
	xorl	%esi, %esi
	callq	_ZN2t6nwIiEEPvmT_
	.loc	1 197 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:197:3
	xorl	%eax, %eax
	movl	%eax, %edi
	xorl	%esi, %esi
	callq	_ZN2t6naIiEEPvmT_
	.loc	1 198 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:198:3
	xorl	%eax, %eax
	movl	%eax, %edi
	xorl	%esi, %esi
	callq	_ZN2t6dlIiEEvPvT_
	.loc	1 199 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:199:3
	xorl	%eax, %eax
	movl	%eax, %edi
	xorl	%esi, %esi
	callq	_ZN2t6daIiEEvPvT_
	.loc	1 200 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:200:6
	leaq	-5(%rbp), %rdi
	callq	_ZN2t6awIiEEiv
	.loc	1 201 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:201:3
	movl	$42, %edi
	callq	_Zli5_suffy
	.loc	1 203 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:203:3
	callq	_Z2f1IJZ4mainE2t7EEvv
	.loc	1 204 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:204:3
	callq	_Z2f1IJRA3_iEEvv
	.loc	1 205 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:205:3
	callq	_Z2f1IJPA3_iEEvv
	.loc	1 206 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:206:3
	callq	_Z2f7I2t1Evv
	.loc	1 207 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:207:3
	callq	_Z2f8I2t1iEvv
	.loc	1 209 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:209:3
	callq	_ZN2ns8ttp_userINS_5inner3ttpEEEvv
	.loc	1 210 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:210:3
	callq	_Z2f1IJPiPDnEEvv
	.loc	1 212 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:212:3
	callq	_Z2f1IJ2t7IiEEEvv
	.loc	1 213 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:213:3
	callq	_Z2f7ITtTpTyEN2ns3inl2t9EEvv
	.loc	1 214 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:214:3
	callq	_Z2f1IJU7_AtomiciEEvv
	.loc	1 215 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:215:3
	callq	_Z2f1IJilVcEEvv
	.loc	1 216 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:216:3
	callq	_Z2f1IJDv2_iEEvv
	.loc	1 217 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:217:3
	callq	_Z2f1IJVKPiEEvv
	.loc	1 218 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:218:3
	callq	_Z2f1IJVKvEEvv
	.loc	1 219 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:219:3
	callq	_Z2f1IJ2t1IJZ4mainE3$_0EEEEvv
	.loc	1 220 7                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:220:7
	leaq	-7(%rbp), %rdi
	callq	_ZN3t10C2IvEEv
	.loc	1 221 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:221:3
	callq	_Z2f1IJM3udtKFvvEEEvv
	.loc	1 222 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:222:3
	callq	_Z2f1IJM3udtVFvvREEEvv
	.loc	1 223 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:223:3
	callq	_Z2f1IJM3udtVKFvvOEEEvv
	.loc	1 224 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:224:3
	callq	_Z2f9IiEPFvvEv
	.loc	1 225 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:225:3
	callq	_Z2f1IJKPFvvEEEvv
	.loc	1 226 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:226:3
	callq	_Z2f1IJRA1_KcEEvv
	.loc	1 227 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:227:3
	callq	_Z2f1IJKFvvREEEvv
	.loc	1 228 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:228:3
	callq	_Z2f1IJVFvvOEEEvv
	.loc	1 229 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:229:3
	callq	_Z2f1IJVKFvvEEEvv
	.loc	1 230 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:230:3
	callq	_Z2f1IJA1_KPiEEvv
	.loc	1 231 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:231:3
	callq	_Z2f1IJRA1_KPiEEvv
	.loc	1 232 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:232:3
	callq	_Z2f1IJRKM3udtFvvEEEvv
	.loc	1 233 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:233:3
	callq	_Z2f1IJFPFvfEiEEEvv
	.loc	1 234 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:234:3
	callq	_Z2f1IJA1_2t1IJiEEEEvv
	.loc	1 235 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:235:3
	callq	_Z2f1IJPDoFvvEEEvv
	.loc	1 236 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:236:3
	callq	_Z2f1IJFvZ4mainE3$_1EEEvv
	.loc	1 240 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:240:3
	callq	_Z2f1IJFvZ4mainE2t8Z4mainE3$_1EEEvv
	.loc	1 241 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:241:3
	callq	_Z2f1IJFvZ4mainE2t8EEEvv
	.loc	1 242 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:242:3
	callq	_Z19operator_not_reallyIiEvv
	.loc	1 244 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:244:3
	callq	_Z3f11IDB3_TnT_LS0_2EEvv
	.loc	1 245 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:245:3
	callq	_Z3f11IKDU5_TnT_LS0_2EEvv
	.loc	1 246 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:246:3
	callq	_Z3f11IDB65_TnT_LS0_2EEvv
	.loc	1 247 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:247:3
	callq	_Z3f11IKDU65_TnT_LS0_2EEvv
	.loc	1 248 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:248:3
	callq	_Z2f1IJFv2t1IJEES1_EEEvv
	.loc	1 249 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:249:3
	callq	_Z2f1IJM2t1IJEEiEEvv
	.loc	1 251 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:251:3
	callq	_Z2f1IJU9swiftcallFvvEEEvv
	.loc	1 253 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:253:3
	callq	_Z2f1IJFivEEEvv
	.loc	1 254 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:254:3
	callq	_Z3f10ILN2ns3$_0E0EEvv
	.loc	1 255 1                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:255:1
	xorl	%eax, %eax
	.loc	1 255 1 epilogue_begin is_stmt 0 # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:255:1
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp3:
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJiEEvv,"axG",@progbits,_Z2f1IJiEEvv,comdat
	.weak	_Z2f1IJiEEvv                    # -- Begin function _Z2f1IJiEEvv
	.p2align	4
	.type	_Z2f1IJiEEvv,@function
_Z2f1IJiEEvv:                           # @_Z2f1IJiEEvv
.Lfunc_begin2:
	.loc	1 18 0 is_stmt 1                # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp4:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp5:
.Lfunc_end2:
	.size	_Z2f1IJiEEvv, .Lfunc_end2-_Z2f1IJiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJfEEvv,"axG",@progbits,_Z2f1IJfEEvv,comdat
	.weak	_Z2f1IJfEEvv                    # -- Begin function _Z2f1IJfEEvv
	.p2align	4
	.type	_Z2f1IJfEEvv,@function
_Z2f1IJfEEvv:                           # @_Z2f1IJfEEvv
.Lfunc_begin3:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp6:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp7:
.Lfunc_end3:
	.size	_Z2f1IJfEEvv, .Lfunc_end3-_Z2f1IJfEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJbEEvv,"axG",@progbits,_Z2f1IJbEEvv,comdat
	.weak	_Z2f1IJbEEvv                    # -- Begin function _Z2f1IJbEEvv
	.p2align	4
	.type	_Z2f1IJbEEvv,@function
_Z2f1IJbEEvv:                           # @_Z2f1IJbEEvv
.Lfunc_begin4:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp8:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp9:
.Lfunc_end4:
	.size	_Z2f1IJbEEvv, .Lfunc_end4-_Z2f1IJbEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJdEEvv,"axG",@progbits,_Z2f1IJdEEvv,comdat
	.weak	_Z2f1IJdEEvv                    # -- Begin function _Z2f1IJdEEvv
	.p2align	4
	.type	_Z2f1IJdEEvv,@function
_Z2f1IJdEEvv:                           # @_Z2f1IJdEEvv
.Lfunc_begin5:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp10:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp11:
.Lfunc_end5:
	.size	_Z2f1IJdEEvv, .Lfunc_end5-_Z2f1IJdEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJlEEvv,"axG",@progbits,_Z2f1IJlEEvv,comdat
	.weak	_Z2f1IJlEEvv                    # -- Begin function _Z2f1IJlEEvv
	.p2align	4
	.type	_Z2f1IJlEEvv,@function
_Z2f1IJlEEvv:                           # @_Z2f1IJlEEvv
.Lfunc_begin6:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp12:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp13:
.Lfunc_end6:
	.size	_Z2f1IJlEEvv, .Lfunc_end6-_Z2f1IJlEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJsEEvv,"axG",@progbits,_Z2f1IJsEEvv,comdat
	.weak	_Z2f1IJsEEvv                    # -- Begin function _Z2f1IJsEEvv
	.p2align	4
	.type	_Z2f1IJsEEvv,@function
_Z2f1IJsEEvv:                           # @_Z2f1IJsEEvv
.Lfunc_begin7:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp14:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp15:
.Lfunc_end7:
	.size	_Z2f1IJsEEvv, .Lfunc_end7-_Z2f1IJsEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJjEEvv,"axG",@progbits,_Z2f1IJjEEvv,comdat
	.weak	_Z2f1IJjEEvv                    # -- Begin function _Z2f1IJjEEvv
	.p2align	4
	.type	_Z2f1IJjEEvv,@function
_Z2f1IJjEEvv:                           # @_Z2f1IJjEEvv
.Lfunc_begin8:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp16:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp17:
.Lfunc_end8:
	.size	_Z2f1IJjEEvv, .Lfunc_end8-_Z2f1IJjEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJyEEvv,"axG",@progbits,_Z2f1IJyEEvv,comdat
	.weak	_Z2f1IJyEEvv                    # -- Begin function _Z2f1IJyEEvv
	.p2align	4
	.type	_Z2f1IJyEEvv,@function
_Z2f1IJyEEvv:                           # @_Z2f1IJyEEvv
.Lfunc_begin9:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp18:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp19:
.Lfunc_end9:
	.size	_Z2f1IJyEEvv, .Lfunc_end9-_Z2f1IJyEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJxEEvv,"axG",@progbits,_Z2f1IJxEEvv,comdat
	.weak	_Z2f1IJxEEvv                    # -- Begin function _Z2f1IJxEEvv
	.p2align	4
	.type	_Z2f1IJxEEvv,@function
_Z2f1IJxEEvv:                           # @_Z2f1IJxEEvv
.Lfunc_begin10:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp20:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp21:
.Lfunc_end10:
	.size	_Z2f1IJxEEvv, .Lfunc_end10-_Z2f1IJxEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJ3udtEEvv,"axG",@progbits,_Z2f1IJ3udtEEvv,comdat
	.weak	_Z2f1IJ3udtEEvv                 # -- Begin function _Z2f1IJ3udtEEvv
	.p2align	4
	.type	_Z2f1IJ3udtEEvv,@function
_Z2f1IJ3udtEEvv:                        # @_Z2f1IJ3udtEEvv
.Lfunc_begin11:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp22:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp23:
.Lfunc_end11:
	.size	_Z2f1IJ3udtEEvv, .Lfunc_end11-_Z2f1IJ3udtEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJN2ns3udtEEEvv,"axG",@progbits,_Z2f1IJN2ns3udtEEEvv,comdat
	.weak	_Z2f1IJN2ns3udtEEEvv            # -- Begin function _Z2f1IJN2ns3udtEEEvv
	.p2align	4
	.type	_Z2f1IJN2ns3udtEEEvv,@function
_Z2f1IJN2ns3udtEEEvv:                   # @_Z2f1IJN2ns3udtEEEvv
.Lfunc_begin12:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp24:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp25:
.Lfunc_end12:
	.size	_Z2f1IJN2ns3udtEEEvv, .Lfunc_end12-_Z2f1IJN2ns3udtEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPN2ns3udtEEEvv,"axG",@progbits,_Z2f1IJPN2ns3udtEEEvv,comdat
	.weak	_Z2f1IJPN2ns3udtEEEvv           # -- Begin function _Z2f1IJPN2ns3udtEEEvv
	.p2align	4
	.type	_Z2f1IJPN2ns3udtEEEvv,@function
_Z2f1IJPN2ns3udtEEEvv:                  # @_Z2f1IJPN2ns3udtEEEvv
.Lfunc_begin13:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp26:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp27:
.Lfunc_end13:
	.size	_Z2f1IJPN2ns3udtEEEvv, .Lfunc_end13-_Z2f1IJPN2ns3udtEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJN2ns5inner3udtEEEvv,"axG",@progbits,_Z2f1IJN2ns5inner3udtEEEvv,comdat
	.weak	_Z2f1IJN2ns5inner3udtEEEvv      # -- Begin function _Z2f1IJN2ns5inner3udtEEEvv
	.p2align	4
	.type	_Z2f1IJN2ns5inner3udtEEEvv,@function
_Z2f1IJN2ns5inner3udtEEEvv:             # @_Z2f1IJN2ns5inner3udtEEEvv
.Lfunc_begin14:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp28:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp29:
.Lfunc_end14:
	.size	_Z2f1IJN2ns5inner3udtEEEvv, .Lfunc_end14-_Z2f1IJN2ns5inner3udtEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJ2t1IJiEEEEvv,"axG",@progbits,_Z2f1IJ2t1IJiEEEEvv,comdat
	.weak	_Z2f1IJ2t1IJiEEEEvv             # -- Begin function _Z2f1IJ2t1IJiEEEEvv
	.p2align	4
	.type	_Z2f1IJ2t1IJiEEEEvv,@function
_Z2f1IJ2t1IJiEEEEvv:                    # @_Z2f1IJ2t1IJiEEEEvv
.Lfunc_begin15:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp30:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp31:
.Lfunc_end15:
	.size	_Z2f1IJ2t1IJiEEEEvv, .Lfunc_end15-_Z2f1IJ2t1IJiEEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJifEEvv,"axG",@progbits,_Z2f1IJifEEvv,comdat
	.weak	_Z2f1IJifEEvv                   # -- Begin function _Z2f1IJifEEvv
	.p2align	4
	.type	_Z2f1IJifEEvv,@function
_Z2f1IJifEEvv:                          # @_Z2f1IJifEEvv
.Lfunc_begin16:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp32:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp33:
.Lfunc_end16:
	.size	_Z2f1IJifEEvv, .Lfunc_end16-_Z2f1IJifEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPiEEvv,"axG",@progbits,_Z2f1IJPiEEvv,comdat
	.weak	_Z2f1IJPiEEvv                   # -- Begin function _Z2f1IJPiEEvv
	.p2align	4
	.type	_Z2f1IJPiEEvv,@function
_Z2f1IJPiEEvv:                          # @_Z2f1IJPiEEvv
.Lfunc_begin17:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp34:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp35:
.Lfunc_end17:
	.size	_Z2f1IJPiEEvv, .Lfunc_end17-_Z2f1IJPiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJRiEEvv,"axG",@progbits,_Z2f1IJRiEEvv,comdat
	.weak	_Z2f1IJRiEEvv                   # -- Begin function _Z2f1IJRiEEvv
	.p2align	4
	.type	_Z2f1IJRiEEvv,@function
_Z2f1IJRiEEvv:                          # @_Z2f1IJRiEEvv
.Lfunc_begin18:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp36:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp37:
.Lfunc_end18:
	.size	_Z2f1IJRiEEvv, .Lfunc_end18-_Z2f1IJRiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJOiEEvv,"axG",@progbits,_Z2f1IJOiEEvv,comdat
	.weak	_Z2f1IJOiEEvv                   # -- Begin function _Z2f1IJOiEEvv
	.p2align	4
	.type	_Z2f1IJOiEEvv,@function
_Z2f1IJOiEEvv:                          # @_Z2f1IJOiEEvv
.Lfunc_begin19:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp38:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp39:
.Lfunc_end19:
	.size	_Z2f1IJOiEEvv, .Lfunc_end19-_Z2f1IJOiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJKiEEvv,"axG",@progbits,_Z2f1IJKiEEvv,comdat
	.weak	_Z2f1IJKiEEvv                   # -- Begin function _Z2f1IJKiEEvv
	.p2align	4
	.type	_Z2f1IJKiEEvv,@function
_Z2f1IJKiEEvv:                          # @_Z2f1IJKiEEvv
.Lfunc_begin20:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp40:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp41:
.Lfunc_end20:
	.size	_Z2f1IJKiEEvv, .Lfunc_end20-_Z2f1IJKiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJA3_iEEvv,"axG",@progbits,_Z2f1IJA3_iEEvv,comdat
	.weak	_Z2f1IJA3_iEEvv                 # -- Begin function _Z2f1IJA3_iEEvv
	.p2align	4
	.type	_Z2f1IJA3_iEEvv,@function
_Z2f1IJA3_iEEvv:                        # @_Z2f1IJA3_iEEvv
.Lfunc_begin21:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp42:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp43:
.Lfunc_end21:
	.size	_Z2f1IJA3_iEEvv, .Lfunc_end21-_Z2f1IJA3_iEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJvEEvv,"axG",@progbits,_Z2f1IJvEEvv,comdat
	.weak	_Z2f1IJvEEvv                    # -- Begin function _Z2f1IJvEEvv
	.p2align	4
	.type	_Z2f1IJvEEvv,@function
_Z2f1IJvEEvv:                           # @_Z2f1IJvEEvv
.Lfunc_begin22:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp44:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp45:
.Lfunc_end22:
	.size	_Z2f1IJvEEvv, .Lfunc_end22-_Z2f1IJvEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJN11outer_class11inner_classEEEvv,"axG",@progbits,_Z2f1IJN11outer_class11inner_classEEEvv,comdat
	.weak	_Z2f1IJN11outer_class11inner_classEEEvv # -- Begin function _Z2f1IJN11outer_class11inner_classEEEvv
	.p2align	4
	.type	_Z2f1IJN11outer_class11inner_classEEEvv,@function
_Z2f1IJN11outer_class11inner_classEEEvv: # @_Z2f1IJN11outer_class11inner_classEEEvv
.Lfunc_begin23:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp46:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp47:
.Lfunc_end23:
	.size	_Z2f1IJN11outer_class11inner_classEEEvv, .Lfunc_end23-_Z2f1IJN11outer_class11inner_classEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJmEEvv,"axG",@progbits,_Z2f1IJmEEvv,comdat
	.weak	_Z2f1IJmEEvv                    # -- Begin function _Z2f1IJmEEvv
	.p2align	4
	.type	_Z2f1IJmEEvv,@function
_Z2f1IJmEEvv:                           # @_Z2f1IJmEEvv
.Lfunc_begin24:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp48:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp49:
.Lfunc_end24:
	.size	_Z2f1IJmEEvv, .Lfunc_end24-_Z2f1IJmEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f2ILb1ELi3EEvv,"axG",@progbits,_Z2f2ILb1ELi3EEvv,comdat
	.weak	_Z2f2ILb1ELi3EEvv               # -- Begin function _Z2f2ILb1ELi3EEvv
	.p2align	4
	.type	_Z2f2ILb1ELi3EEvv,@function
_Z2f2ILb1ELi3EEvv:                      # @_Z2f2ILb1ELi3EEvv
.Lfunc_begin25:
	.loc	1 22 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:22:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp50:
	.loc	1 22 37 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:22:37
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp51:
.Lfunc_end25:
	.size	_Z2f2ILb1ELi3EEvv, .Lfunc_end25-_Z2f2ILb1ELi3EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IN2ns11EnumerationETpTnT_JLS1_1ELS1_2EEEvv,"axG",@progbits,_Z2f3IN2ns11EnumerationETpTnT_JLS1_1ELS1_2EEEvv,comdat
	.weak	_Z2f3IN2ns11EnumerationETpTnT_JLS1_1ELS1_2EEEvv # -- Begin function _Z2f3IN2ns11EnumerationETpTnT_JLS1_1ELS1_2EEEvv
	.p2align	4
	.type	_Z2f3IN2ns11EnumerationETpTnT_JLS1_1ELS1_2EEEvv,@function
_Z2f3IN2ns11EnumerationETpTnT_JLS1_1ELS1_2EEEvv: # @_Z2f3IN2ns11EnumerationETpTnT_JLS1_1ELS1_2EEEvv
.Lfunc_begin26:
	.loc	1 23 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp52:
	.loc	1 23 42 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:42
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp53:
.Lfunc_end26:
	.size	_Z2f3IN2ns11EnumerationETpTnT_JLS1_1ELS1_2EEEvv, .Lfunc_end26-_Z2f3IN2ns11EnumerationETpTnT_JLS1_1ELS1_2EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IN2ns16EnumerationClassETpTnT_JLS1_1ELS1_2EEEvv,"axG",@progbits,_Z2f3IN2ns16EnumerationClassETpTnT_JLS1_1ELS1_2EEEvv,comdat
	.weak	_Z2f3IN2ns16EnumerationClassETpTnT_JLS1_1ELS1_2EEEvv # -- Begin function _Z2f3IN2ns16EnumerationClassETpTnT_JLS1_1ELS1_2EEEvv
	.p2align	4
	.type	_Z2f3IN2ns16EnumerationClassETpTnT_JLS1_1ELS1_2EEEvv,@function
_Z2f3IN2ns16EnumerationClassETpTnT_JLS1_1ELS1_2EEEvv: # @_Z2f3IN2ns16EnumerationClassETpTnT_JLS1_1ELS1_2EEEvv
.Lfunc_begin27:
	.loc	1 23 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp54:
	.loc	1 23 42 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:42
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp55:
.Lfunc_end27:
	.size	_Z2f3IN2ns16EnumerationClassETpTnT_JLS1_1ELS1_2EEEvv, .Lfunc_end27-_Z2f3IN2ns16EnumerationClassETpTnT_JLS1_1ELS1_2EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IN2ns16EnumerationSmallETpTnT_JLS1_255EEEvv,"axG",@progbits,_Z2f3IN2ns16EnumerationSmallETpTnT_JLS1_255EEEvv,comdat
	.weak	_Z2f3IN2ns16EnumerationSmallETpTnT_JLS1_255EEEvv # -- Begin function _Z2f3IN2ns16EnumerationSmallETpTnT_JLS1_255EEEvv
	.p2align	4
	.type	_Z2f3IN2ns16EnumerationSmallETpTnT_JLS1_255EEEvv,@function
_Z2f3IN2ns16EnumerationSmallETpTnT_JLS1_255EEEvv: # @_Z2f3IN2ns16EnumerationSmallETpTnT_JLS1_255EEEvv
.Lfunc_begin28:
	.loc	1 23 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp56:
	.loc	1 23 42 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:42
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp57:
.Lfunc_end28:
	.size	_Z2f3IN2ns16EnumerationSmallETpTnT_JLS1_255EEEvv, .Lfunc_end28-_Z2f3IN2ns16EnumerationSmallETpTnT_JLS1_255EEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4                               # -- Begin function _Z2f3IN2ns3$_0ETpTnT_JLS1_1ELS1_2EEEvv
	.type	_Z2f3IN2ns3$_0ETpTnT_JLS1_1ELS1_2EEEvv,@function
_Z2f3IN2ns3$_0ETpTnT_JLS1_1ELS1_2EEEvv: # @"_Z2f3IN2ns3$_0ETpTnT_JLS1_1ELS1_2EEEvv"
.Lfunc_begin29:
	.loc	1 23 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp58:
	.loc	1 23 42 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:42
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp59:
.Lfunc_end29:
	.size	_Z2f3IN2ns3$_0ETpTnT_JLS1_1ELS1_2EEEvv, .Lfunc_end29-_Z2f3IN2ns3$_0ETpTnT_JLS1_1ELS1_2EEEvv
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function _Z2f3IN12_GLOBAL__N_19LocalEnumETpTnT_JLS1_0EEEvv
	.type	_Z2f3IN12_GLOBAL__N_19LocalEnumETpTnT_JLS1_0EEEvv,@function
_Z2f3IN12_GLOBAL__N_19LocalEnumETpTnT_JLS1_0EEEvv: # @_Z2f3IN12_GLOBAL__N_19LocalEnumETpTnT_JLS1_0EEEvv
.Lfunc_begin30:
	.loc	1 23 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp60:
	.loc	1 23 42 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:42
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp61:
.Lfunc_end30:
	.size	_Z2f3IN12_GLOBAL__N_19LocalEnumETpTnT_JLS1_0EEEvv, .Lfunc_end30-_Z2f3IN12_GLOBAL__N_19LocalEnumETpTnT_JLS1_0EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IPiTpTnT_JXadL_Z1iEEEEvv,"axG",@progbits,_Z2f3IPiTpTnT_JXadL_Z1iEEEEvv,comdat
	.weak	_Z2f3IPiTpTnT_JXadL_Z1iEEEEvv   # -- Begin function _Z2f3IPiTpTnT_JXadL_Z1iEEEEvv
	.p2align	4
	.type	_Z2f3IPiTpTnT_JXadL_Z1iEEEEvv,@function
_Z2f3IPiTpTnT_JXadL_Z1iEEEEvv:          # @_Z2f3IPiTpTnT_JXadL_Z1iEEEEvv
.Lfunc_begin31:
	.loc	1 23 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp62:
	.loc	1 23 42 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:42
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp63:
.Lfunc_end31:
	.size	_Z2f3IPiTpTnT_JXadL_Z1iEEEEvv, .Lfunc_end31-_Z2f3IPiTpTnT_JXadL_Z1iEEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IPiTpTnT_JLS0_0EEEvv,"axG",@progbits,_Z2f3IPiTpTnT_JLS0_0EEEvv,comdat
	.weak	_Z2f3IPiTpTnT_JLS0_0EEEvv       # -- Begin function _Z2f3IPiTpTnT_JLS0_0EEEvv
	.p2align	4
	.type	_Z2f3IPiTpTnT_JLS0_0EEEvv,@function
_Z2f3IPiTpTnT_JLS0_0EEEvv:              # @_Z2f3IPiTpTnT_JLS0_0EEEvv
.Lfunc_begin32:
	.loc	1 23 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp64:
	.loc	1 23 42 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:42
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp65:
.Lfunc_end32:
	.size	_Z2f3IPiTpTnT_JLS0_0EEEvv, .Lfunc_end32-_Z2f3IPiTpTnT_JLS0_0EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3ImTpTnT_JLm1EEEvv,"axG",@progbits,_Z2f3ImTpTnT_JLm1EEEvv,comdat
	.weak	_Z2f3ImTpTnT_JLm1EEEvv          # -- Begin function _Z2f3ImTpTnT_JLm1EEEvv
	.p2align	4
	.type	_Z2f3ImTpTnT_JLm1EEEvv,@function
_Z2f3ImTpTnT_JLm1EEEvv:                 # @_Z2f3ImTpTnT_JLm1EEEvv
.Lfunc_begin33:
	.loc	1 23 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp66:
	.loc	1 23 42 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:42
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp67:
.Lfunc_end33:
	.size	_Z2f3ImTpTnT_JLm1EEEvv, .Lfunc_end33-_Z2f3ImTpTnT_JLm1EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IyTpTnT_JLy1EEEvv,"axG",@progbits,_Z2f3IyTpTnT_JLy1EEEvv,comdat
	.weak	_Z2f3IyTpTnT_JLy1EEEvv          # -- Begin function _Z2f3IyTpTnT_JLy1EEEvv
	.p2align	4
	.type	_Z2f3IyTpTnT_JLy1EEEvv,@function
_Z2f3IyTpTnT_JLy1EEEvv:                 # @_Z2f3IyTpTnT_JLy1EEEvv
.Lfunc_begin34:
	.loc	1 23 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp68:
	.loc	1 23 42 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:42
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp69:
.Lfunc_end34:
	.size	_Z2f3IyTpTnT_JLy1EEEvv, .Lfunc_end34-_Z2f3IyTpTnT_JLy1EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IlTpTnT_JLl1EEEvv,"axG",@progbits,_Z2f3IlTpTnT_JLl1EEEvv,comdat
	.weak	_Z2f3IlTpTnT_JLl1EEEvv          # -- Begin function _Z2f3IlTpTnT_JLl1EEEvv
	.p2align	4
	.type	_Z2f3IlTpTnT_JLl1EEEvv,@function
_Z2f3IlTpTnT_JLl1EEEvv:                 # @_Z2f3IlTpTnT_JLl1EEEvv
.Lfunc_begin35:
	.loc	1 23 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp70:
	.loc	1 23 42 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:42
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp71:
.Lfunc_end35:
	.size	_Z2f3IlTpTnT_JLl1EEEvv, .Lfunc_end35-_Z2f3IlTpTnT_JLl1EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IjTpTnT_JLj1EEEvv,"axG",@progbits,_Z2f3IjTpTnT_JLj1EEEvv,comdat
	.weak	_Z2f3IjTpTnT_JLj1EEEvv          # -- Begin function _Z2f3IjTpTnT_JLj1EEEvv
	.p2align	4
	.type	_Z2f3IjTpTnT_JLj1EEEvv,@function
_Z2f3IjTpTnT_JLj1EEEvv:                 # @_Z2f3IjTpTnT_JLj1EEEvv
.Lfunc_begin36:
	.loc	1 23 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp72:
	.loc	1 23 42 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:42
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp73:
.Lfunc_end36:
	.size	_Z2f3IjTpTnT_JLj1EEEvv, .Lfunc_end36-_Z2f3IjTpTnT_JLj1EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IsTpTnT_JLs1EEEvv,"axG",@progbits,_Z2f3IsTpTnT_JLs1EEEvv,comdat
	.weak	_Z2f3IsTpTnT_JLs1EEEvv          # -- Begin function _Z2f3IsTpTnT_JLs1EEEvv
	.p2align	4
	.type	_Z2f3IsTpTnT_JLs1EEEvv,@function
_Z2f3IsTpTnT_JLs1EEEvv:                 # @_Z2f3IsTpTnT_JLs1EEEvv
.Lfunc_begin37:
	.loc	1 23 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp74:
	.loc	1 23 42 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:42
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp75:
.Lfunc_end37:
	.size	_Z2f3IsTpTnT_JLs1EEEvv, .Lfunc_end37-_Z2f3IsTpTnT_JLs1EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IhTpTnT_JLh0EEEvv,"axG",@progbits,_Z2f3IhTpTnT_JLh0EEEvv,comdat
	.weak	_Z2f3IhTpTnT_JLh0EEEvv          # -- Begin function _Z2f3IhTpTnT_JLh0EEEvv
	.p2align	4
	.type	_Z2f3IhTpTnT_JLh0EEEvv,@function
_Z2f3IhTpTnT_JLh0EEEvv:                 # @_Z2f3IhTpTnT_JLh0EEEvv
.Lfunc_begin38:
	.loc	1 23 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp76:
	.loc	1 23 42 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:42
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp77:
.Lfunc_end38:
	.size	_Z2f3IhTpTnT_JLh0EEEvv, .Lfunc_end38-_Z2f3IhTpTnT_JLh0EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IaTpTnT_JLa0EEEvv,"axG",@progbits,_Z2f3IaTpTnT_JLa0EEEvv,comdat
	.weak	_Z2f3IaTpTnT_JLa0EEEvv          # -- Begin function _Z2f3IaTpTnT_JLa0EEEvv
	.p2align	4
	.type	_Z2f3IaTpTnT_JLa0EEEvv,@function
_Z2f3IaTpTnT_JLa0EEEvv:                 # @_Z2f3IaTpTnT_JLa0EEEvv
.Lfunc_begin39:
	.loc	1 23 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp78:
	.loc	1 23 42 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:42
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp79:
.Lfunc_end39:
	.size	_Z2f3IaTpTnT_JLa0EEEvv, .Lfunc_end39-_Z2f3IaTpTnT_JLa0EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3ItTpTnT_JLt1ELt2EEEvv,"axG",@progbits,_Z2f3ItTpTnT_JLt1ELt2EEEvv,comdat
	.weak	_Z2f3ItTpTnT_JLt1ELt2EEEvv      # -- Begin function _Z2f3ItTpTnT_JLt1ELt2EEEvv
	.p2align	4
	.type	_Z2f3ItTpTnT_JLt1ELt2EEEvv,@function
_Z2f3ItTpTnT_JLt1ELt2EEEvv:             # @_Z2f3ItTpTnT_JLt1ELt2EEEvv
.Lfunc_begin40:
	.loc	1 23 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp80:
	.loc	1 23 42 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:42
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp81:
.Lfunc_end40:
	.size	_Z2f3ItTpTnT_JLt1ELt2EEEvv, .Lfunc_end40-_Z2f3ItTpTnT_JLt1ELt2EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IcTpTnT_JLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv,"axG",@progbits,_Z2f3IcTpTnT_JLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv,comdat
	.weak	_Z2f3IcTpTnT_JLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv # -- Begin function _Z2f3IcTpTnT_JLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv
	.p2align	4
	.type	_Z2f3IcTpTnT_JLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv,@function
_Z2f3IcTpTnT_JLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv: # @_Z2f3IcTpTnT_JLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv
.Lfunc_begin41:
	.loc	1 23 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp82:
	.loc	1 23 42 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:42
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp83:
.Lfunc_end41:
	.size	_Z2f3IcTpTnT_JLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv, .Lfunc_end41-_Z2f3IcTpTnT_JLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3InTpTnT_JLn18446744073709551614EEEvv,"axG",@progbits,_Z2f3InTpTnT_JLn18446744073709551614EEEvv,comdat
	.weak	_Z2f3InTpTnT_JLn18446744073709551614EEEvv # -- Begin function _Z2f3InTpTnT_JLn18446744073709551614EEEvv
	.p2align	4
	.type	_Z2f3InTpTnT_JLn18446744073709551614EEEvv,@function
_Z2f3InTpTnT_JLn18446744073709551614EEEvv: # @_Z2f3InTpTnT_JLn18446744073709551614EEEvv
.Lfunc_begin42:
	.loc	1 23 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp84:
	.loc	1 23 42 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:23:42
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp85:
.Lfunc_end42:
	.size	_Z2f3InTpTnT_JLn18446744073709551614EEEvv, .Lfunc_end42-_Z2f3InTpTnT_JLn18446744073709551614EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f4IjLj3EEvv,"axG",@progbits,_Z2f4IjLj3EEvv,comdat
	.weak	_Z2f4IjLj3EEvv                  # -- Begin function _Z2f4IjLj3EEvv
	.p2align	4
	.type	_Z2f4IjLj3EEvv,@function
_Z2f4IjLj3EEvv:                         # @_Z2f4IjLj3EEvv
.Lfunc_begin43:
	.loc	1 24 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:24:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp86:
	.loc	1 24 48 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:24:48
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp87:
.Lfunc_end43:
	.size	_Z2f4IjLj3EEvv, .Lfunc_end43-_Z2f4IjLj3EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJ2t3IiLb0EEEEvv,"axG",@progbits,_Z2f1IJ2t3IiLb0EEEEvv,comdat
	.weak	_Z2f1IJ2t3IiLb0EEEEvv           # -- Begin function _Z2f1IJ2t3IiLb0EEEEvv
	.p2align	4
	.type	_Z2f1IJ2t3IiLb0EEEEvv,@function
_Z2f1IJ2t3IiLb0EEEEvv:                  # @_Z2f1IJ2t3IiLb0EEEEvv
.Lfunc_begin44:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp88:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp89:
.Lfunc_end44:
	.size	_Z2f1IJ2t3IiLb0EEEEvv, .Lfunc_end44-_Z2f1IJ2t3IiLb0EEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv,"axG",@progbits,_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv,comdat
	.weak	_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv  # -- Begin function _Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv
	.p2align	4
	.type	_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv,@function
_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv:         # @_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv
.Lfunc_begin45:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp90:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp91:
.Lfunc_end45:
	.size	_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv, .Lfunc_end45-_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4                               # -- Begin function _Z2f1IJZ4mainE3$_0EEvv
	.type	_Z2f1IJZ4mainE3$_0EEvv,@function
_Z2f1IJZ4mainE3$_0EEvv:                 # @"_Z2f1IJZ4mainE3$_0EEvv"
.Lfunc_begin46:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp92:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp93:
.Lfunc_end46:
	.size	_Z2f1IJZ4mainE3$_0EEvv, .Lfunc_end46-_Z2f1IJZ4mainE3$_0EEvv
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function _Z2f1IJ2t3IS0_IZ4mainE3$_0Lb0EELb0EEEEvv
	.type	_Z2f1IJ2t3IS0_IZ4mainE3$_0Lb0EELb0EEEEvv,@function
_Z2f1IJ2t3IS0_IZ4mainE3$_0Lb0EELb0EEEEvv: # @"_Z2f1IJ2t3IS0_IZ4mainE3$_0Lb0EELb0EEEEvv"
.Lfunc_begin47:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp94:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp95:
.Lfunc_end47:
	.size	_Z2f1IJ2t3IS0_IZ4mainE3$_0Lb0EELb0EEEEvv, .Lfunc_end47-_Z2f1IJ2t3IS0_IZ4mainE3$_0Lb0EELb0EEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJFifEEEvv,"axG",@progbits,_Z2f1IJFifEEEvv,comdat
	.weak	_Z2f1IJFifEEEvv                 # -- Begin function _Z2f1IJFifEEEvv
	.p2align	4
	.type	_Z2f1IJFifEEEvv,@function
_Z2f1IJFifEEEvv:                        # @_Z2f1IJFifEEEvv
.Lfunc_begin48:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp96:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp97:
.Lfunc_end48:
	.size	_Z2f1IJFifEEEvv, .Lfunc_end48-_Z2f1IJFifEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJFvzEEEvv,"axG",@progbits,_Z2f1IJFvzEEEvv,comdat
	.weak	_Z2f1IJFvzEEEvv                 # -- Begin function _Z2f1IJFvzEEEvv
	.p2align	4
	.type	_Z2f1IJFvzEEEvv,@function
_Z2f1IJFvzEEEvv:                        # @_Z2f1IJFvzEEEvv
.Lfunc_begin49:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp98:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp99:
.Lfunc_end49:
	.size	_Z2f1IJFvzEEEvv, .Lfunc_end49-_Z2f1IJFvzEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJFvizEEEvv,"axG",@progbits,_Z2f1IJFvizEEEvv,comdat
	.weak	_Z2f1IJFvizEEEvv                # -- Begin function _Z2f1IJFvizEEEvv
	.p2align	4
	.type	_Z2f1IJFvizEEEvv,@function
_Z2f1IJFvizEEEvv:                       # @_Z2f1IJFvizEEEvv
.Lfunc_begin50:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp100:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp101:
.Lfunc_end50:
	.size	_Z2f1IJFvizEEEvv, .Lfunc_end50-_Z2f1IJFvizEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJRKiEEvv,"axG",@progbits,_Z2f1IJRKiEEvv,comdat
	.weak	_Z2f1IJRKiEEvv                  # -- Begin function _Z2f1IJRKiEEvv
	.p2align	4
	.type	_Z2f1IJRKiEEvv,@function
_Z2f1IJRKiEEvv:                         # @_Z2f1IJRKiEEvv
.Lfunc_begin51:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp102:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp103:
.Lfunc_end51:
	.size	_Z2f1IJRKiEEvv, .Lfunc_end51-_Z2f1IJRKiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJRPKiEEvv,"axG",@progbits,_Z2f1IJRPKiEEvv,comdat
	.weak	_Z2f1IJRPKiEEvv                 # -- Begin function _Z2f1IJRPKiEEvv
	.p2align	4
	.type	_Z2f1IJRPKiEEvv,@function
_Z2f1IJRPKiEEvv:                        # @_Z2f1IJRPKiEEvv
.Lfunc_begin52:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp104:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp105:
.Lfunc_end52:
	.size	_Z2f1IJRPKiEEvv, .Lfunc_end52-_Z2f1IJRPKiEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4                               # -- Begin function _Z2f1IJN12_GLOBAL__N_12t5EEEvv
	.type	_Z2f1IJN12_GLOBAL__N_12t5EEEvv,@function
_Z2f1IJN12_GLOBAL__N_12t5EEEvv:         # @_Z2f1IJN12_GLOBAL__N_12t5EEEvv
.Lfunc_begin53:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp106:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp107:
.Lfunc_end53:
	.size	_Z2f1IJN12_GLOBAL__N_12t5EEEvv, .Lfunc_end53-_Z2f1IJN12_GLOBAL__N_12t5EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJDnEEvv,"axG",@progbits,_Z2f1IJDnEEvv,comdat
	.weak	_Z2f1IJDnEEvv                   # -- Begin function _Z2f1IJDnEEvv
	.p2align	4
	.type	_Z2f1IJDnEEvv,@function
_Z2f1IJDnEEvv:                          # @_Z2f1IJDnEEvv
.Lfunc_begin54:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp108:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp109:
.Lfunc_end54:
	.size	_Z2f1IJDnEEvv, .Lfunc_end54-_Z2f1IJDnEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPlS0_EEvv,"axG",@progbits,_Z2f1IJPlS0_EEvv,comdat
	.weak	_Z2f1IJPlS0_EEvv                # -- Begin function _Z2f1IJPlS0_EEvv
	.p2align	4
	.type	_Z2f1IJPlS0_EEvv,@function
_Z2f1IJPlS0_EEvv:                       # @_Z2f1IJPlS0_EEvv
.Lfunc_begin55:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp110:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp111:
.Lfunc_end55:
	.size	_Z2f1IJPlS0_EEvv, .Lfunc_end55-_Z2f1IJPlS0_EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPlP3udtEEvv,"axG",@progbits,_Z2f1IJPlP3udtEEvv,comdat
	.weak	_Z2f1IJPlP3udtEEvv              # -- Begin function _Z2f1IJPlP3udtEEvv
	.p2align	4
	.type	_Z2f1IJPlP3udtEEvv,@function
_Z2f1IJPlP3udtEEvv:                     # @_Z2f1IJPlP3udtEEvv
.Lfunc_begin56:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp112:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp113:
.Lfunc_end56:
	.size	_Z2f1IJPlP3udtEEvv, .Lfunc_end56-_Z2f1IJPlP3udtEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJKPvEEvv,"axG",@progbits,_Z2f1IJKPvEEvv,comdat
	.weak	_Z2f1IJKPvEEvv                  # -- Begin function _Z2f1IJKPvEEvv
	.p2align	4
	.type	_Z2f1IJKPvEEvv,@function
_Z2f1IJKPvEEvv:                         # @_Z2f1IJKPvEEvv
.Lfunc_begin57:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp114:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp115:
.Lfunc_end57:
	.size	_Z2f1IJKPvEEvv, .Lfunc_end57-_Z2f1IJKPvEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPKPKvEEvv,"axG",@progbits,_Z2f1IJPKPKvEEvv,comdat
	.weak	_Z2f1IJPKPKvEEvv                # -- Begin function _Z2f1IJPKPKvEEvv
	.p2align	4
	.type	_Z2f1IJPKPKvEEvv,@function
_Z2f1IJPKPKvEEvv:                       # @_Z2f1IJPKPKvEEvv
.Lfunc_begin58:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp116:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp117:
.Lfunc_end58:
	.size	_Z2f1IJPKPKvEEvv, .Lfunc_end58-_Z2f1IJPKPKvEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJFvvEEEvv,"axG",@progbits,_Z2f1IJFvvEEEvv,comdat
	.weak	_Z2f1IJFvvEEEvv                 # -- Begin function _Z2f1IJFvvEEEvv
	.p2align	4
	.type	_Z2f1IJFvvEEEvv,@function
_Z2f1IJFvvEEEvv:                        # @_Z2f1IJFvvEEEvv
.Lfunc_begin59:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp118:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp119:
.Lfunc_end59:
	.size	_Z2f1IJFvvEEEvv, .Lfunc_end59-_Z2f1IJFvvEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPFvvEEEvv,"axG",@progbits,_Z2f1IJPFvvEEEvv,comdat
	.weak	_Z2f1IJPFvvEEEvv                # -- Begin function _Z2f1IJPFvvEEEvv
	.p2align	4
	.type	_Z2f1IJPFvvEEEvv,@function
_Z2f1IJPFvvEEEvv:                       # @_Z2f1IJPFvvEEEvv
.Lfunc_begin60:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp120:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp121:
.Lfunc_end60:
	.size	_Z2f1IJPFvvEEEvv, .Lfunc_end60-_Z2f1IJPFvvEEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4                               # -- Begin function _Z2f1IJPZ4mainE3$_0EEvv
	.type	_Z2f1IJPZ4mainE3$_0EEvv,@function
_Z2f1IJPZ4mainE3$_0EEvv:                # @"_Z2f1IJPZ4mainE3$_0EEvv"
.Lfunc_begin61:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp122:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp123:
.Lfunc_end61:
	.size	_Z2f1IJPZ4mainE3$_0EEvv, .Lfunc_end61-_Z2f1IJPZ4mainE3$_0EEvv
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function _Z2f1IJZ4mainE3$_1EEvv
	.type	_Z2f1IJZ4mainE3$_1EEvv,@function
_Z2f1IJZ4mainE3$_1EEvv:                 # @"_Z2f1IJZ4mainE3$_1EEvv"
.Lfunc_begin62:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp124:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp125:
.Lfunc_end62:
	.size	_Z2f1IJZ4mainE3$_1EEvv, .Lfunc_end62-_Z2f1IJZ4mainE3$_1EEvv
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function _Z2f1IJPZ4mainE3$_1EEvv
	.type	_Z2f1IJPZ4mainE3$_1EEvv,@function
_Z2f1IJPZ4mainE3$_1EEvv:                # @"_Z2f1IJPZ4mainE3$_1EEvv"
.Lfunc_begin63:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp126:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp127:
.Lfunc_end63:
	.size	_Z2f1IJPZ4mainE3$_1EEvv, .Lfunc_end63-_Z2f1IJPZ4mainE3$_1EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f5IJ2t1IJiEEEiEvv,"axG",@progbits,_Z2f5IJ2t1IJiEEEiEvv,comdat
	.weak	_Z2f5IJ2t1IJiEEEiEvv            # -- Begin function _Z2f5IJ2t1IJiEEEiEvv
	.p2align	4
	.type	_Z2f5IJ2t1IJiEEEiEvv,@function
_Z2f5IJ2t1IJiEEEiEvv:                   # @_Z2f5IJ2t1IJiEEEiEvv
.Lfunc_begin64:
	.loc	1 37 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:37:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp128:
	.loc	1 37 57 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:37:57
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp129:
.Lfunc_end64:
	.size	_Z2f5IJ2t1IJiEEEiEvv, .Lfunc_end64-_Z2f5IJ2t1IJiEEEiEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f5IJEiEvv,"axG",@progbits,_Z2f5IJEiEvv,comdat
	.weak	_Z2f5IJEiEvv                    # -- Begin function _Z2f5IJEiEvv
	.p2align	4
	.type	_Z2f5IJEiEvv,@function
_Z2f5IJEiEvv:                           # @_Z2f5IJEiEvv
.Lfunc_begin65:
	.loc	1 37 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:37:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp130:
	.loc	1 37 57 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:37:57
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp131:
.Lfunc_end65:
	.size	_Z2f5IJEiEvv, .Lfunc_end65-_Z2f5IJEiEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f6I2t1IJiEEJEEvv,"axG",@progbits,_Z2f6I2t1IJiEEJEEvv,comdat
	.weak	_Z2f6I2t1IJiEEJEEvv             # -- Begin function _Z2f6I2t1IJiEEJEEvv
	.p2align	4
	.type	_Z2f6I2t1IJiEEJEEvv,@function
_Z2f6I2t1IJiEEJEEvv:                    # @_Z2f6I2t1IJiEEJEEvv
.Lfunc_begin66:
	.loc	1 38 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:38:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp132:
	.loc	1 38 51 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:38:51
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp133:
.Lfunc_end66:
	.size	_Z2f6I2t1IJiEEJEEvv, .Lfunc_end66-_Z2f6I2t1IJiEEJEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJEEvv,"axG",@progbits,_Z2f1IJEEvv,comdat
	.weak	_Z2f1IJEEvv                     # -- Begin function _Z2f1IJEEvv
	.p2align	4
	.type	_Z2f1IJEEvv,@function
_Z2f1IJEEvv:                            # @_Z2f1IJEEvv
.Lfunc_begin67:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp134:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp135:
.Lfunc_end67:
	.size	_Z2f1IJEEvv, .Lfunc_end67-_Z2f1IJEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPKvS1_EEvv,"axG",@progbits,_Z2f1IJPKvS1_EEvv,comdat
	.weak	_Z2f1IJPKvS1_EEvv               # -- Begin function _Z2f1IJPKvS1_EEvv
	.p2align	4
	.type	_Z2f1IJPKvS1_EEvv,@function
_Z2f1IJPKvS1_EEvv:                      # @_Z2f1IJPKvS1_EEvv
.Lfunc_begin68:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp136:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp137:
.Lfunc_end68:
	.size	_Z2f1IJPKvS1_EEvv, .Lfunc_end68-_Z2f1IJPKvS1_EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJP2t1IJPiEEEEvv,"axG",@progbits,_Z2f1IJP2t1IJPiEEEEvv,comdat
	.weak	_Z2f1IJP2t1IJPiEEEEvv           # -- Begin function _Z2f1IJP2t1IJPiEEEEvv
	.p2align	4
	.type	_Z2f1IJP2t1IJPiEEEEvv,@function
_Z2f1IJP2t1IJPiEEEEvv:                  # @_Z2f1IJP2t1IJPiEEEEvv
.Lfunc_begin69:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp138:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp139:
.Lfunc_end69:
	.size	_Z2f1IJP2t1IJPiEEEEvv, .Lfunc_end69-_Z2f1IJP2t1IJPiEEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJA_PiEEvv,"axG",@progbits,_Z2f1IJA_PiEEvv,comdat
	.weak	_Z2f1IJA_PiEEvv                 # -- Begin function _Z2f1IJA_PiEEvv
	.p2align	4
	.type	_Z2f1IJA_PiEEvv,@function
_Z2f1IJA_PiEEvv:                        # @_Z2f1IJA_PiEEvv
.Lfunc_begin70:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp140:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp141:
.Lfunc_end70:
	.size	_Z2f1IJA_PiEEvv, .Lfunc_end70-_Z2f1IJA_PiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6lsIiEEvi,"axG",@progbits,_ZN2t6lsIiEEvi,comdat
	.weak	_ZN2t6lsIiEEvi                  # -- Begin function _ZN2t6lsIiEEvi
	.p2align	4
	.type	_ZN2t6lsIiEEvi,@function
_ZN2t6lsIiEEvi:                         # @_ZN2t6lsIiEEvi
.Lfunc_begin71:
	.loc	1 40 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:40:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp142:
	.loc	1 40 47 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:40:47
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp143:
.Lfunc_end71:
	.size	_ZN2t6lsIiEEvi, .Lfunc_end71-_ZN2t6lsIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6ltIiEEvi,"axG",@progbits,_ZN2t6ltIiEEvi,comdat
	.weak	_ZN2t6ltIiEEvi                  # -- Begin function _ZN2t6ltIiEEvi
	.p2align	4
	.type	_ZN2t6ltIiEEvi,@function
_ZN2t6ltIiEEvi:                         # @_ZN2t6ltIiEEvi
.Lfunc_begin72:
	.loc	1 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp144:
	.loc	1 41 46 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:41:46
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp145:
.Lfunc_end72:
	.size	_ZN2t6ltIiEEvi, .Lfunc_end72-_ZN2t6ltIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6leIiEEvi,"axG",@progbits,_ZN2t6leIiEEvi,comdat
	.weak	_ZN2t6leIiEEvi                  # -- Begin function _ZN2t6leIiEEvi
	.p2align	4
	.type	_ZN2t6leIiEEvi,@function
_ZN2t6leIiEEvi:                         # @_ZN2t6leIiEEvi
.Lfunc_begin73:
	.loc	1 42 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:42:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp146:
	.loc	1 42 47 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:42:47
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp147:
.Lfunc_end73:
	.size	_ZN2t6leIiEEvi, .Lfunc_end73-_ZN2t6leIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6cvP2t1IJfEEIiEEv,"axG",@progbits,_ZN2t6cvP2t1IJfEEIiEEv,comdat
	.weak	_ZN2t6cvP2t1IJfEEIiEEv          # -- Begin function _ZN2t6cvP2t1IJfEEIiEEv
	.p2align	4
	.type	_ZN2t6cvP2t1IJfEEIiEEv,@function
_ZN2t6cvP2t1IJfEEIiEEv:                 # @_ZN2t6cvP2t1IJfEEIiEEv
.Lfunc_begin74:
	.loc	1 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp148:
	.loc	1 43 56 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:43:56
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	.loc	1 43 56 epilogue_begin is_stmt 0 # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:43:56
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp149:
.Lfunc_end74:
	.size	_ZN2t6cvP2t1IJfEEIiEEv, .Lfunc_end74-_ZN2t6cvP2t1IJfEEIiEEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6miIiEEvi,"axG",@progbits,_ZN2t6miIiEEvi,comdat
	.weak	_ZN2t6miIiEEvi                  # -- Begin function _ZN2t6miIiEEvi
	.p2align	4
	.type	_ZN2t6miIiEEvi,@function
_ZN2t6miIiEEvi:                         # @_ZN2t6miIiEEvi
.Lfunc_begin75:
	.loc	1 44 0 is_stmt 1                # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:44:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp150:
	.loc	1 44 46 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:44:46
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp151:
.Lfunc_end75:
	.size	_ZN2t6miIiEEvi, .Lfunc_end75-_ZN2t6miIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6mlIiEEvi,"axG",@progbits,_ZN2t6mlIiEEvi,comdat
	.weak	_ZN2t6mlIiEEvi                  # -- Begin function _ZN2t6mlIiEEvi
	.p2align	4
	.type	_ZN2t6mlIiEEvi,@function
_ZN2t6mlIiEEvi:                         # @_ZN2t6mlIiEEvi
.Lfunc_begin76:
	.loc	1 45 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:45:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp152:
	.loc	1 45 46 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:45:46
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp153:
.Lfunc_end76:
	.size	_ZN2t6mlIiEEvi, .Lfunc_end76-_ZN2t6mlIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6dvIiEEvi,"axG",@progbits,_ZN2t6dvIiEEvi,comdat
	.weak	_ZN2t6dvIiEEvi                  # -- Begin function _ZN2t6dvIiEEvi
	.p2align	4
	.type	_ZN2t6dvIiEEvi,@function
_ZN2t6dvIiEEvi:                         # @_ZN2t6dvIiEEvi
.Lfunc_begin77:
	.loc	1 46 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:46:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp154:
	.loc	1 46 46 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:46:46
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp155:
.Lfunc_end77:
	.size	_ZN2t6dvIiEEvi, .Lfunc_end77-_ZN2t6dvIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6rmIiEEvi,"axG",@progbits,_ZN2t6rmIiEEvi,comdat
	.weak	_ZN2t6rmIiEEvi                  # -- Begin function _ZN2t6rmIiEEvi
	.p2align	4
	.type	_ZN2t6rmIiEEvi,@function
_ZN2t6rmIiEEvi:                         # @_ZN2t6rmIiEEvi
.Lfunc_begin78:
	.loc	1 47 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:47:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp156:
	.loc	1 47 46 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:47:46
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp157:
.Lfunc_end78:
	.size	_ZN2t6rmIiEEvi, .Lfunc_end78-_ZN2t6rmIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6eoIiEEvi,"axG",@progbits,_ZN2t6eoIiEEvi,comdat
	.weak	_ZN2t6eoIiEEvi                  # -- Begin function _ZN2t6eoIiEEvi
	.p2align	4
	.type	_ZN2t6eoIiEEvi,@function
_ZN2t6eoIiEEvi:                         # @_ZN2t6eoIiEEvi
.Lfunc_begin79:
	.loc	1 48 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:48:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp158:
	.loc	1 48 46 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:48:46
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp159:
.Lfunc_end79:
	.size	_ZN2t6eoIiEEvi, .Lfunc_end79-_ZN2t6eoIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6anIiEEvi,"axG",@progbits,_ZN2t6anIiEEvi,comdat
	.weak	_ZN2t6anIiEEvi                  # -- Begin function _ZN2t6anIiEEvi
	.p2align	4
	.type	_ZN2t6anIiEEvi,@function
_ZN2t6anIiEEvi:                         # @_ZN2t6anIiEEvi
.Lfunc_begin80:
	.loc	1 49 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:49:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp160:
	.loc	1 49 46 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:49:46
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp161:
.Lfunc_end80:
	.size	_ZN2t6anIiEEvi, .Lfunc_end80-_ZN2t6anIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6orIiEEvi,"axG",@progbits,_ZN2t6orIiEEvi,comdat
	.weak	_ZN2t6orIiEEvi                  # -- Begin function _ZN2t6orIiEEvi
	.p2align	4
	.type	_ZN2t6orIiEEvi,@function
_ZN2t6orIiEEvi:                         # @_ZN2t6orIiEEvi
.Lfunc_begin81:
	.loc	1 50 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:50:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp162:
	.loc	1 50 46 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:50:46
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp163:
.Lfunc_end81:
	.size	_ZN2t6orIiEEvi, .Lfunc_end81-_ZN2t6orIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6coIiEEvv,"axG",@progbits,_ZN2t6coIiEEvv,comdat
	.weak	_ZN2t6coIiEEvv                  # -- Begin function _ZN2t6coIiEEvv
	.p2align	4
	.type	_ZN2t6coIiEEvv,@function
_ZN2t6coIiEEvv:                         # @_ZN2t6coIiEEvv
.Lfunc_begin82:
	.loc	1 51 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:51:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp164:
	.loc	1 51 43 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:51:43
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp165:
.Lfunc_end82:
	.size	_ZN2t6coIiEEvv, .Lfunc_end82-_ZN2t6coIiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6ntIiEEvv,"axG",@progbits,_ZN2t6ntIiEEvv,comdat
	.weak	_ZN2t6ntIiEEvv                  # -- Begin function _ZN2t6ntIiEEvv
	.p2align	4
	.type	_ZN2t6ntIiEEvv,@function
_ZN2t6ntIiEEvv:                         # @_ZN2t6ntIiEEvv
.Lfunc_begin83:
	.loc	1 52 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:52:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp166:
	.loc	1 52 43 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:52:43
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp167:
.Lfunc_end83:
	.size	_ZN2t6ntIiEEvv, .Lfunc_end83-_ZN2t6ntIiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6aSIiEEvi,"axG",@progbits,_ZN2t6aSIiEEvi,comdat
	.weak	_ZN2t6aSIiEEvi                  # -- Begin function _ZN2t6aSIiEEvi
	.p2align	4
	.type	_ZN2t6aSIiEEvi,@function
_ZN2t6aSIiEEvi:                         # @_ZN2t6aSIiEEvi
.Lfunc_begin84:
	.loc	1 53 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:53:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp168:
	.loc	1 53 46 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:53:46
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp169:
.Lfunc_end84:
	.size	_ZN2t6aSIiEEvi, .Lfunc_end84-_ZN2t6aSIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6gtIiEEvi,"axG",@progbits,_ZN2t6gtIiEEvi,comdat
	.weak	_ZN2t6gtIiEEvi                  # -- Begin function _ZN2t6gtIiEEvi
	.p2align	4
	.type	_ZN2t6gtIiEEvi,@function
_ZN2t6gtIiEEvi:                         # @_ZN2t6gtIiEEvi
.Lfunc_begin85:
	.loc	1 54 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:54:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp170:
	.loc	1 54 46 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:54:46
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp171:
.Lfunc_end85:
	.size	_ZN2t6gtIiEEvi, .Lfunc_end85-_ZN2t6gtIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6cmIiEEvi,"axG",@progbits,_ZN2t6cmIiEEvi,comdat
	.weak	_ZN2t6cmIiEEvi                  # -- Begin function _ZN2t6cmIiEEvi
	.p2align	4
	.type	_ZN2t6cmIiEEvi,@function
_ZN2t6cmIiEEvi:                         # @_ZN2t6cmIiEEvi
.Lfunc_begin86:
	.loc	1 55 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:55:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp172:
	.loc	1 55 46 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:55:46
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp173:
.Lfunc_end86:
	.size	_ZN2t6cmIiEEvi, .Lfunc_end86-_ZN2t6cmIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6clIiEEvv,"axG",@progbits,_ZN2t6clIiEEvv,comdat
	.weak	_ZN2t6clIiEEvv                  # -- Begin function _ZN2t6clIiEEvv
	.p2align	4
	.type	_ZN2t6clIiEEvv,@function
_ZN2t6clIiEEvv:                         # @_ZN2t6clIiEEvv
.Lfunc_begin87:
	.loc	1 56 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:56:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp174:
	.loc	1 56 44 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:56:44
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp175:
.Lfunc_end87:
	.size	_ZN2t6clIiEEvv, .Lfunc_end87-_ZN2t6clIiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6ixIiEEvi,"axG",@progbits,_ZN2t6ixIiEEvi,comdat
	.weak	_ZN2t6ixIiEEvi                  # -- Begin function _ZN2t6ixIiEEvi
	.p2align	4
	.type	_ZN2t6ixIiEEvi,@function
_ZN2t6ixIiEEvi:                         # @_ZN2t6ixIiEEvi
.Lfunc_begin88:
	.loc	1 57 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:57:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp176:
	.loc	1 57 47 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:57:47
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp177:
.Lfunc_end88:
	.size	_ZN2t6ixIiEEvi, .Lfunc_end88-_ZN2t6ixIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6ssIiEEvi,"axG",@progbits,_ZN2t6ssIiEEvi,comdat
	.weak	_ZN2t6ssIiEEvi                  # -- Begin function _ZN2t6ssIiEEvi
	.p2align	4
	.type	_ZN2t6ssIiEEvi,@function
_ZN2t6ssIiEEvi:                         # @_ZN2t6ssIiEEvi
.Lfunc_begin89:
	.loc	1 58 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:58:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp178:
	.loc	1 58 48 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:58:48
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp179:
.Lfunc_end89:
	.size	_ZN2t6ssIiEEvi, .Lfunc_end89-_ZN2t6ssIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6nwIiEEPvmT_,"axG",@progbits,_ZN2t6nwIiEEPvmT_,comdat
	.weak	_ZN2t6nwIiEEPvmT_               # -- Begin function _ZN2t6nwIiEEPvmT_
	.p2align	4
	.type	_ZN2t6nwIiEEPvmT_,@function
_ZN2t6nwIiEEPvmT_:                      # @_ZN2t6nwIiEEPvmT_
.Lfunc_begin90:
	.loc	1 59 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:59:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	.loc	1 59 0 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:59:0
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Lfunc_end90:
	.size	_ZN2t6nwIiEEPvmT_, .Lfunc_end90-_ZN2t6nwIiEEPvmT_
	.cfi_endproc
	.file	2 "/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/x86_64-redhat-linux/bits" "c++config.h" md5 0x9e5d800a0ad50a6623343c536b5593c0
                                        # -- End function
	.section	.text._ZN2t6naIiEEPvmT_,"axG",@progbits,_ZN2t6naIiEEPvmT_,comdat
	.weak	_ZN2t6naIiEEPvmT_               # -- Begin function _ZN2t6naIiEEPvmT_
	.p2align	4
	.type	_ZN2t6naIiEEPvmT_,@function
_ZN2t6naIiEEPvmT_:                      # @_ZN2t6naIiEEPvmT_
.Lfunc_begin91:
	.loc	1 63 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:63:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	.loc	1 63 0 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:63:0
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Lfunc_end91:
	.size	_ZN2t6naIiEEPvmT_, .Lfunc_end91-_ZN2t6naIiEEPvmT_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6dlIiEEvPvT_,"axG",@progbits,_ZN2t6dlIiEEvPvT_,comdat
	.weak	_ZN2t6dlIiEEvPvT_               # -- Begin function _ZN2t6dlIiEEvPvT_
	.p2align	4
	.type	_ZN2t6dlIiEEvPvT_,@function
_ZN2t6dlIiEEvPvT_:                      # @_ZN2t6dlIiEEvPvT_
.Lfunc_begin92:
	.loc	1 62 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:62:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp180:
	.loc	1 62 58 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:62:58
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp181:
.Lfunc_end92:
	.size	_ZN2t6dlIiEEvPvT_, .Lfunc_end92-_ZN2t6dlIiEEvPvT_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6daIiEEvPvT_,"axG",@progbits,_ZN2t6daIiEEvPvT_,comdat
	.weak	_ZN2t6daIiEEvPvT_               # -- Begin function _ZN2t6daIiEEvPvT_
	.p2align	4
	.type	_ZN2t6daIiEEvPvT_,@function
_ZN2t6daIiEEvPvT_:                      # @_ZN2t6daIiEEvPvT_
.Lfunc_begin93:
	.loc	1 66 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:66:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp182:
	.loc	1 66 60 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:66:60
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp183:
.Lfunc_end93:
	.size	_ZN2t6daIiEEvPvT_, .Lfunc_end93-_ZN2t6daIiEEvPvT_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6awIiEEiv,"axG",@progbits,_ZN2t6awIiEEiv,comdat
	.weak	_ZN2t6awIiEEiv                  # -- Begin function _ZN2t6awIiEEiv
	.p2align	4
	.type	_ZN2t6awIiEEiv,@function
_ZN2t6awIiEEiv:                         # @_ZN2t6awIiEEiv
.Lfunc_begin94:
	.loc	1 67 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:67:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	.loc	1 67 0 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:67:0
	movq	%rdi, -8(%rbp)
.Lfunc_end94:
	.size	_ZN2t6awIiEEiv, .Lfunc_end94-_ZN2t6awIiEEiv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4                               # -- Begin function _Z2f1IJZ4mainE2t7EEvv
	.type	_Z2f1IJZ4mainE2t7EEvv,@function
_Z2f1IJZ4mainE2t7EEvv:                  # @_Z2f1IJZ4mainE2t7EEvv
.Lfunc_begin95:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp184:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp185:
.Lfunc_end95:
	.size	_Z2f1IJZ4mainE2t7EEvv, .Lfunc_end95-_Z2f1IJZ4mainE2t7EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJRA3_iEEvv,"axG",@progbits,_Z2f1IJRA3_iEEvv,comdat
	.weak	_Z2f1IJRA3_iEEvv                # -- Begin function _Z2f1IJRA3_iEEvv
	.p2align	4
	.type	_Z2f1IJRA3_iEEvv,@function
_Z2f1IJRA3_iEEvv:                       # @_Z2f1IJRA3_iEEvv
.Lfunc_begin96:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp186:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp187:
.Lfunc_end96:
	.size	_Z2f1IJRA3_iEEvv, .Lfunc_end96-_Z2f1IJRA3_iEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPA3_iEEvv,"axG",@progbits,_Z2f1IJPA3_iEEvv,comdat
	.weak	_Z2f1IJPA3_iEEvv                # -- Begin function _Z2f1IJPA3_iEEvv
	.p2align	4
	.type	_Z2f1IJPA3_iEEvv,@function
_Z2f1IJPA3_iEEvv:                       # @_Z2f1IJPA3_iEEvv
.Lfunc_begin97:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp188:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp189:
.Lfunc_end97:
	.size	_Z2f1IJPA3_iEEvv, .Lfunc_end97-_Z2f1IJPA3_iEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f7I2t1Evv,"axG",@progbits,_Z2f7I2t1Evv,comdat
	.weak	_Z2f7I2t1Evv                    # -- Begin function _Z2f7I2t1Evv
	.p2align	4
	.type	_Z2f7I2t1Evv,@function
_Z2f7I2t1Evv:                           # @_Z2f7I2t1Evv
.Lfunc_begin98:
	.loc	1 70 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:70:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp190:
	.loc	1 70 54 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:70:54
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp191:
.Lfunc_end98:
	.size	_Z2f7I2t1Evv, .Lfunc_end98-_Z2f7I2t1Evv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f8I2t1iEvv,"axG",@progbits,_Z2f8I2t1iEvv,comdat
	.weak	_Z2f8I2t1iEvv                   # -- Begin function _Z2f8I2t1iEvv
	.p2align	4
	.type	_Z2f8I2t1iEvv,@function
_Z2f8I2t1iEvv:                          # @_Z2f8I2t1iEvv
.Lfunc_begin99:
	.loc	1 71 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:71:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp192:
	.loc	1 71 67 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:71:67
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp193:
.Lfunc_end99:
	.size	_Z2f8I2t1iEvv, .Lfunc_end99-_Z2f8I2t1iEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2ns8ttp_userINS_5inner3ttpEEEvv,"axG",@progbits,_ZN2ns8ttp_userINS_5inner3ttpEEEvv,comdat
	.weak	_ZN2ns8ttp_userINS_5inner3ttpEEEvv # -- Begin function _ZN2ns8ttp_userINS_5inner3ttpEEEvv
	.p2align	4
	.type	_ZN2ns8ttp_userINS_5inner3ttpEEEvv,@function
_ZN2ns8ttp_userINS_5inner3ttpEEEvv:     # @_ZN2ns8ttp_userINS_5inner3ttpEEEvv
.Lfunc_begin100:
	.loc	1 12 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:12:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp194:
	.loc	1 12 57 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:12:57
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp195:
.Lfunc_end100:
	.size	_ZN2ns8ttp_userINS_5inner3ttpEEEvv, .Lfunc_end100-_ZN2ns8ttp_userINS_5inner3ttpEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPiPDnEEvv,"axG",@progbits,_Z2f1IJPiPDnEEvv,comdat
	.weak	_Z2f1IJPiPDnEEvv                # -- Begin function _Z2f1IJPiPDnEEvv
	.p2align	4
	.type	_Z2f1IJPiPDnEEvv,@function
_Z2f1IJPiPDnEEvv:                       # @_Z2f1IJPiPDnEEvv
.Lfunc_begin101:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp196:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp197:
.Lfunc_end101:
	.size	_Z2f1IJPiPDnEEvv, .Lfunc_end101-_Z2f1IJPiPDnEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJ2t7IiEEEvv,"axG",@progbits,_Z2f1IJ2t7IiEEEvv,comdat
	.weak	_Z2f1IJ2t7IiEEEvv               # -- Begin function _Z2f1IJ2t7IiEEEvv
	.p2align	4
	.type	_Z2f1IJ2t7IiEEEvv,@function
_Z2f1IJ2t7IiEEEvv:                      # @_Z2f1IJ2t7IiEEEvv
.Lfunc_begin102:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp198:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp199:
.Lfunc_end102:
	.size	_Z2f1IJ2t7IiEEEvv, .Lfunc_end102-_Z2f1IJ2t7IiEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f7ITtTpTyEN2ns3inl2t9EEvv,"axG",@progbits,_Z2f7ITtTpTyEN2ns3inl2t9EEvv,comdat
	.weak	_Z2f7ITtTpTyEN2ns3inl2t9EEvv    # -- Begin function _Z2f7ITtTpTyEN2ns3inl2t9EEvv
	.p2align	4
	.type	_Z2f7ITtTpTyEN2ns3inl2t9EEvv,@function
_Z2f7ITtTpTyEN2ns3inl2t9EEvv:           # @_Z2f7ITtTpTyEN2ns3inl2t9EEvv
.Lfunc_begin103:
	.loc	1 70 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:70:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp200:
	.loc	1 70 54 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:70:54
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp201:
.Lfunc_end103:
	.size	_Z2f7ITtTpTyEN2ns3inl2t9EEvv, .Lfunc_end103-_Z2f7ITtTpTyEN2ns3inl2t9EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJU7_AtomiciEEvv,"axG",@progbits,_Z2f1IJU7_AtomiciEEvv,comdat
	.weak	_Z2f1IJU7_AtomiciEEvv           # -- Begin function _Z2f1IJU7_AtomiciEEvv
	.p2align	4
	.type	_Z2f1IJU7_AtomiciEEvv,@function
_Z2f1IJU7_AtomiciEEvv:                  # @_Z2f1IJU7_AtomiciEEvv
.Lfunc_begin104:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp202:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp203:
.Lfunc_end104:
	.size	_Z2f1IJU7_AtomiciEEvv, .Lfunc_end104-_Z2f1IJU7_AtomiciEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJilVcEEvv,"axG",@progbits,_Z2f1IJilVcEEvv,comdat
	.weak	_Z2f1IJilVcEEvv                 # -- Begin function _Z2f1IJilVcEEvv
	.p2align	4
	.type	_Z2f1IJilVcEEvv,@function
_Z2f1IJilVcEEvv:                        # @_Z2f1IJilVcEEvv
.Lfunc_begin105:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp204:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp205:
.Lfunc_end105:
	.size	_Z2f1IJilVcEEvv, .Lfunc_end105-_Z2f1IJilVcEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJDv2_iEEvv,"axG",@progbits,_Z2f1IJDv2_iEEvv,comdat
	.weak	_Z2f1IJDv2_iEEvv                # -- Begin function _Z2f1IJDv2_iEEvv
	.p2align	4
	.type	_Z2f1IJDv2_iEEvv,@function
_Z2f1IJDv2_iEEvv:                       # @_Z2f1IJDv2_iEEvv
.Lfunc_begin106:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp206:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp207:
.Lfunc_end106:
	.size	_Z2f1IJDv2_iEEvv, .Lfunc_end106-_Z2f1IJDv2_iEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJVKPiEEvv,"axG",@progbits,_Z2f1IJVKPiEEvv,comdat
	.weak	_Z2f1IJVKPiEEvv                 # -- Begin function _Z2f1IJVKPiEEvv
	.p2align	4
	.type	_Z2f1IJVKPiEEvv,@function
_Z2f1IJVKPiEEvv:                        # @_Z2f1IJVKPiEEvv
.Lfunc_begin107:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp208:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp209:
.Lfunc_end107:
	.size	_Z2f1IJVKPiEEvv, .Lfunc_end107-_Z2f1IJVKPiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJVKvEEvv,"axG",@progbits,_Z2f1IJVKvEEvv,comdat
	.weak	_Z2f1IJVKvEEvv                  # -- Begin function _Z2f1IJVKvEEvv
	.p2align	4
	.type	_Z2f1IJVKvEEvv,@function
_Z2f1IJVKvEEvv:                         # @_Z2f1IJVKvEEvv
.Lfunc_begin108:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp210:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp211:
.Lfunc_end108:
	.size	_Z2f1IJVKvEEvv, .Lfunc_end108-_Z2f1IJVKvEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4                               # -- Begin function _Z2f1IJ2t1IJZ4mainE3$_0EEEEvv
	.type	_Z2f1IJ2t1IJZ4mainE3$_0EEEEvv,@function
_Z2f1IJ2t1IJZ4mainE3$_0EEEEvv:          # @"_Z2f1IJ2t1IJZ4mainE3$_0EEEEvv"
.Lfunc_begin109:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp212:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp213:
.Lfunc_end109:
	.size	_Z2f1IJ2t1IJZ4mainE3$_0EEEEvv, .Lfunc_end109-_Z2f1IJ2t1IJZ4mainE3$_0EEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3t10C2IvEEv,"axG",@progbits,_ZN3t10C2IvEEv,comdat
	.weak	_ZN3t10C2IvEEv                  # -- Begin function _ZN3t10C2IvEEv
	.p2align	4
	.type	_ZN3t10C2IvEEv,@function
_ZN3t10C2IvEEv:                         # @_ZN3t10C2IvEEv
.Lfunc_begin110:
	.loc	1 85 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:85:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp214:
	.loc	1 85 39 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:85:39
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp215:
.Lfunc_end110:
	.size	_ZN3t10C2IvEEv, .Lfunc_end110-_ZN3t10C2IvEEv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJM3udtKFvvEEEvv,"axG",@progbits,_Z2f1IJM3udtKFvvEEEvv,comdat
	.weak	_Z2f1IJM3udtKFvvEEEvv           # -- Begin function _Z2f1IJM3udtKFvvEEEvv
	.p2align	4
	.type	_Z2f1IJM3udtKFvvEEEvv,@function
_Z2f1IJM3udtKFvvEEEvv:                  # @_Z2f1IJM3udtKFvvEEEvv
.Lfunc_begin111:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp216:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp217:
.Lfunc_end111:
	.size	_Z2f1IJM3udtKFvvEEEvv, .Lfunc_end111-_Z2f1IJM3udtKFvvEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJM3udtVFvvREEEvv,"axG",@progbits,_Z2f1IJM3udtVFvvREEEvv,comdat
	.weak	_Z2f1IJM3udtVFvvREEEvv          # -- Begin function _Z2f1IJM3udtVFvvREEEvv
	.p2align	4
	.type	_Z2f1IJM3udtVFvvREEEvv,@function
_Z2f1IJM3udtVFvvREEEvv:                 # @_Z2f1IJM3udtVFvvREEEvv
.Lfunc_begin112:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp218:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp219:
.Lfunc_end112:
	.size	_Z2f1IJM3udtVFvvREEEvv, .Lfunc_end112-_Z2f1IJM3udtVFvvREEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJM3udtVKFvvOEEEvv,"axG",@progbits,_Z2f1IJM3udtVKFvvOEEEvv,comdat
	.weak	_Z2f1IJM3udtVKFvvOEEEvv         # -- Begin function _Z2f1IJM3udtVKFvvOEEEvv
	.p2align	4
	.type	_Z2f1IJM3udtVKFvvOEEEvv,@function
_Z2f1IJM3udtVKFvvOEEEvv:                # @_Z2f1IJM3udtVKFvvOEEEvv
.Lfunc_begin113:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp220:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp221:
.Lfunc_end113:
	.size	_Z2f1IJM3udtVKFvvOEEEvv, .Lfunc_end113-_Z2f1IJM3udtVKFvvOEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f9IiEPFvvEv,"axG",@progbits,_Z2f9IiEPFvvEv,comdat
	.weak	_Z2f9IiEPFvvEv                  # -- Begin function _Z2f9IiEPFvvEv
	.p2align	4
	.type	_Z2f9IiEPFvvEv,@function
_Z2f9IiEPFvvEv:                         # @_Z2f9IiEPFvvEv
.Lfunc_begin114:
	.loc	1 83 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:83:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp222:
	.loc	1 83 40 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:83:40
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	.loc	1 83 40 epilogue_begin is_stmt 0 # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:83:40
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp223:
.Lfunc_end114:
	.size	_Z2f9IiEPFvvEv, .Lfunc_end114-_Z2f9IiEPFvvEv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJKPFvvEEEvv,"axG",@progbits,_Z2f1IJKPFvvEEEvv,comdat
	.weak	_Z2f1IJKPFvvEEEvv               # -- Begin function _Z2f1IJKPFvvEEEvv
	.p2align	4
	.type	_Z2f1IJKPFvvEEEvv,@function
_Z2f1IJKPFvvEEEvv:                      # @_Z2f1IJKPFvvEEEvv
.Lfunc_begin115:
	.loc	1 18 0 is_stmt 1                # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp224:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp225:
.Lfunc_end115:
	.size	_Z2f1IJKPFvvEEEvv, .Lfunc_end115-_Z2f1IJKPFvvEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJRA1_KcEEvv,"axG",@progbits,_Z2f1IJRA1_KcEEvv,comdat
	.weak	_Z2f1IJRA1_KcEEvv               # -- Begin function _Z2f1IJRA1_KcEEvv
	.p2align	4
	.type	_Z2f1IJRA1_KcEEvv,@function
_Z2f1IJRA1_KcEEvv:                      # @_Z2f1IJRA1_KcEEvv
.Lfunc_begin116:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp226:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp227:
.Lfunc_end116:
	.size	_Z2f1IJRA1_KcEEvv, .Lfunc_end116-_Z2f1IJRA1_KcEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJKFvvREEEvv,"axG",@progbits,_Z2f1IJKFvvREEEvv,comdat
	.weak	_Z2f1IJKFvvREEEvv               # -- Begin function _Z2f1IJKFvvREEEvv
	.p2align	4
	.type	_Z2f1IJKFvvREEEvv,@function
_Z2f1IJKFvvREEEvv:                      # @_Z2f1IJKFvvREEEvv
.Lfunc_begin117:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp228:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp229:
.Lfunc_end117:
	.size	_Z2f1IJKFvvREEEvv, .Lfunc_end117-_Z2f1IJKFvvREEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJVFvvOEEEvv,"axG",@progbits,_Z2f1IJVFvvOEEEvv,comdat
	.weak	_Z2f1IJVFvvOEEEvv               # -- Begin function _Z2f1IJVFvvOEEEvv
	.p2align	4
	.type	_Z2f1IJVFvvOEEEvv,@function
_Z2f1IJVFvvOEEEvv:                      # @_Z2f1IJVFvvOEEEvv
.Lfunc_begin118:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp230:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp231:
.Lfunc_end118:
	.size	_Z2f1IJVFvvOEEEvv, .Lfunc_end118-_Z2f1IJVFvvOEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJVKFvvEEEvv,"axG",@progbits,_Z2f1IJVKFvvEEEvv,comdat
	.weak	_Z2f1IJVKFvvEEEvv               # -- Begin function _Z2f1IJVKFvvEEEvv
	.p2align	4
	.type	_Z2f1IJVKFvvEEEvv,@function
_Z2f1IJVKFvvEEEvv:                      # @_Z2f1IJVKFvvEEEvv
.Lfunc_begin119:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp232:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp233:
.Lfunc_end119:
	.size	_Z2f1IJVKFvvEEEvv, .Lfunc_end119-_Z2f1IJVKFvvEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJA1_KPiEEvv,"axG",@progbits,_Z2f1IJA1_KPiEEvv,comdat
	.weak	_Z2f1IJA1_KPiEEvv               # -- Begin function _Z2f1IJA1_KPiEEvv
	.p2align	4
	.type	_Z2f1IJA1_KPiEEvv,@function
_Z2f1IJA1_KPiEEvv:                      # @_Z2f1IJA1_KPiEEvv
.Lfunc_begin120:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp234:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp235:
.Lfunc_end120:
	.size	_Z2f1IJA1_KPiEEvv, .Lfunc_end120-_Z2f1IJA1_KPiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJRA1_KPiEEvv,"axG",@progbits,_Z2f1IJRA1_KPiEEvv,comdat
	.weak	_Z2f1IJRA1_KPiEEvv              # -- Begin function _Z2f1IJRA1_KPiEEvv
	.p2align	4
	.type	_Z2f1IJRA1_KPiEEvv,@function
_Z2f1IJRA1_KPiEEvv:                     # @_Z2f1IJRA1_KPiEEvv
.Lfunc_begin121:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp236:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp237:
.Lfunc_end121:
	.size	_Z2f1IJRA1_KPiEEvv, .Lfunc_end121-_Z2f1IJRA1_KPiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJRKM3udtFvvEEEvv,"axG",@progbits,_Z2f1IJRKM3udtFvvEEEvv,comdat
	.weak	_Z2f1IJRKM3udtFvvEEEvv          # -- Begin function _Z2f1IJRKM3udtFvvEEEvv
	.p2align	4
	.type	_Z2f1IJRKM3udtFvvEEEvv,@function
_Z2f1IJRKM3udtFvvEEEvv:                 # @_Z2f1IJRKM3udtFvvEEEvv
.Lfunc_begin122:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp238:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp239:
.Lfunc_end122:
	.size	_Z2f1IJRKM3udtFvvEEEvv, .Lfunc_end122-_Z2f1IJRKM3udtFvvEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJFPFvfEiEEEvv,"axG",@progbits,_Z2f1IJFPFvfEiEEEvv,comdat
	.weak	_Z2f1IJFPFvfEiEEEvv             # -- Begin function _Z2f1IJFPFvfEiEEEvv
	.p2align	4
	.type	_Z2f1IJFPFvfEiEEEvv,@function
_Z2f1IJFPFvfEiEEEvv:                    # @_Z2f1IJFPFvfEiEEEvv
.Lfunc_begin123:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp240:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp241:
.Lfunc_end123:
	.size	_Z2f1IJFPFvfEiEEEvv, .Lfunc_end123-_Z2f1IJFPFvfEiEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJA1_2t1IJiEEEEvv,"axG",@progbits,_Z2f1IJA1_2t1IJiEEEEvv,comdat
	.weak	_Z2f1IJA1_2t1IJiEEEEvv          # -- Begin function _Z2f1IJA1_2t1IJiEEEEvv
	.p2align	4
	.type	_Z2f1IJA1_2t1IJiEEEEvv,@function
_Z2f1IJA1_2t1IJiEEEEvv:                 # @_Z2f1IJA1_2t1IJiEEEEvv
.Lfunc_begin124:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp242:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp243:
.Lfunc_end124:
	.size	_Z2f1IJA1_2t1IJiEEEEvv, .Lfunc_end124-_Z2f1IJA1_2t1IJiEEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPDoFvvEEEvv,"axG",@progbits,_Z2f1IJPDoFvvEEEvv,comdat
	.weak	_Z2f1IJPDoFvvEEEvv              # -- Begin function _Z2f1IJPDoFvvEEEvv
	.p2align	4
	.type	_Z2f1IJPDoFvvEEEvv,@function
_Z2f1IJPDoFvvEEEvv:                     # @_Z2f1IJPDoFvvEEEvv
.Lfunc_begin125:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp244:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp245:
.Lfunc_end125:
	.size	_Z2f1IJPDoFvvEEEvv, .Lfunc_end125-_Z2f1IJPDoFvvEEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4                               # -- Begin function _Z2f1IJFvZ4mainE3$_1EEEvv
	.type	_Z2f1IJFvZ4mainE3$_1EEEvv,@function
_Z2f1IJFvZ4mainE3$_1EEEvv:              # @"_Z2f1IJFvZ4mainE3$_1EEEvv"
.Lfunc_begin126:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp246:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp247:
.Lfunc_end126:
	.size	_Z2f1IJFvZ4mainE3$_1EEEvv, .Lfunc_end126-_Z2f1IJFvZ4mainE3$_1EEEvv
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function _Z2f1IJFvZ4mainE2t8Z4mainE3$_1EEEvv
	.type	_Z2f1IJFvZ4mainE2t8Z4mainE3$_1EEEvv,@function
_Z2f1IJFvZ4mainE2t8Z4mainE3$_1EEEvv:    # @"_Z2f1IJFvZ4mainE2t8Z4mainE3$_1EEEvv"
.Lfunc_begin127:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp248:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp249:
.Lfunc_end127:
	.size	_Z2f1IJFvZ4mainE2t8Z4mainE3$_1EEEvv, .Lfunc_end127-_Z2f1IJFvZ4mainE2t8Z4mainE3$_1EEEvv
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function _Z2f1IJFvZ4mainE2t8EEEvv
	.type	_Z2f1IJFvZ4mainE2t8EEEvv,@function
_Z2f1IJFvZ4mainE2t8EEEvv:               # @_Z2f1IJFvZ4mainE2t8EEEvv
.Lfunc_begin128:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp250:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp251:
.Lfunc_end128:
	.size	_Z2f1IJFvZ4mainE2t8EEEvv, .Lfunc_end128-_Z2f1IJFvZ4mainE2t8EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z19operator_not_reallyIiEvv,"axG",@progbits,_Z19operator_not_reallyIiEvv,comdat
	.weak	_Z19operator_not_reallyIiEvv    # -- Begin function _Z19operator_not_reallyIiEvv
	.p2align	4
	.type	_Z19operator_not_reallyIiEvv,@function
_Z19operator_not_reallyIiEvv:           # @_Z19operator_not_reallyIiEvv
.Lfunc_begin129:
	.loc	1 88 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:88:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp252:
	.loc	1 88 51 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:88:51
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp253:
.Lfunc_end129:
	.size	_Z19operator_not_reallyIiEvv, .Lfunc_end129-_Z19operator_not_reallyIiEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z3f11IDB3_TnT_LS0_2EEvv,"axG",@progbits,_Z3f11IDB3_TnT_LS0_2EEvv,comdat
	.weak	_Z3f11IDB3_TnT_LS0_2EEvv        # -- Begin function _Z3f11IDB3_TnT_LS0_2EEvv
	.p2align	4
	.type	_Z3f11IDB3_TnT_LS0_2EEvv,@function
_Z3f11IDB3_TnT_LS0_2EEvv:               # @_Z3f11IDB3_TnT_LS0_2EEvv
.Lfunc_begin130:
	.loc	1 98 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:98:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp254:
	.loc	1 98 40 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:98:40
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp255:
.Lfunc_end130:
	.size	_Z3f11IDB3_TnT_LS0_2EEvv, .Lfunc_end130-_Z3f11IDB3_TnT_LS0_2EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z3f11IKDU5_TnT_LS0_2EEvv,"axG",@progbits,_Z3f11IKDU5_TnT_LS0_2EEvv,comdat
	.weak	_Z3f11IKDU5_TnT_LS0_2EEvv       # -- Begin function _Z3f11IKDU5_TnT_LS0_2EEvv
	.p2align	4
	.type	_Z3f11IKDU5_TnT_LS0_2EEvv,@function
_Z3f11IKDU5_TnT_LS0_2EEvv:              # @_Z3f11IKDU5_TnT_LS0_2EEvv
.Lfunc_begin131:
	.loc	1 98 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:98:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp256:
	.loc	1 98 40 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:98:40
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp257:
.Lfunc_end131:
	.size	_Z3f11IKDU5_TnT_LS0_2EEvv, .Lfunc_end131-_Z3f11IKDU5_TnT_LS0_2EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z3f11IDB65_TnT_LS0_2EEvv,"axG",@progbits,_Z3f11IDB65_TnT_LS0_2EEvv,comdat
	.weak	_Z3f11IDB65_TnT_LS0_2EEvv       # -- Begin function _Z3f11IDB65_TnT_LS0_2EEvv
	.p2align	4
	.type	_Z3f11IDB65_TnT_LS0_2EEvv,@function
_Z3f11IDB65_TnT_LS0_2EEvv:              # @_Z3f11IDB65_TnT_LS0_2EEvv
.Lfunc_begin132:
	.loc	1 98 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:98:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp258:
	.loc	1 98 40 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:98:40
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp259:
.Lfunc_end132:
	.size	_Z3f11IDB65_TnT_LS0_2EEvv, .Lfunc_end132-_Z3f11IDB65_TnT_LS0_2EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z3f11IKDU65_TnT_LS0_2EEvv,"axG",@progbits,_Z3f11IKDU65_TnT_LS0_2EEvv,comdat
	.weak	_Z3f11IKDU65_TnT_LS0_2EEvv      # -- Begin function _Z3f11IKDU65_TnT_LS0_2EEvv
	.p2align	4
	.type	_Z3f11IKDU65_TnT_LS0_2EEvv,@function
_Z3f11IKDU65_TnT_LS0_2EEvv:             # @_Z3f11IKDU65_TnT_LS0_2EEvv
.Lfunc_begin133:
	.loc	1 98 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:98:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp260:
	.loc	1 98 40 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:98:40
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp261:
.Lfunc_end133:
	.size	_Z3f11IKDU65_TnT_LS0_2EEvv, .Lfunc_end133-_Z3f11IKDU65_TnT_LS0_2EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJFv2t1IJEES1_EEEvv,"axG",@progbits,_Z2f1IJFv2t1IJEES1_EEEvv,comdat
	.weak	_Z2f1IJFv2t1IJEES1_EEEvv        # -- Begin function _Z2f1IJFv2t1IJEES1_EEEvv
	.p2align	4
	.type	_Z2f1IJFv2t1IJEES1_EEEvv,@function
_Z2f1IJFv2t1IJEES1_EEEvv:               # @_Z2f1IJFv2t1IJEES1_EEEvv
.Lfunc_begin134:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp262:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp263:
.Lfunc_end134:
	.size	_Z2f1IJFv2t1IJEES1_EEEvv, .Lfunc_end134-_Z2f1IJFv2t1IJEES1_EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJM2t1IJEEiEEvv,"axG",@progbits,_Z2f1IJM2t1IJEEiEEvv,comdat
	.weak	_Z2f1IJM2t1IJEEiEEvv            # -- Begin function _Z2f1IJM2t1IJEEiEEvv
	.p2align	4
	.type	_Z2f1IJM2t1IJEEiEEvv,@function
_Z2f1IJM2t1IJEEiEEvv:                   # @_Z2f1IJM2t1IJEEiEEvv
.Lfunc_begin135:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp264:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp265:
.Lfunc_end135:
	.size	_Z2f1IJM2t1IJEEiEEvv, .Lfunc_end135-_Z2f1IJM2t1IJEEiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJU9swiftcallFvvEEEvv,"axG",@progbits,_Z2f1IJU9swiftcallFvvEEEvv,comdat
	.weak	_Z2f1IJU9swiftcallFvvEEEvv      # -- Begin function _Z2f1IJU9swiftcallFvvEEEvv
	.p2align	4
	.type	_Z2f1IJU9swiftcallFvvEEEvv,@function
_Z2f1IJU9swiftcallFvvEEEvv:             # @_Z2f1IJU9swiftcallFvvEEEvv
.Lfunc_begin136:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp266:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp267:
.Lfunc_end136:
	.size	_Z2f1IJU9swiftcallFvvEEEvv, .Lfunc_end136-_Z2f1IJU9swiftcallFvvEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJFivEEEvv,"axG",@progbits,_Z2f1IJFivEEEvv,comdat
	.weak	_Z2f1IJFivEEEvv                 # -- Begin function _Z2f1IJFivEEEvv
	.p2align	4
	.type	_Z2f1IJFivEEEvv,@function
_Z2f1IJFivEEEvv:                        # @_Z2f1IJFivEEEvv
.Lfunc_begin137:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp268:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp269:
.Lfunc_end137:
	.size	_Z2f1IJFivEEEvv, .Lfunc_end137-_Z2f1IJFivEEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4                               # -- Begin function _Z3f10ILN2ns3$_0E0EEvv
	.type	_Z3f10ILN2ns3$_0E0EEvv,@function
_Z3f10ILN2ns3$_0E0EEvv:                 # @"_Z3f10ILN2ns3$_0E0EEvv"
.Lfunc_begin138:
	.loc	1 96 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:96:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp270:
	.loc	1 96 48 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:96:48
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp271:
.Lfunc_end138:
	.size	_Z3f10ILN2ns3$_0E0EEvv, .Lfunc_end138-_Z3f10ILN2ns3$_0E0EEvv
	.cfi_endproc
                                        # -- End function
	.globl	_ZN2t83memEv                    # -- Begin function _ZN2t83memEv
	.p2align	4
	.type	_ZN2t83memEv,@function
_ZN2t83memEv:                           # @_ZN2t83memEv
.Lfunc_begin139:
	.loc	1 256 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:256:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
.Ltmp272:
	.loc	1 258 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:258:3
	callq	_Z2f1IJZN2t83memEvE2t7EEvv
	.loc	1 259 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:259:3
	callq	_Z2f1IJM2t8FvvEEEvv
	.loc	1 260 1 epilogue_begin          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:260:1
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp273:
.Lfunc_end139:
	.size	_ZN2t83memEv, .Lfunc_end139-_ZN2t83memEv
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function _Z2f1IJZN2t83memEvE2t7EEvv
	.type	_Z2f1IJZN2t83memEvE2t7EEvv,@function
_Z2f1IJZN2t83memEvE2t7EEvv:             # @_Z2f1IJZN2t83memEvE2t7EEvv
.Lfunc_begin140:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp274:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp275:
.Lfunc_end140:
	.size	_Z2f1IJZN2t83memEvE2t7EEvv, .Lfunc_end140-_Z2f1IJZN2t83memEvE2t7EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJM2t8FvvEEEvv,"axG",@progbits,_Z2f1IJM2t8FvvEEEvv,comdat
	.weak	_Z2f1IJM2t8FvvEEEvv             # -- Begin function _Z2f1IJM2t8FvvEEEvv
	.p2align	4
	.type	_Z2f1IJM2t8FvvEEEvv,@function
_Z2f1IJM2t8FvvEEEvv:                    # @_Z2f1IJM2t8FvvEEEvv
.Lfunc_begin141:
	.loc	1 18 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:18:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp276:
	.loc	1 21 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:21:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp277:
.Lfunc_end141:
	.size	_Z2f1IJM2t8FvvEEEvv, .Lfunc_end141-_Z2f1IJM2t8FvvEEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.globl	_ZN18complex_type_units2f1Ev    # -- Begin function _ZN18complex_type_units2f1Ev
	.p2align	4
	.type	_ZN18complex_type_units2f1Ev,@function
_ZN18complex_type_units2f1Ev:           # @_ZN18complex_type_units2f1Ev
.Lfunc_begin142:
	.loc	1 272 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:272:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp278:
	.loc	1 275 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:275:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp279:
.Lfunc_end142:
	.size	_ZN18complex_type_units2f1Ev, .Lfunc_end142-_ZN18complex_type_units2f1Ev
	.cfi_endproc
                                        # -- End function
	.globl	_ZN18ptr_to_member_test4testEv  # -- Begin function _ZN18ptr_to_member_test4testEv
	.p2align	4
	.type	_ZN18ptr_to_member_test4testEv,@function
_ZN18ptr_to_member_test4testEv:         # @_ZN18ptr_to_member_test4testEv
.Lfunc_begin143:
	.loc	1 284 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:284:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp280:
	.loc	1 284 15 prologue_end           # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:284:15
	callq	_ZN18ptr_to_member_test1fIXadL_ZNS_1S8data_memEEEEEvv
	.loc	1 284 34 epilogue_begin is_stmt 0 # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:284:34
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp281:
.Lfunc_end143:
	.size	_ZN18ptr_to_member_test4testEv, .Lfunc_end143-_ZN18ptr_to_member_test4testEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN18ptr_to_member_test1fIXadL_ZNS_1S8data_memEEEEEvv,"axG",@progbits,_ZN18ptr_to_member_test1fIXadL_ZNS_1S8data_memEEEEEvv,comdat
	.weak	_ZN18ptr_to_member_test1fIXadL_ZNS_1S8data_memEEEEEvv # -- Begin function _ZN18ptr_to_member_test1fIXadL_ZNS_1S8data_memEEEEEvv
	.p2align	4
	.type	_ZN18ptr_to_member_test1fIXadL_ZNS_1S8data_memEEEEEvv,@function
_ZN18ptr_to_member_test1fIXadL_ZNS_1S8data_memEEEEEvv: # @_ZN18ptr_to_member_test1fIXadL_ZNS_1S8data_memEEEEEvv
.Lfunc_begin144:
	.loc	1 283 0 is_stmt 1               # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:283:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp282:
	.loc	1 283 32 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:283:32
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp283:
.Lfunc_end144:
	.size	_ZN18ptr_to_member_test1fIXadL_ZNS_1S8data_memEEEEEvv, .Lfunc_end144-_ZN18ptr_to_member_test1fIXadL_ZNS_1S8data_memEEEEEvv
	.cfi_endproc
                                        # -- End function
	.type	i,@object                       # @i
	.data
	.globl	i
	.p2align	2, 0x0
i:
	.long	3                               # 0x3
	.size	i, 4

	.type	.L__const.main.L,@object        # @__const.main.L
	.section	.rodata,"a",@progbits
.L__const.main.L:
	.zero	1
	.size	.L__const.main.L, 1

	.file	3 "/opt/llvm/stable/Toolchains/llvm-sand.xctoolchain/usr/lib/clang/15/include" "__stddef_max_align_t.h" md5 0x3c0a2f19d136d39aa835c737c7105def
	.file	4 "/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11" "cstddef"
	.file	5 "/usr/include/bits" "types.h" md5 0x58b79843d97f4309eefa4aa722dac91e
	.file	6 "/usr/include/bits" "stdint-intn.h" md5 0xb26974ec56196748bbc399ee826d2a0e
	.file	7 "/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11" "cstdint"
	.file	8 "/usr/include" "stdint.h" md5 0x8e56ab3ccd56760d8ae9848ebf326071
	.file	9 "/usr/include/bits" "stdint-uintn.h" md5 0x3d2fbc5d847dd222c2fbd70457568436
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	37                              # DW_FORM_strx1
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	85                              # DW_AT_ranges
	.byte	35                              # DW_FORM_rnglistx
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	116                             # DW_AT_rnglists_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	57                              # DW_TAG_namespace
	.byte	1                               # DW_CHILDREN_yes
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	4                               # DW_TAG_enumeration_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	40                              # DW_TAG_enumerator
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	28                              # DW_AT_const_value
	.byte	15                              # DW_FORM_udata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	57                              # DW_TAG_namespace
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	40                              # DW_TAG_enumerator
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	28                              # DW_AT_const_value
	.byte	13                              # DW_FORM_sdata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	4                               # DW_TAG_enumeration_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	109                             # DW_AT_enum_class
	.byte	25                              # DW_FORM_flag_present
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.byte	4                               # DW_TAG_enumeration_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	12                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	38                              # DW_FORM_strx2
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	13                              # Abbreviation Code
	.ascii	"\206\202\001"                  # DW_TAG_GNU_template_template_param
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.ascii	"\220B"                         # DW_AT_GNU_template_name
	.byte	38                              # DW_FORM_strx2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	14                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	15                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	16                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	17                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	30                              # DW_AT_default_value
	.byte	25                              # DW_FORM_flag_present
	.byte	28                              # DW_AT_const_value
	.byte	15                              # DW_FORM_udata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	18                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	19                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	20                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	52                              # DW_AT_artificial
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	21                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	22                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	23                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	24                              # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	25                              # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	26                              # Abbreviation Code
	.byte	58                              # DW_TAG_imported_module
	.byte	0                               # DW_CHILDREN_no
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	24                              # DW_AT_import
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	27                              # Abbreviation Code
	.byte	2                               # DW_TAG_class_type
	.byte	0                               # DW_CHILDREN_no
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	28                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	29                              # Abbreviation Code
	.ascii	"\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	30                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	31                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	32                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	28                              # DW_AT_const_value
	.byte	15                              # DW_FORM_udata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	33                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	28                              # DW_AT_const_value
	.byte	13                              # DW_FORM_sdata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	34                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	28                              # DW_AT_const_value
	.byte	13                              # DW_FORM_sdata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	35                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	28                              # DW_AT_const_value
	.byte	15                              # DW_FORM_udata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	36                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	37                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	38                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	28                              # DW_AT_const_value
	.byte	10                              # DW_FORM_block1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	39                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	38                              # DW_FORM_strx2
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	40                              # Abbreviation Code
	.ascii	"\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	41                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	42                              # Abbreviation Code
	.ascii	"\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	43                              # Abbreviation Code
	.ascii	"\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	44                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	45                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	46                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	47                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	100                             # DW_AT_object_pointer
	.byte	19                              # DW_FORM_ref4
	.byte	71                              # DW_AT_specification
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	48                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	52                              # DW_AT_artificial
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	49                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	50                              # Abbreviation Code
	.byte	22                              # DW_TAG_typedef
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	5                               # DW_FORM_data2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	51                              # Abbreviation Code
	.byte	8                               # DW_TAG_imported_declaration
	.byte	0                               # DW_CHILDREN_no
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	24                              # DW_AT_import
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	52                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	71                              # DW_AT_specification
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	53                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	100                             # DW_AT_object_pointer
	.byte	19                              # DW_FORM_ref4
	.byte	110                             # DW_AT_linkage_name
	.byte	38                              # DW_FORM_strx2
	.byte	71                              # DW_AT_specification
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	54                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	38                              # DW_FORM_strx2
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	55                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	28                              # DW_AT_const_value
	.byte	13                              # DW_FORM_sdata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	56                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	28                              # DW_AT_const_value
	.byte	15                              # DW_FORM_udata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	57                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	28                              # DW_AT_const_value
	.byte	10                              # DW_FORM_block1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	58                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	100                             # DW_AT_object_pointer
	.byte	19                              # DW_FORM_ref4
	.byte	59                              # DW_AT_decl_line
	.byte	5                               # DW_FORM_data2
	.byte	71                              # DW_AT_specification
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	59                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	38                              # DW_FORM_strx2
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	5                               # DW_FORM_data2
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	60                              # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	5                               # DW_FORM_data2
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	61                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	5                               # DW_FORM_data2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	62                              # Abbreviation Code
	.byte	13                              # DW_TAG_member
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	5                               # DW_FORM_data2
	.byte	56                              # DW_AT_data_member_location
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	63                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	30                              # DW_AT_default_value
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	64                              # Abbreviation Code
	.byte	2                               # DW_TAG_class_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	5                               # DW_FORM_data2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	65                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	66                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	30                              # DW_AT_default_value
	.byte	25                              # DW_FORM_flag_present
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	67                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	38                              # DW_FORM_strx2
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	5                               # DW_FORM_data2
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	68                              # Abbreviation Code
	.byte	22                              # DW_TAG_typedef
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	69                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	70                              # Abbreviation Code
	.byte	16                              # DW_TAG_reference_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	71                              # Abbreviation Code
	.byte	66                              # DW_TAG_rvalue_reference_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	72                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	73                              # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	74                              # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	55                              # DW_AT_count
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	75                              # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	76                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	77                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	78                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	79                              # Abbreviation Code
	.byte	24                              # DW_TAG_unspecified_parameters
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	80                              # Abbreviation Code
	.byte	59                              # DW_TAG_unspecified_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	81                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	82                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	83                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	84                              # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	85                              # Abbreviation Code
	.byte	71                              # DW_TAG_atomic_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	86                              # Abbreviation Code
	.byte	53                              # DW_TAG_volatile_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	87                              # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.ascii	"\207B"                         # DW_AT_GNU_vector
	.byte	25                              # DW_FORM_flag_present
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	88                              # Abbreviation Code
	.byte	53                              # DW_TAG_volatile_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	89                              # Abbreviation Code
	.byte	31                              # DW_TAG_ptr_to_member_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	29                              # DW_AT_containing_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	90                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	119                             # DW_AT_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	91                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	120                             # DW_AT_rvalue_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	92                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	119                             # DW_AT_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	93                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	120                             # DW_AT_rvalue_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	94                              # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	95                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	96                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	97                              # Abbreviation Code
	.byte	22                              # DW_TAG_typedef
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	98                              # Abbreviation Code
	.byte	13                              # DW_TAG_member
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	56                              # DW_AT_data_member_location
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x2a8d DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.quad	0                               # DW_AT_low_pc
	.byte	0                               # DW_AT_ranges
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.long	.Lrnglists_table_base0          # DW_AT_rnglists_base
	.byte	2                               # Abbrev [2] 0x2b:0xb DW_TAG_variable
	.byte	3                               # DW_AT_name
	.long	54                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	31                              # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	161
	.byte	0
	.byte	3                               # Abbrev [3] 0x36:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	4                               # Abbrev [4] 0x3a:0x12 DW_TAG_namespace
	.byte	5                               # Abbrev [5] 0x3b:0xd DW_TAG_enumeration_type
	.long	76                              # DW_AT_type
	.byte	7                               # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.byte	6                               # Abbrev [6] 0x44:0x3 DW_TAG_enumerator
	.byte	6                               # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	7                               # Abbrev [7] 0x48:0x3 DW_TAG_structure_type
	.short	258                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0x4c:0x4 DW_TAG_base_type
	.byte	5                               # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	8                               # Abbrev [8] 0x50:0x63 DW_TAG_namespace
	.byte	8                               # DW_AT_name
	.byte	5                               # Abbrev [5] 0x52:0x13 DW_TAG_enumeration_type
	.long	54                              # DW_AT_type
	.byte	12                              # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	13                              # DW_AT_decl_line
	.byte	9                               # Abbrev [9] 0x5b:0x3 DW_TAG_enumerator
	.byte	9                               # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	9                               # Abbrev [9] 0x5e:0x3 DW_TAG_enumerator
	.byte	10                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	9                               # Abbrev [9] 0x61:0x3 DW_TAG_enumerator
	.byte	11                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x65:0x13 DW_TAG_enumeration_type
	.long	54                              # DW_AT_type
                                        # DW_AT_enum_class
	.byte	13                              # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	9                               # Abbrev [9] 0x6e:0x3 DW_TAG_enumerator
	.byte	9                               # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	9                               # Abbrev [9] 0x71:0x3 DW_TAG_enumerator
	.byte	10                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	9                               # Abbrev [9] 0x74:0x3 DW_TAG_enumerator
	.byte	11                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x78:0xe DW_TAG_enumeration_type
	.long	179                             # DW_AT_type
	.byte	16                              # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	16                              # DW_AT_decl_line
	.byte	6                               # Abbrev [6] 0x81:0x4 DW_TAG_enumerator
	.byte	15                              # DW_AT_name
	.ascii	"\377\001"                      # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	11                              # Abbrev [11] 0x86:0x12 DW_TAG_enumeration_type
	.long	54                              # DW_AT_type
	.byte	4                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	15                              # DW_AT_decl_line
	.byte	9                               # Abbrev [9] 0x8e:0x3 DW_TAG_enumerator
	.byte	17                              # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	9                               # Abbrev [9] 0x91:0x3 DW_TAG_enumerator
	.byte	18                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	9                               # Abbrev [9] 0x94:0x3 DW_TAG_enumerator
	.byte	19                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x98:0x13 DW_TAG_subprogram
	.byte	101                             # DW_AT_low_pc
	.long	.Lfunc_end100-.Lfunc_begin100   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	312                             # DW_AT_linkage_name
	.short	313                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	13                              # Abbrev [13] 0xa6:0x4 DW_TAG_GNU_template_template_param
	.byte	20                              # DW_AT_name
	.short	311                             # DW_AT_GNU_template_name
	.byte	0                               # End Of Children Mark
	.byte	14                              # Abbrev [14] 0xab:0x2 DW_TAG_structure_type
	.byte	162                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	8                               # Abbrev [8] 0xad:0x5 DW_TAG_namespace
	.byte	169                             # DW_AT_name
	.byte	14                              # Abbrev [14] 0xaf:0x2 DW_TAG_structure_type
	.byte	162                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0xb3:0x4 DW_TAG_base_type
	.byte	14                              # DW_AT_name
	.byte	8                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	15                              # Abbrev [15] 0xb7:0x14 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	23                              # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	16                              # Abbrev [16] 0xbd:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	17                              # Abbrev [17] 0xc3:0x7 DW_TAG_template_value_parameter
	.long	203                             # DW_AT_type
	.byte	22                              # DW_AT_name
                                        # DW_AT_default_value
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0xcb:0x4 DW_TAG_base_type
	.byte	21                              # DW_AT_name
	.byte	2                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	15                              # Abbrev [15] 0xcf:0x14 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	24                              # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	84                              # DW_AT_decl_line
	.byte	18                              # Abbrev [18] 0xd5:0xd DW_TAG_subprogram
	.byte	80                              # DW_AT_linkage_name
	.byte	81                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	85                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	19                              # Abbrev [19] 0xda:0x2 DW_TAG_template_type_parameter
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xdc:0x5 DW_TAG_formal_parameter
	.long	5538                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0xe3:0x17 DW_TAG_subprogram
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	138                             # DW_AT_linkage_name
	.byte	139                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	69                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	22                              # Abbrev [22] 0xef:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	1                               # DW_AT_decl_file
	.byte	69                              # DW_AT_decl_line
	.long	7503                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0xfa:0x85 DW_TAG_subprogram
	.byte	2                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	140                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	100                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x109:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.byte	198                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	102                             # DW_AT_decl_line
	.long	372                             # DW_AT_type
	.byte	25                              # Abbrev [25] 0x114:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	126
	.short	408                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	103                             # DW_AT_decl_line
	.long	367                             # DW_AT_type
	.byte	25                              # Abbrev [25] 0x120:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	125
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	136                             # DW_AT_decl_line
	.long	8174                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x12c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	151                             # DW_AT_decl_line
	.long	7618                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x138:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	123
	.short	413                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	176                             # DW_AT_decl_line
	.long	3255                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x144:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	122
	.short	414                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	211                             # DW_AT_decl_line
	.long	8190                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x150:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	121
	.short	416                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	220                             # DW_AT_decl_line
	.long	207                             # DW_AT_type
	.byte	25                              # Abbrev [25] 0x15c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	417                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	243                             # DW_AT_decl_line
	.long	8199                            # DW_AT_type
	.byte	26                              # Abbrev [26] 0x168:0x7 DW_TAG_imported_module
	.byte	1                               # DW_AT_decl_file
	.byte	208                             # DW_AT_decl_line
	.long	80                              # DW_AT_import
	.byte	27                              # Abbrev [27] 0x16f:0x5 DW_TAG_class_type
	.byte	5                               # DW_AT_calling_convention
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	103                             # DW_AT_decl_line
	.byte	28                              # Abbrev [28] 0x174:0x5 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	101                             # DW_AT_decl_line
	.byte	7                               # Abbrev [7] 0x179:0x3 DW_TAG_structure_type
	.short	299                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	14                              # Abbrev [14] 0x17c:0x2 DW_TAG_structure_type
	.byte	84                              # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x17f:0x2d DW_TAG_subprogram
	.byte	3                               # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	141                             # DW_AT_linkage_name
	.byte	142                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x18b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	7518                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x197:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8240                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1a3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1a5:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x1ac:0x2d DW_TAG_subprogram
	.byte	4                               # DW_AT_low_pc
	.long	.Lfunc_end3-.Lfunc_begin3       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	143                             # DW_AT_linkage_name
	.byte	144                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x1b8:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	3920                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1c4:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8257                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1d0:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1d2:0x5 DW_TAG_template_type_parameter
	.long	3935                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x1d9:0x2d DW_TAG_subprogram
	.byte	5                               # DW_AT_low_pc
	.long	.Lfunc_end4-.Lfunc_begin4       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	145                             # DW_AT_linkage_name
	.byte	146                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x1e5:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8274                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1f1:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8290                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1fd:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1ff:0x5 DW_TAG_template_type_parameter
	.long	203                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x206:0x2d DW_TAG_subprogram
	.byte	6                               # DW_AT_low_pc
	.long	.Lfunc_end5-.Lfunc_begin5       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	148                             # DW_AT_linkage_name
	.byte	149                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x212:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8307                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x21e:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8323                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x22a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x22c:0x5 DW_TAG_template_type_parameter
	.long	7499                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x233:0x2d DW_TAG_subprogram
	.byte	7                               # DW_AT_low_pc
	.long	.Lfunc_end6-.Lfunc_begin6       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	150                             # DW_AT_linkage_name
	.byte	151                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x23f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8340                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x24b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8356                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x257:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x259:0x5 DW_TAG_template_type_parameter
	.long	7187                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x260:0x2d DW_TAG_subprogram
	.byte	8                               # DW_AT_low_pc
	.long	.Lfunc_end7-.Lfunc_begin7       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	152                             # DW_AT_linkage_name
	.byte	153                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x26c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8373                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x278:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8389                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x284:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x286:0x5 DW_TAG_template_type_parameter
	.long	7151                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x28d:0x2d DW_TAG_subprogram
	.byte	9                               # DW_AT_low_pc
	.long	.Lfunc_end8-.Lfunc_begin8       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	154                             # DW_AT_linkage_name
	.byte	155                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x299:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8406                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x2a5:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8422                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x2b1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2b3:0x5 DW_TAG_template_type_parameter
	.long	76                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x2ba:0x2d DW_TAG_subprogram
	.byte	10                              # DW_AT_low_pc
	.long	.Lfunc_end9-.Lfunc_begin9       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	157                             # DW_AT_linkage_name
	.byte	158                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x2c6:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8439                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x2d2:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8455                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x2de:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2e0:0x5 DW_TAG_template_type_parameter
	.long	7503                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x2e7:0x2d DW_TAG_subprogram
	.byte	11                              # DW_AT_low_pc
	.long	.Lfunc_end10-.Lfunc_begin10     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	160                             # DW_AT_linkage_name
	.byte	161                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x2f3:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8472                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x2ff:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8488                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x30b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x30d:0x5 DW_TAG_template_type_parameter
	.long	7507                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x314:0x2d DW_TAG_subprogram
	.byte	12                              # DW_AT_low_pc
	.long	.Lfunc_end11-.Lfunc_begin11     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	163                             # DW_AT_linkage_name
	.byte	164                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x320:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8505                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x32c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8521                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x338:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x33a:0x5 DW_TAG_template_type_parameter
	.long	7511                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x341:0x2d DW_TAG_subprogram
	.byte	13                              # DW_AT_low_pc
	.long	.Lfunc_end12-.Lfunc_begin12     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	165                             # DW_AT_linkage_name
	.byte	166                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x34d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8538                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x359:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8554                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x365:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x367:0x5 DW_TAG_template_type_parameter
	.long	171                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x36e:0x2d DW_TAG_subprogram
	.byte	14                              # DW_AT_low_pc
	.long	.Lfunc_end13-.Lfunc_begin13     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	167                             # DW_AT_linkage_name
	.byte	168                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x37a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8571                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x386:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8587                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x392:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x394:0x5 DW_TAG_template_type_parameter
	.long	7513                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x39b:0x2d DW_TAG_subprogram
	.byte	15                              # DW_AT_low_pc
	.long	.Lfunc_end14-.Lfunc_begin14     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	170                             # DW_AT_linkage_name
	.byte	171                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x3a7:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8604                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x3b3:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8620                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x3bf:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x3c1:0x5 DW_TAG_template_type_parameter
	.long	175                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x3c8:0x2d DW_TAG_subprogram
	.byte	16                              # DW_AT_low_pc
	.long	.Lfunc_end15-.Lfunc_begin15     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	173                             # DW_AT_linkage_name
	.byte	174                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x3d4:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8637                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x3e0:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8653                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x3ec:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x3ee:0x5 DW_TAG_template_type_parameter
	.long	7518                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x3f5:0x32 DW_TAG_subprogram
	.byte	17                              # DW_AT_low_pc
	.long	.Lfunc_end16-.Lfunc_begin16     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	175                             # DW_AT_linkage_name
	.byte	176                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x401:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8670                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x40d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8691                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x419:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x41b:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	30                              # Abbrev [30] 0x420:0x5 DW_TAG_template_type_parameter
	.long	3935                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x427:0x2d DW_TAG_subprogram
	.byte	18                              # DW_AT_low_pc
	.long	.Lfunc_end17-.Lfunc_begin17     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	177                             # DW_AT_linkage_name
	.byte	178                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x433:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	7730                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x43f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8713                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x44b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x44d:0x5 DW_TAG_template_type_parameter
	.long	7533                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x454:0x2d DW_TAG_subprogram
	.byte	19                              # DW_AT_low_pc
	.long	.Lfunc_end18-.Lfunc_begin18     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	179                             # DW_AT_linkage_name
	.byte	180                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x460:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8730                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x46c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8746                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x478:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x47a:0x5 DW_TAG_template_type_parameter
	.long	7538                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x481:0x2d DW_TAG_subprogram
	.byte	20                              # DW_AT_low_pc
	.long	.Lfunc_end19-.Lfunc_begin19     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	181                             # DW_AT_linkage_name
	.byte	182                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x48d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8763                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x499:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8779                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x4a5:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x4a7:0x5 DW_TAG_template_type_parameter
	.long	7543                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x4ae:0x2d DW_TAG_subprogram
	.byte	21                              # DW_AT_low_pc
	.long	.Lfunc_end20-.Lfunc_begin20     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	183                             # DW_AT_linkage_name
	.byte	184                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x4ba:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8796                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x4c6:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8812                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x4d2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x4d4:0x5 DW_TAG_template_type_parameter
	.long	7548                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x4db:0x2d DW_TAG_subprogram
	.byte	22                              # DW_AT_low_pc
	.long	.Lfunc_end21-.Lfunc_begin21     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	186                             # DW_AT_linkage_name
	.byte	187                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x4e7:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8829                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x4f3:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8845                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x4ff:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x501:0x5 DW_TAG_template_type_parameter
	.long	7553                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x508:0x29 DW_TAG_subprogram
	.byte	23                              # DW_AT_low_pc
	.long	.Lfunc_end22-.Lfunc_begin22     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	188                             # DW_AT_linkage_name
	.byte	189                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x514:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8862                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x520:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8874                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x52c:0x4 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x52e:0x1 DW_TAG_template_type_parameter
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x531:0x2d DW_TAG_subprogram
	.byte	24                              # DW_AT_low_pc
	.long	.Lfunc_end23-.Lfunc_begin23     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	192                             # DW_AT_linkage_name
	.byte	193                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x53d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8887                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x549:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8903                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x555:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x557:0x5 DW_TAG_template_type_parameter
	.long	7575                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x55e:0x2d DW_TAG_subprogram
	.byte	25                              # DW_AT_low_pc
	.long	.Lfunc_end24-.Lfunc_begin24     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	194                             # DW_AT_linkage_name
	.byte	195                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x56a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8920                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x576:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8936                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x582:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x584:0x5 DW_TAG_template_type_parameter
	.long	4803                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x58b:0x1b DW_TAG_subprogram
	.byte	26                              # DW_AT_low_pc
	.long	.Lfunc_end25-.Lfunc_begin25     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	196                             # DW_AT_linkage_name
	.byte	197                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	22                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	32                              # Abbrev [32] 0x597:0x7 DW_TAG_template_value_parameter
	.long	203                             # DW_AT_type
	.byte	22                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	33                              # Abbrev [33] 0x59e:0x7 DW_TAG_template_value_parameter
	.long	54                              # DW_AT_type
	.byte	3                               # DW_AT_name
	.byte	3                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x5a6:0x22 DW_TAG_subprogram
	.byte	27                              # DW_AT_low_pc
	.long	.Lfunc_end26-.Lfunc_begin26     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	199                             # DW_AT_linkage_name
	.byte	200                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x5b2:0x6 DW_TAG_template_type_parameter
	.long	82                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x5b8:0xf DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	34                              # Abbrev [34] 0x5ba:0x6 DW_TAG_template_value_parameter
	.long	82                              # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0x5c0:0x6 DW_TAG_template_value_parameter
	.long	82                              # DW_AT_type
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x5c8:0x22 DW_TAG_subprogram
	.byte	28                              # DW_AT_low_pc
	.long	.Lfunc_end27-.Lfunc_begin27     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	201                             # DW_AT_linkage_name
	.byte	202                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x5d4:0x6 DW_TAG_template_type_parameter
	.long	101                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x5da:0xf DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	34                              # Abbrev [34] 0x5dc:0x6 DW_TAG_template_value_parameter
	.long	101                             # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0x5e2:0x6 DW_TAG_template_value_parameter
	.long	101                             # DW_AT_type
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x5ea:0x1d DW_TAG_subprogram
	.byte	29                              # DW_AT_low_pc
	.long	.Lfunc_end28-.Lfunc_begin28     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	203                             # DW_AT_linkage_name
	.byte	204                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x5f6:0x6 DW_TAG_template_type_parameter
	.long	120                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x5fc:0xa DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	35                              # Abbrev [35] 0x5fe:0x7 DW_TAG_template_value_parameter
	.long	120                             # DW_AT_type
	.ascii	"\377\001"                      # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	36                              # Abbrev [36] 0x607:0x22 DW_TAG_subprogram
	.byte	30                              # DW_AT_low_pc
	.long	.Lfunc_end29-.Lfunc_begin29     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	205                             # DW_AT_linkage_name
	.byte	206                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
	.byte	16                              # Abbrev [16] 0x613:0x6 DW_TAG_template_type_parameter
	.long	134                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x619:0xf DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	34                              # Abbrev [34] 0x61b:0x6 DW_TAG_template_value_parameter
	.long	134                             # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0x621:0x6 DW_TAG_template_value_parameter
	.long	134                             # DW_AT_type
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	36                              # Abbrev [36] 0x629:0x1c DW_TAG_subprogram
	.byte	31                              # DW_AT_low_pc
	.long	.Lfunc_end30-.Lfunc_begin30     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	207                             # DW_AT_linkage_name
	.byte	208                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
	.byte	16                              # Abbrev [16] 0x635:0x6 DW_TAG_template_type_parameter
	.long	59                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x63b:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	35                              # Abbrev [35] 0x63d:0x6 DW_TAG_template_value_parameter
	.long	59                              # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x645:0x1f DW_TAG_subprogram
	.byte	32                              # DW_AT_low_pc
	.long	.Lfunc_end31-.Lfunc_begin31     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	209                             # DW_AT_linkage_name
	.byte	210                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x651:0x6 DW_TAG_template_type_parameter
	.long	7533                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x657:0xc DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	37                              # Abbrev [37] 0x659:0x9 DW_TAG_template_value_parameter
	.long	7533                            # DW_AT_type
	.byte	3                               # DW_AT_location
	.byte	161
	.byte	0
	.byte	159
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x664:0x1c DW_TAG_subprogram
	.byte	33                              # DW_AT_low_pc
	.long	.Lfunc_end32-.Lfunc_begin32     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	211                             # DW_AT_linkage_name
	.byte	212                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x670:0x6 DW_TAG_template_type_parameter
	.long	7533                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x676:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	35                              # Abbrev [35] 0x678:0x6 DW_TAG_template_value_parameter
	.long	7533                            # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x680:0x1c DW_TAG_subprogram
	.byte	34                              # DW_AT_low_pc
	.long	.Lfunc_end33-.Lfunc_begin33     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	213                             # DW_AT_linkage_name
	.byte	214                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x68c:0x6 DW_TAG_template_type_parameter
	.long	4803                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x692:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	35                              # Abbrev [35] 0x694:0x6 DW_TAG_template_value_parameter
	.long	4803                            # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x69c:0x1c DW_TAG_subprogram
	.byte	35                              # DW_AT_low_pc
	.long	.Lfunc_end34-.Lfunc_begin34     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	215                             # DW_AT_linkage_name
	.byte	216                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x6a8:0x6 DW_TAG_template_type_parameter
	.long	7503                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x6ae:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	35                              # Abbrev [35] 0x6b0:0x6 DW_TAG_template_value_parameter
	.long	7503                            # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x6b8:0x1c DW_TAG_subprogram
	.byte	36                              # DW_AT_low_pc
	.long	.Lfunc_end35-.Lfunc_begin35     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	217                             # DW_AT_linkage_name
	.byte	218                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x6c4:0x6 DW_TAG_template_type_parameter
	.long	7187                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x6ca:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	34                              # Abbrev [34] 0x6cc:0x6 DW_TAG_template_value_parameter
	.long	7187                            # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x6d4:0x1c DW_TAG_subprogram
	.byte	37                              # DW_AT_low_pc
	.long	.Lfunc_end36-.Lfunc_begin36     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	219                             # DW_AT_linkage_name
	.byte	220                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x6e0:0x6 DW_TAG_template_type_parameter
	.long	76                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x6e6:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	35                              # Abbrev [35] 0x6e8:0x6 DW_TAG_template_value_parameter
	.long	76                              # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x6f0:0x1c DW_TAG_subprogram
	.byte	38                              # DW_AT_low_pc
	.long	.Lfunc_end37-.Lfunc_begin37     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	221                             # DW_AT_linkage_name
	.byte	222                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x6fc:0x6 DW_TAG_template_type_parameter
	.long	7151                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x702:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	34                              # Abbrev [34] 0x704:0x6 DW_TAG_template_value_parameter
	.long	7151                            # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x70c:0x1c DW_TAG_subprogram
	.byte	39                              # DW_AT_low_pc
	.long	.Lfunc_end38-.Lfunc_begin38     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	223                             # DW_AT_linkage_name
	.byte	224                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x718:0x6 DW_TAG_template_type_parameter
	.long	179                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x71e:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	35                              # Abbrev [35] 0x720:0x6 DW_TAG_template_value_parameter
	.long	179                             # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x728:0x1c DW_TAG_subprogram
	.byte	40                              # DW_AT_low_pc
	.long	.Lfunc_end39-.Lfunc_begin39     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	225                             # DW_AT_linkage_name
	.byte	226                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x734:0x6 DW_TAG_template_type_parameter
	.long	7131                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x73a:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	34                              # Abbrev [34] 0x73c:0x6 DW_TAG_template_value_parameter
	.long	7131                            # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x744:0x22 DW_TAG_subprogram
	.byte	41                              # DW_AT_low_pc
	.long	.Lfunc_end40-.Lfunc_begin40     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	227                             # DW_AT_linkage_name
	.byte	228                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x750:0x6 DW_TAG_template_type_parameter
	.long	7343                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x756:0xf DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	35                              # Abbrev [35] 0x758:0x6 DW_TAG_template_value_parameter
	.long	7343                            # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	35                              # Abbrev [35] 0x75e:0x6 DW_TAG_template_value_parameter
	.long	7343                            # DW_AT_type
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x766:0x5a DW_TAG_subprogram
	.byte	42                              # DW_AT_low_pc
	.long	.Lfunc_end41-.Lfunc_begin41     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	230                             # DW_AT_linkage_name
	.byte	231                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x772:0x6 DW_TAG_template_type_parameter
	.long	7578                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x778:0x47 DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	34                              # Abbrev [34] 0x77a:0x6 DW_TAG_template_value_parameter
	.long	7578                            # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0x780:0x6 DW_TAG_template_value_parameter
	.long	7578                            # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0x786:0x6 DW_TAG_template_value_parameter
	.long	7578                            # DW_AT_type
	.byte	6                               # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0x78c:0x6 DW_TAG_template_value_parameter
	.long	7578                            # DW_AT_type
	.byte	7                               # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0x792:0x6 DW_TAG_template_value_parameter
	.long	7578                            # DW_AT_type
	.byte	13                              # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0x798:0x6 DW_TAG_template_value_parameter
	.long	7578                            # DW_AT_type
	.byte	14                              # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0x79e:0x6 DW_TAG_template_value_parameter
	.long	7578                            # DW_AT_type
	.byte	31                              # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0x7a4:0x6 DW_TAG_template_value_parameter
	.long	7578                            # DW_AT_type
	.byte	32                              # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0x7aa:0x6 DW_TAG_template_value_parameter
	.long	7578                            # DW_AT_type
	.byte	33                              # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0x7b0:0x7 DW_TAG_template_value_parameter
	.long	7578                            # DW_AT_type
	.asciz	"\377"                          # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0x7b7:0x7 DW_TAG_template_value_parameter
	.long	7578                            # DW_AT_type
	.ascii	"\200\177"                      # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x7c0:0x2c DW_TAG_subprogram
	.byte	43                              # DW_AT_low_pc
	.long	.Lfunc_end42-.Lfunc_begin42     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	233                             # DW_AT_linkage_name
	.byte	234                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x7cc:0x6 DW_TAG_template_type_parameter
	.long	7582                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x7d2:0x19 DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	38                              # Abbrev [38] 0x7d4:0x16 DW_TAG_template_value_parameter
	.long	7582                            # DW_AT_type
	.byte	16                              # DW_AT_const_value
	.byte	254
	.byte	255
	.byte	255
	.byte	255
	.byte	255
	.byte	255
	.byte	255
	.byte	255
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x7ec:0x19 DW_TAG_subprogram
	.byte	44                              # DW_AT_low_pc
	.long	.Lfunc_end43-.Lfunc_begin43     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	235                             # DW_AT_linkage_name
	.byte	236                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	24                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x7f8:0x6 DW_TAG_template_type_parameter
	.long	76                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	35                              # Abbrev [35] 0x7fe:0x6 DW_TAG_template_value_parameter
	.long	76                              # DW_AT_type
	.byte	3                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x805:0x2d DW_TAG_subprogram
	.byte	45                              # DW_AT_low_pc
	.long	.Lfunc_end44-.Lfunc_begin44     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	237                             # DW_AT_linkage_name
	.byte	238                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x811:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8953                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x81d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	8969                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x829:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x82b:0x5 DW_TAG_template_type_parameter
	.long	183                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x832:0x2d DW_TAG_subprogram
	.byte	46                              # DW_AT_low_pc
	.long	.Lfunc_end45-.Lfunc_begin45     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	240                             # DW_AT_linkage_name
	.byte	241                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x83e:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8986                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x84a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9002                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x856:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x858:0x5 DW_TAG_template_type_parameter
	.long	7586                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	36                              # Abbrev [36] 0x85f:0x2d DW_TAG_subprogram
	.byte	47                              # DW_AT_low_pc
	.long	.Lfunc_end46-.Lfunc_begin46     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	242                             # DW_AT_linkage_name
	.byte	243                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0x86b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	7824                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x877:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9019                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x883:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x885:0x5 DW_TAG_template_type_parameter
	.long	367                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	36                              # Abbrev [36] 0x88c:0x2d DW_TAG_subprogram
	.byte	48                              # DW_AT_low_pc
	.long	.Lfunc_end47-.Lfunc_begin47     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	246                             # DW_AT_linkage_name
	.byte	247                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0x898:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9036                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x8a4:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9052                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x8b0:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x8b2:0x5 DW_TAG_template_type_parameter
	.long	7602                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x8b9:0x2d DW_TAG_subprogram
	.byte	49                              # DW_AT_low_pc
	.long	.Lfunc_end48-.Lfunc_begin48     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	248                             # DW_AT_linkage_name
	.byte	249                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x8c5:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9069                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x8d1:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9085                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x8dd:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x8df:0x5 DW_TAG_template_type_parameter
	.long	7638                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x8e6:0x2d DW_TAG_subprogram
	.byte	50                              # DW_AT_low_pc
	.long	.Lfunc_end49-.Lfunc_begin49     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	250                             # DW_AT_linkage_name
	.byte	251                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x8f2:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9102                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x8fe:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9118                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x90a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x90c:0x5 DW_TAG_template_type_parameter
	.long	7649                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x913:0x2d DW_TAG_subprogram
	.byte	51                              # DW_AT_low_pc
	.long	.Lfunc_end50-.Lfunc_begin50     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	252                             # DW_AT_linkage_name
	.byte	253                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x91f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9135                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x92b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9151                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x937:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x939:0x5 DW_TAG_template_type_parameter
	.long	7652                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x940:0x2d DW_TAG_subprogram
	.byte	52                              # DW_AT_low_pc
	.long	.Lfunc_end51-.Lfunc_begin51     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	254                             # DW_AT_linkage_name
	.byte	255                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x94c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9168                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x958:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9184                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x964:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x966:0x5 DW_TAG_template_type_parameter
	.long	7660                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x96d:0x2f DW_TAG_subprogram
	.byte	53                              # DW_AT_low_pc
	.long	.Lfunc_end52-.Lfunc_begin52     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	256                             # DW_AT_linkage_name
	.short	257                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x97b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9201                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x987:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9217                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x993:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x995:0x5 DW_TAG_template_type_parameter
	.long	7665                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x99c:0x2f DW_TAG_subprogram
	.byte	54                              # DW_AT_low_pc
	.long	.Lfunc_end53-.Lfunc_begin53     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	259                             # DW_AT_linkage_name
	.short	260                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0x9aa:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9234                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x9b6:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9250                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x9c2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x9c4:0x5 DW_TAG_template_type_parameter
	.long	72                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x9cb:0x2f DW_TAG_subprogram
	.byte	55                              # DW_AT_low_pc
	.long	.Lfunc_end54-.Lfunc_begin54     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	262                             # DW_AT_linkage_name
	.short	263                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x9d9:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9267                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x9e5:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9283                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x9f1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x9f3:0x5 DW_TAG_template_type_parameter
	.long	7675                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x9fa:0x34 DW_TAG_subprogram
	.byte	56                              # DW_AT_low_pc
	.long	.Lfunc_end55-.Lfunc_begin55     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	264                             # DW_AT_linkage_name
	.short	265                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0xa08:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9300                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0xa14:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9321                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0xa20:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0xa22:0x5 DW_TAG_template_type_parameter
	.long	7678                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0xa27:0x5 DW_TAG_template_type_parameter
	.long	7678                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xa2e:0x34 DW_TAG_subprogram
	.byte	57                              # DW_AT_low_pc
	.long	.Lfunc_end56-.Lfunc_begin56     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	266                             # DW_AT_linkage_name
	.short	267                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0xa3c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9343                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0xa48:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9364                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0xa54:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0xa56:0x5 DW_TAG_template_type_parameter
	.long	7678                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0xa5b:0x5 DW_TAG_template_type_parameter
	.long	7683                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xa62:0x2f DW_TAG_subprogram
	.byte	58                              # DW_AT_low_pc
	.long	.Lfunc_end57-.Lfunc_begin57     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	268                             # DW_AT_linkage_name
	.short	269                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0xa70:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9386                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0xa7c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9402                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0xa88:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0xa8a:0x5 DW_TAG_template_type_parameter
	.long	7688                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xa91:0x2f DW_TAG_subprogram
	.byte	59                              # DW_AT_low_pc
	.long	.Lfunc_end58-.Lfunc_begin58     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	270                             # DW_AT_linkage_name
	.short	271                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0xa9f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9419                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0xaab:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9435                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0xab7:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0xab9:0x5 DW_TAG_template_type_parameter
	.long	7693                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xac0:0x2f DW_TAG_subprogram
	.byte	60                              # DW_AT_low_pc
	.long	.Lfunc_end59-.Lfunc_begin59     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	272                             # DW_AT_linkage_name
	.short	273                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0xace:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9452                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0xada:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9468                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0xae6:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0xae8:0x5 DW_TAG_template_type_parameter
	.long	7709                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xaef:0x2f DW_TAG_subprogram
	.byte	61                              # DW_AT_low_pc
	.long	.Lfunc_end60-.Lfunc_begin60     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	274                             # DW_AT_linkage_name
	.short	275                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0xafd:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9485                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0xb09:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9501                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0xb15:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0xb17:0x5 DW_TAG_template_type_parameter
	.long	7710                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0xb1e:0x2f DW_TAG_subprogram
	.byte	62                              # DW_AT_low_pc
	.long	.Lfunc_end61-.Lfunc_begin61     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	276                             # DW_AT_linkage_name
	.short	277                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0xb2c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9518                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0xb38:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9534                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0xb44:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0xb46:0x5 DW_TAG_template_type_parameter
	.long	7715                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0xb4d:0x2f DW_TAG_subprogram
	.byte	63                              # DW_AT_low_pc
	.long	.Lfunc_end62-.Lfunc_begin62     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	278                             # DW_AT_linkage_name
	.short	279                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0xb5b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9551                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0xb67:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9567                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0xb73:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0xb75:0x5 DW_TAG_template_type_parameter
	.long	372                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0xb7c:0x2f DW_TAG_subprogram
	.byte	64                              # DW_AT_low_pc
	.long	.Lfunc_end63-.Lfunc_begin63     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	280                             # DW_AT_linkage_name
	.short	281                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0xb8a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9584                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0xb96:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9600                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0xba2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0xba4:0x5 DW_TAG_template_type_parameter
	.long	7720                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xbab:0x1f DW_TAG_subprogram
	.byte	65                              # DW_AT_low_pc
	.long	.Lfunc_end64-.Lfunc_begin64     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	284                             # DW_AT_linkage_name
	.short	285                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	40                              # Abbrev [40] 0xbb9:0x9 DW_TAG_GNU_template_parameter_pack
	.short	282                             # DW_AT_name
	.byte	30                              # Abbrev [30] 0xbbc:0x5 DW_TAG_template_type_parameter
	.long	7518                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	41                              # Abbrev [41] 0xbc2:0x7 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.short	283                             # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xbca:0x19 DW_TAG_subprogram
	.byte	66                              # DW_AT_low_pc
	.long	.Lfunc_end65-.Lfunc_begin65     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	286                             # DW_AT_linkage_name
	.short	287                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	42                              # Abbrev [42] 0xbd8:0x3 DW_TAG_GNU_template_parameter_pack
	.short	282                             # DW_AT_name
	.byte	41                              # Abbrev [41] 0xbdb:0x7 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.short	283                             # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xbe3:0x19 DW_TAG_subprogram
	.byte	67                              # DW_AT_low_pc
	.long	.Lfunc_end66-.Lfunc_begin66     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	288                             # DW_AT_linkage_name
	.short	289                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	38                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	41                              # Abbrev [41] 0xbf1:0x7 DW_TAG_template_type_parameter
	.long	7518                            # DW_AT_type
	.short	282                             # DW_AT_name
	.byte	42                              # Abbrev [42] 0xbf8:0x3 DW_TAG_GNU_template_parameter_pack
	.short	283                             # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xbfc:0x29 DW_TAG_subprogram
	.byte	68                              # DW_AT_low_pc
	.long	.Lfunc_end67-.Lfunc_begin67     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	290                             # DW_AT_linkage_name
	.short	291                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0xc0a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	8123                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0xc16:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9617                            # DW_AT_type
	.byte	43                              # Abbrev [43] 0xc22:0x2 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xc25:0x34 DW_TAG_subprogram
	.byte	69                              # DW_AT_low_pc
	.long	.Lfunc_end68-.Lfunc_begin68     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	292                             # DW_AT_linkage_name
	.short	293                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0xc33:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9628                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0xc3f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9649                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0xc4b:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0xc4d:0x5 DW_TAG_template_type_parameter
	.long	7703                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0xc52:0x5 DW_TAG_template_type_parameter
	.long	7703                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xc59:0x2f DW_TAG_subprogram
	.byte	70                              # DW_AT_low_pc
	.long	.Lfunc_end69-.Lfunc_begin69     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	295                             # DW_AT_linkage_name
	.short	296                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0xc67:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9671                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0xc73:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9687                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0xc7f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0xc81:0x5 DW_TAG_template_type_parameter
	.long	7725                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xc88:0x2f DW_TAG_subprogram
	.byte	71                              # DW_AT_low_pc
	.long	.Lfunc_end70-.Lfunc_begin70     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	297                             # DW_AT_linkage_name
	.short	298                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0xc96:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9704                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0xca2:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9720                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0xcae:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0xcb0:0x5 DW_TAG_template_type_parameter
	.long	7746                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0xcb7:0x20e DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	25                              # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	39                              # DW_AT_decl_line
	.byte	18                              # Abbrev [18] 0xcbd:0x16 DW_TAG_subprogram
	.byte	26                              # DW_AT_linkage_name
	.byte	27                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	40                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xcc2:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xcc8:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	44                              # Abbrev [44] 0xccd:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xcd3:0x16 DW_TAG_subprogram
	.byte	28                              # DW_AT_linkage_name
	.byte	29                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xcd8:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xcde:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	44                              # Abbrev [44] 0xce3:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xce9:0x16 DW_TAG_subprogram
	.byte	30                              # DW_AT_linkage_name
	.byte	31                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	42                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xcee:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xcf4:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	44                              # Abbrev [44] 0xcf9:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	45                              # Abbrev [45] 0xcff:0x15 DW_TAG_subprogram
	.byte	32                              # DW_AT_linkage_name
	.byte	33                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
	.long	3915                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xd08:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xd0e:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xd14:0x16 DW_TAG_subprogram
	.byte	37                              # DW_AT_linkage_name
	.byte	38                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	44                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xd19:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xd1f:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	44                              # Abbrev [44] 0xd24:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xd2a:0x16 DW_TAG_subprogram
	.byte	39                              # DW_AT_linkage_name
	.byte	40                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	45                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xd2f:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xd35:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	44                              # Abbrev [44] 0xd3a:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xd40:0x16 DW_TAG_subprogram
	.byte	41                              # DW_AT_linkage_name
	.byte	42                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	46                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xd45:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xd4b:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	44                              # Abbrev [44] 0xd50:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xd56:0x16 DW_TAG_subprogram
	.byte	43                              # DW_AT_linkage_name
	.byte	44                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	47                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xd5b:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xd61:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	44                              # Abbrev [44] 0xd66:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xd6c:0x16 DW_TAG_subprogram
	.byte	45                              # DW_AT_linkage_name
	.byte	46                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	48                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xd71:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xd77:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	44                              # Abbrev [44] 0xd7c:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xd82:0x16 DW_TAG_subprogram
	.byte	47                              # DW_AT_linkage_name
	.byte	48                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	49                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xd87:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xd8d:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	44                              # Abbrev [44] 0xd92:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xd98:0x16 DW_TAG_subprogram
	.byte	49                              # DW_AT_linkage_name
	.byte	50                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	50                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xd9d:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xda3:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	44                              # Abbrev [44] 0xda8:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xdae:0x11 DW_TAG_subprogram
	.byte	51                              # DW_AT_linkage_name
	.byte	52                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	51                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xdb3:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xdb9:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xdbf:0x11 DW_TAG_subprogram
	.byte	53                              # DW_AT_linkage_name
	.byte	54                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	52                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xdc4:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xdca:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xdd0:0x16 DW_TAG_subprogram
	.byte	55                              # DW_AT_linkage_name
	.byte	56                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	53                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xdd5:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xddb:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	44                              # Abbrev [44] 0xde0:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xde6:0x16 DW_TAG_subprogram
	.byte	57                              # DW_AT_linkage_name
	.byte	58                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	54                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xdeb:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xdf1:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	44                              # Abbrev [44] 0xdf6:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xdfc:0x16 DW_TAG_subprogram
	.byte	59                              # DW_AT_linkage_name
	.byte	60                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	55                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xe01:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xe07:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	44                              # Abbrev [44] 0xe0c:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xe12:0x11 DW_TAG_subprogram
	.byte	61                              # DW_AT_linkage_name
	.byte	62                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	56                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xe17:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xe1d:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xe23:0x16 DW_TAG_subprogram
	.byte	63                              # DW_AT_linkage_name
	.byte	64                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	57                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xe28:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xe2e:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	44                              # Abbrev [44] 0xe33:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xe39:0x16 DW_TAG_subprogram
	.byte	65                              # DW_AT_linkage_name
	.byte	66                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xe3e:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xe44:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	44                              # Abbrev [44] 0xe49:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	45                              # Abbrev [45] 0xe4f:0x1a DW_TAG_subprogram
	.byte	67                              # DW_AT_linkage_name
	.byte	68                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	59                              # DW_AT_decl_line
	.long	4587                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xe58:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	44                              # Abbrev [44] 0xe5e:0x5 DW_TAG_formal_parameter
	.long	4590                            # DW_AT_type
	.byte	44                              # Abbrev [44] 0xe63:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	45                              # Abbrev [45] 0xe69:0x1a DW_TAG_subprogram
	.byte	72                              # DW_AT_linkage_name
	.byte	73                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	63                              # DW_AT_decl_line
	.long	4587                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xe72:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	44                              # Abbrev [44] 0xe78:0x5 DW_TAG_formal_parameter
	.long	4590                            # DW_AT_type
	.byte	44                              # Abbrev [44] 0xe7d:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xe83:0x16 DW_TAG_subprogram
	.byte	74                              # DW_AT_linkage_name
	.byte	75                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	62                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xe88:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	44                              # Abbrev [44] 0xe8e:0x5 DW_TAG_formal_parameter
	.long	4587                            # DW_AT_type
	.byte	44                              # Abbrev [44] 0xe93:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0xe99:0x16 DW_TAG_subprogram
	.byte	76                              # DW_AT_linkage_name
	.byte	77                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	66                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xe9e:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	44                              # Abbrev [44] 0xea4:0x5 DW_TAG_formal_parameter
	.long	4587                            # DW_AT_type
	.byte	44                              # Abbrev [44] 0xea9:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	45                              # Abbrev [45] 0xeaf:0x15 DW_TAG_subprogram
	.byte	78                              # DW_AT_linkage_name
	.byte	79                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	67                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xeb8:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xebe:0x5 DW_TAG_formal_parameter
	.long	3781                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0xec5:0x5 DW_TAG_pointer_type
	.long	3255                            # DW_AT_type
	.byte	47                              # Abbrev [47] 0xeca:0x2b DW_TAG_subprogram
	.byte	72                              # DW_AT_low_pc
	.long	.Lfunc_end71-.Lfunc_begin71     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3802                            # DW_AT_object_pointer
	.long	3261                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0xeda:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0xee4:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	40                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	16                              # Abbrev [16] 0xeee:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0xef5:0x2b DW_TAG_subprogram
	.byte	73                              # DW_AT_low_pc
	.long	.Lfunc_end72-.Lfunc_begin72     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3845                            # DW_AT_object_pointer
	.long	3283                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0xf05:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0xf0f:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	16                              # Abbrev [16] 0xf19:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0xf20:0x2b DW_TAG_subprogram
	.byte	74                              # DW_AT_low_pc
	.long	.Lfunc_end73-.Lfunc_begin73     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3888                            # DW_AT_object_pointer
	.long	3305                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0xf30:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0xf3a:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	42                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	16                              # Abbrev [16] 0xf44:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0xf4b:0x5 DW_TAG_pointer_type
	.long	3920                            # DW_AT_type
	.byte	15                              # Abbrev [15] 0xf50:0xf DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	36                              # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0xf56:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0xf58:0x5 DW_TAG_template_type_parameter
	.long	3935                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0xf5f:0x4 DW_TAG_base_type
	.byte	35                              # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	47                              # Abbrev [47] 0xf63:0x21 DW_TAG_subprogram
	.byte	75                              # DW_AT_low_pc
	.long	.Lfunc_end74-.Lfunc_begin74     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3955                            # DW_AT_object_pointer
	.long	3327                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0xf73:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	16                              # Abbrev [16] 0xf7d:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0xf84:0x2b DW_TAG_subprogram
	.byte	76                              # DW_AT_low_pc
	.long	.Lfunc_end75-.Lfunc_begin75     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3988                            # DW_AT_object_pointer
	.long	3348                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0xf94:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0xf9e:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	44                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	16                              # Abbrev [16] 0xfa8:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0xfaf:0x2b DW_TAG_subprogram
	.byte	77                              # DW_AT_low_pc
	.long	.Lfunc_end76-.Lfunc_begin76     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4031                            # DW_AT_object_pointer
	.long	3370                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0xfbf:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0xfc9:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	45                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	16                              # Abbrev [16] 0xfd3:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0xfda:0x2b DW_TAG_subprogram
	.byte	78                              # DW_AT_low_pc
	.long	.Lfunc_end77-.Lfunc_begin77     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4074                            # DW_AT_object_pointer
	.long	3392                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0xfea:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0xff4:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	46                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	16                              # Abbrev [16] 0xffe:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1005:0x2b DW_TAG_subprogram
	.byte	79                              # DW_AT_low_pc
	.long	.Lfunc_end78-.Lfunc_begin78     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4117                            # DW_AT_object_pointer
	.long	3414                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0x1015:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x101f:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	47                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	16                              # Abbrev [16] 0x1029:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1030:0x2b DW_TAG_subprogram
	.byte	80                              # DW_AT_low_pc
	.long	.Lfunc_end79-.Lfunc_begin79     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4160                            # DW_AT_object_pointer
	.long	3436                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0x1040:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x104a:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	48                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	16                              # Abbrev [16] 0x1054:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x105b:0x2b DW_TAG_subprogram
	.byte	81                              # DW_AT_low_pc
	.long	.Lfunc_end80-.Lfunc_begin80     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4203                            # DW_AT_object_pointer
	.long	3458                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0x106b:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x1075:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	49                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	16                              # Abbrev [16] 0x107f:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1086:0x2b DW_TAG_subprogram
	.byte	82                              # DW_AT_low_pc
	.long	.Lfunc_end81-.Lfunc_begin81     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4246                            # DW_AT_object_pointer
	.long	3480                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0x1096:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x10a0:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	50                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	16                              # Abbrev [16] 0x10aa:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x10b1:0x21 DW_TAG_subprogram
	.byte	83                              # DW_AT_low_pc
	.long	.Lfunc_end82-.Lfunc_begin82     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4289                            # DW_AT_object_pointer
	.long	3502                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0x10c1:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	16                              # Abbrev [16] 0x10cb:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x10d2:0x21 DW_TAG_subprogram
	.byte	84                              # DW_AT_low_pc
	.long	.Lfunc_end83-.Lfunc_begin83     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4322                            # DW_AT_object_pointer
	.long	3519                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0x10e2:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	16                              # Abbrev [16] 0x10ec:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x10f3:0x2b DW_TAG_subprogram
	.byte	85                              # DW_AT_low_pc
	.long	.Lfunc_end84-.Lfunc_begin84     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4355                            # DW_AT_object_pointer
	.long	3536                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0x1103:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x110d:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	53                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	16                              # Abbrev [16] 0x1117:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x111e:0x2b DW_TAG_subprogram
	.byte	86                              # DW_AT_low_pc
	.long	.Lfunc_end85-.Lfunc_begin85     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4398                            # DW_AT_object_pointer
	.long	3558                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0x112e:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x1138:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	54                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	16                              # Abbrev [16] 0x1142:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1149:0x2b DW_TAG_subprogram
	.byte	87                              # DW_AT_low_pc
	.long	.Lfunc_end86-.Lfunc_begin86     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4441                            # DW_AT_object_pointer
	.long	3580                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0x1159:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x1163:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	55                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	16                              # Abbrev [16] 0x116d:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1174:0x21 DW_TAG_subprogram
	.byte	88                              # DW_AT_low_pc
	.long	.Lfunc_end87-.Lfunc_begin87     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4484                            # DW_AT_object_pointer
	.long	3602                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0x1184:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	16                              # Abbrev [16] 0x118e:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1195:0x2b DW_TAG_subprogram
	.byte	89                              # DW_AT_low_pc
	.long	.Lfunc_end88-.Lfunc_begin88     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4517                            # DW_AT_object_pointer
	.long	3619                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0x11a5:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x11af:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	57                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	16                              # Abbrev [16] 0x11b9:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x11c0:0x2b DW_TAG_subprogram
	.byte	90                              # DW_AT_low_pc
	.long	.Lfunc_end89-.Lfunc_begin89     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4560                            # DW_AT_object_pointer
	.long	3641                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0x11d0:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	9737                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x11da:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	16                              # Abbrev [16] 0x11e4:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	49                              # Abbrev [49] 0x11eb:0x1 DW_TAG_pointer_type
	.byte	8                               # Abbrev [8] 0x11ec:0xd7 DW_TAG_namespace
	.byte	69                              # DW_AT_name
	.byte	50                              # Abbrev [50] 0x11ee:0x9 DW_TAG_typedef
	.long	4803                            # DW_AT_type
	.byte	71                              # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	2441                            # DW_AT_decl_line
	.byte	51                              # Abbrev [51] 0x11f7:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.long	7106                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x11fe:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	47                              # DW_AT_decl_line
	.long	7115                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x1205:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	48                              # DW_AT_decl_line
	.long	7135                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x120c:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	49                              # DW_AT_decl_line
	.long	7155                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x1213:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	50                              # DW_AT_decl_line
	.long	7171                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x121a:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	52                              # DW_AT_decl_line
	.long	7191                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x1221:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	53                              # DW_AT_decl_line
	.long	7199                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x1228:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	54                              # DW_AT_decl_line
	.long	7207                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x122f:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	55                              # DW_AT_decl_line
	.long	7215                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x1236:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	57                              # DW_AT_decl_line
	.long	7223                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x123d:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.long	7239                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x1244:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	59                              # DW_AT_decl_line
	.long	7255                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x124b:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	60                              # DW_AT_decl_line
	.long	7271                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x1252:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	62                              # DW_AT_decl_line
	.long	7287                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x1259:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	63                              # DW_AT_decl_line
	.long	7303                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x1260:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	65                              # DW_AT_decl_line
	.long	7311                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x1267:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	66                              # DW_AT_decl_line
	.long	7327                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x126e:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	67                              # DW_AT_decl_line
	.long	7347                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x1275:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	68                              # DW_AT_decl_line
	.long	7363                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x127c:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	70                              # DW_AT_decl_line
	.long	7379                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x1283:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	71                              # DW_AT_decl_line
	.long	7387                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x128a:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	72                              # DW_AT_decl_line
	.long	7395                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x1291:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
	.long	7403                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x1298:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	75                              # DW_AT_decl_line
	.long	7411                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x129f:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	76                              # DW_AT_decl_line
	.long	7427                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x12a6:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	77                              # DW_AT_decl_line
	.long	7443                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x12ad:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	78                              # DW_AT_decl_line
	.long	7459                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x12b4:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	80                              # DW_AT_decl_line
	.long	7475                            # DW_AT_import
	.byte	51                              # Abbrev [51] 0x12bb:0x7 DW_TAG_imported_declaration
	.byte	7                               # DW_AT_decl_file
	.byte	81                              # DW_AT_decl_line
	.long	7491                            # DW_AT_import
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0x12c3:0x4 DW_TAG_base_type
	.byte	70                              # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	52                              # Abbrev [52] 0x12c7:0x13 DW_TAG_subprogram
	.byte	91                              # DW_AT_low_pc
	.long	.Lfunc_end90-.Lfunc_begin90     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3663                            # DW_AT_specification
	.byte	16                              # Abbrev [16] 0x12d3:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	52                              # Abbrev [52] 0x12da:0x13 DW_TAG_subprogram
	.byte	92                              # DW_AT_low_pc
	.long	.Lfunc_end91-.Lfunc_begin91     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3689                            # DW_AT_specification
	.byte	16                              # Abbrev [16] 0x12e6:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	52                              # Abbrev [52] 0x12ed:0x27 DW_TAG_subprogram
	.byte	93                              # DW_AT_low_pc
	.long	.Lfunc_end92-.Lfunc_begin92     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3715                            # DW_AT_specification
	.byte	22                              # Abbrev [22] 0x12f9:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	1                               # DW_AT_decl_file
	.byte	62                              # DW_AT_decl_line
	.long	4587                            # DW_AT_type
	.byte	22                              # Abbrev [22] 0x1303:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	62                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	16                              # Abbrev [16] 0x130d:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	52                              # Abbrev [52] 0x1314:0x27 DW_TAG_subprogram
	.byte	94                              # DW_AT_low_pc
	.long	.Lfunc_end93-.Lfunc_begin93     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3737                            # DW_AT_specification
	.byte	22                              # Abbrev [22] 0x1320:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	1                               # DW_AT_decl_file
	.byte	66                              # DW_AT_decl_line
	.long	4587                            # DW_AT_type
	.byte	22                              # Abbrev [22] 0x132a:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	66                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	16                              # Abbrev [16] 0x1334:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	52                              # Abbrev [52] 0x133b:0x13 DW_TAG_subprogram
	.byte	95                              # DW_AT_low_pc
	.long	.Lfunc_end94-.Lfunc_begin94     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3759                            # DW_AT_specification
	.byte	16                              # Abbrev [16] 0x1347:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x134e:0x2f DW_TAG_subprogram
	.byte	96                              # DW_AT_low_pc
	.long	.Lfunc_end95-.Lfunc_begin95     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	300                             # DW_AT_linkage_name
	.short	301                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0x135c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9742                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1368:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9758                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1374:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1376:0x5 DW_TAG_template_type_parameter
	.long	377                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x137d:0x2f DW_TAG_subprogram
	.byte	97                              # DW_AT_low_pc
	.long	.Lfunc_end96-.Lfunc_begin96     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	302                             # DW_AT_linkage_name
	.short	303                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x138b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9775                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1397:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9791                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x13a3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x13a5:0x5 DW_TAG_template_type_parameter
	.long	7757                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x13ac:0x2f DW_TAG_subprogram
	.byte	98                              # DW_AT_low_pc
	.long	.Lfunc_end97-.Lfunc_begin97     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	304                             # DW_AT_linkage_name
	.short	305                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x13ba:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9808                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x13c6:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9824                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x13d2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x13d4:0x5 DW_TAG_template_type_parameter
	.long	7762                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x13db:0x13 DW_TAG_subprogram
	.byte	99                              # DW_AT_low_pc
	.long	.Lfunc_end98-.Lfunc_begin98     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	307                             # DW_AT_linkage_name
	.short	308                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	70                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	13                              # Abbrev [13] 0x13e9:0x4 DW_TAG_GNU_template_template_param
	.byte	20                              # DW_AT_name
	.short	306                             # DW_AT_GNU_template_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x13ee:0x1a DW_TAG_subprogram
	.byte	100                             # DW_AT_low_pc
	.long	.Lfunc_end99-.Lfunc_begin99     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	309                             # DW_AT_linkage_name
	.short	310                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	71                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	13                              # Abbrev [13] 0x13fc:0x4 DW_TAG_GNU_template_template_param
	.byte	20                              # DW_AT_name
	.short	306                             # DW_AT_GNU_template_name
	.byte	41                              # Abbrev [41] 0x1400:0x7 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.short	283                             # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1408:0x34 DW_TAG_subprogram
	.byte	102                             # DW_AT_low_pc
	.long	.Lfunc_end101-.Lfunc_begin101   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	314                             # DW_AT_linkage_name
	.short	315                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x1416:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9841                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1422:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9862                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x142e:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1430:0x5 DW_TAG_template_type_parameter
	.long	7533                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1435:0x5 DW_TAG_template_type_parameter
	.long	7767                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x143c:0x2f DW_TAG_subprogram
	.byte	103                             # DW_AT_low_pc
	.long	.Lfunc_end102-.Lfunc_begin102   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	317                             # DW_AT_linkage_name
	.short	318                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x144a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9884                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1456:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9900                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1462:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1464:0x5 DW_TAG_template_type_parameter
	.long	7772                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x146b:0x13 DW_TAG_subprogram
	.byte	104                             # DW_AT_low_pc
	.long	.Lfunc_end103-.Lfunc_begin103   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	320                             # DW_AT_linkage_name
	.short	321                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	70                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	13                              # Abbrev [13] 0x1479:0x4 DW_TAG_GNU_template_template_param
	.byte	20                              # DW_AT_name
	.short	319                             # DW_AT_GNU_template_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x147e:0x2f DW_TAG_subprogram
	.byte	105                             # DW_AT_low_pc
	.long	.Lfunc_end104-.Lfunc_begin104   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	322                             # DW_AT_linkage_name
	.short	323                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x148c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9917                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1498:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9933                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x14a4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x14a6:0x5 DW_TAG_template_type_parameter
	.long	7786                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x14ad:0x39 DW_TAG_subprogram
	.byte	106                             # DW_AT_low_pc
	.long	.Lfunc_end105-.Lfunc_begin105   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	324                             # DW_AT_linkage_name
	.short	325                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x14bb:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	9950                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x14c7:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	9976                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x14d3:0x12 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x14d5:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	30                              # Abbrev [30] 0x14da:0x5 DW_TAG_template_type_parameter
	.long	7187                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x14df:0x5 DW_TAG_template_type_parameter
	.long	7791                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x14e6:0x2f DW_TAG_subprogram
	.byte	107                             # DW_AT_low_pc
	.long	.Lfunc_end106-.Lfunc_begin106   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	326                             # DW_AT_linkage_name
	.short	327                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x14f4:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10003                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1500:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10019                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x150c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x150e:0x5 DW_TAG_template_type_parameter
	.long	7796                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1515:0x2f DW_TAG_subprogram
	.byte	108                             # DW_AT_low_pc
	.long	.Lfunc_end107-.Lfunc_begin107   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	328                             # DW_AT_linkage_name
	.short	329                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x1523:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10036                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x152f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10052                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x153b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x153d:0x5 DW_TAG_template_type_parameter
	.long	7808                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1544:0x2f DW_TAG_subprogram
	.byte	109                             # DW_AT_low_pc
	.long	.Lfunc_end108-.Lfunc_begin108   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	330                             # DW_AT_linkage_name
	.short	331                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x1552:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10069                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x155e:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10085                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x156a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x156c:0x5 DW_TAG_template_type_parameter
	.long	7818                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x1573:0x2f DW_TAG_subprogram
	.byte	110                             # DW_AT_low_pc
	.long	.Lfunc_end109-.Lfunc_begin109   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	333                             # DW_AT_linkage_name
	.short	334                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0x1581:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10102                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x158d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10118                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1599:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x159b:0x5 DW_TAG_template_type_parameter
	.long	7824                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x15a2:0x5 DW_TAG_pointer_type
	.long	207                             # DW_AT_type
	.byte	53                              # Abbrev [53] 0x15a7:0x1f DW_TAG_subprogram
	.byte	111                             # DW_AT_low_pc
	.long	.Lfunc_end110-.Lfunc_begin110   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	5561                            # DW_AT_object_pointer
	.short	335                             # DW_AT_linkage_name
	.long	213                             # DW_AT_specification
	.byte	48                              # Abbrev [48] 0x15b9:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	10135                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	19                              # Abbrev [19] 0x15c3:0x2 DW_TAG_template_type_parameter
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x15c6:0x2f DW_TAG_subprogram
	.byte	112                             # DW_AT_low_pc
	.long	.Lfunc_end111-.Lfunc_begin111   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	336                             # DW_AT_linkage_name
	.short	337                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x15d4:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10140                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x15e0:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10156                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x15ec:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x15ee:0x5 DW_TAG_template_type_parameter
	.long	7840                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x15f5:0x2f DW_TAG_subprogram
	.byte	113                             # DW_AT_low_pc
	.long	.Lfunc_end112-.Lfunc_begin112   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	338                             # DW_AT_linkage_name
	.short	339                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x1603:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10173                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x160f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10189                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x161b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x161d:0x5 DW_TAG_template_type_parameter
	.long	7866                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1624:0x2f DW_TAG_subprogram
	.byte	114                             # DW_AT_low_pc
	.long	.Lfunc_end113-.Lfunc_begin113   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	340                             # DW_AT_linkage_name
	.short	341                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x1632:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10206                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x163e:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10222                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x164a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x164c:0x5 DW_TAG_template_type_parameter
	.long	7892                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	54                              # Abbrev [54] 0x1653:0x19 DW_TAG_subprogram
	.byte	115                             # DW_AT_low_pc
	.long	.Lfunc_end114-.Lfunc_begin114   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	342                             # DW_AT_linkage_name
	.short	343                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	83                              # DW_AT_decl_line
	.long	7710                            # DW_AT_type
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x1665:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x166c:0x2f DW_TAG_subprogram
	.byte	116                             # DW_AT_low_pc
	.long	.Lfunc_end115-.Lfunc_begin115   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	344                             # DW_AT_linkage_name
	.short	345                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x167a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10239                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1686:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10255                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1692:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1694:0x5 DW_TAG_template_type_parameter
	.long	7918                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x169b:0x2f DW_TAG_subprogram
	.byte	117                             # DW_AT_low_pc
	.long	.Lfunc_end116-.Lfunc_begin116   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	346                             # DW_AT_linkage_name
	.short	347                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x16a9:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10272                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x16b5:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10288                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x16c1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x16c3:0x5 DW_TAG_template_type_parameter
	.long	7923                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x16ca:0x2f DW_TAG_subprogram
	.byte	118                             # DW_AT_low_pc
	.long	.Lfunc_end117-.Lfunc_begin117   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	348                             # DW_AT_linkage_name
	.short	349                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x16d8:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10305                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x16e4:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10321                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x16f0:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x16f2:0x5 DW_TAG_template_type_parameter
	.long	7945                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x16f9:0x2f DW_TAG_subprogram
	.byte	119                             # DW_AT_low_pc
	.long	.Lfunc_end118-.Lfunc_begin118   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	350                             # DW_AT_linkage_name
	.short	351                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x1707:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10338                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1713:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10354                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x171f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1721:0x5 DW_TAG_template_type_parameter
	.long	7951                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1728:0x2f DW_TAG_subprogram
	.byte	120                             # DW_AT_low_pc
	.long	.Lfunc_end119-.Lfunc_begin119   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	352                             # DW_AT_linkage_name
	.short	353                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x1736:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10371                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1742:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10387                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x174e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1750:0x5 DW_TAG_template_type_parameter
	.long	7957                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1757:0x2f DW_TAG_subprogram
	.byte	121                             # DW_AT_low_pc
	.long	.Lfunc_end120-.Lfunc_begin120   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	354                             # DW_AT_linkage_name
	.short	355                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x1765:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10404                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1771:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10420                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x177d:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x177f:0x5 DW_TAG_template_type_parameter
	.long	7967                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1786:0x2f DW_TAG_subprogram
	.byte	122                             # DW_AT_low_pc
	.long	.Lfunc_end121-.Lfunc_begin121   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	356                             # DW_AT_linkage_name
	.short	357                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x1794:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10437                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x17a0:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10453                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x17ac:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x17ae:0x5 DW_TAG_template_type_parameter
	.long	7984                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x17b5:0x2f DW_TAG_subprogram
	.byte	123                             # DW_AT_low_pc
	.long	.Lfunc_end122-.Lfunc_begin122   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	358                             # DW_AT_linkage_name
	.short	359                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x17c3:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10470                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x17cf:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10486                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x17db:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x17dd:0x5 DW_TAG_template_type_parameter
	.long	7989                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x17e4:0x2f DW_TAG_subprogram
	.byte	124                             # DW_AT_low_pc
	.long	.Lfunc_end123-.Lfunc_begin123   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	360                             # DW_AT_linkage_name
	.short	361                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x17f2:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10503                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x17fe:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10519                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x180a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x180c:0x5 DW_TAG_template_type_parameter
	.long	8020                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1813:0x2f DW_TAG_subprogram
	.byte	125                             # DW_AT_low_pc
	.long	.Lfunc_end124-.Lfunc_begin124   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	362                             # DW_AT_linkage_name
	.short	363                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x1821:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10536                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x182d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10552                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1839:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x183b:0x5 DW_TAG_template_type_parameter
	.long	8043                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1842:0x2f DW_TAG_subprogram
	.byte	126                             # DW_AT_low_pc
	.long	.Lfunc_end125-.Lfunc_begin125   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	364                             # DW_AT_linkage_name
	.short	365                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x1850:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10569                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x185c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10585                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1868:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x186a:0x5 DW_TAG_template_type_parameter
	.long	7710                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x1871:0x2f DW_TAG_subprogram
	.byte	127                             # DW_AT_low_pc
	.long	.Lfunc_end126-.Lfunc_begin126   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	366                             # DW_AT_linkage_name
	.short	367                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0x187f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10602                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x188b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10618                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1897:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1899:0x5 DW_TAG_template_type_parameter
	.long	8055                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x18a0:0x30 DW_TAG_subprogram
	.ascii	"\200\001"                      # DW_AT_low_pc
	.long	.Lfunc_end127-.Lfunc_begin127   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	368                             # DW_AT_linkage_name
	.short	369                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0x18af:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10635                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x18bb:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10651                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x18c7:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x18c9:0x5 DW_TAG_template_type_parameter
	.long	8062                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x18d0:0x30 DW_TAG_subprogram
	.ascii	"\201\001"                      # DW_AT_low_pc
	.long	.Lfunc_end128-.Lfunc_begin128   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	370                             # DW_AT_linkage_name
	.short	371                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0x18df:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10668                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x18eb:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10684                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x18f7:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x18f9:0x5 DW_TAG_template_type_parameter
	.long	8074                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1900:0x16 DW_TAG_subprogram
	.ascii	"\202\001"                      # DW_AT_low_pc
	.long	.Lfunc_end129-.Lfunc_begin129   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	372                             # DW_AT_linkage_name
	.short	373                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	88                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x190f:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1916:0x1e DW_TAG_subprogram
	.ascii	"\203\001"                      # DW_AT_low_pc
	.long	.Lfunc_end130-.Lfunc_begin130   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	376                             # DW_AT_linkage_name
	.short	377                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	98                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x1925:0x6 DW_TAG_template_type_parameter
	.long	8081                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	55                              # Abbrev [55] 0x192b:0x8 DW_TAG_template_value_parameter
	.long	8081                            # DW_AT_type
	.short	375                             # DW_AT_name
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1934:0x1e DW_TAG_subprogram
	.ascii	"\204\001"                      # DW_AT_low_pc
	.long	.Lfunc_end131-.Lfunc_begin131   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	379                             # DW_AT_linkage_name
	.short	380                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	98                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x1943:0x6 DW_TAG_template_type_parameter
	.long	8086                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	56                              # Abbrev [56] 0x1949:0x8 DW_TAG_template_value_parameter
	.long	8091                            # DW_AT_type
	.short	375                             # DW_AT_name
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1952:0x26 DW_TAG_subprogram
	.ascii	"\205\001"                      # DW_AT_low_pc
	.long	.Lfunc_end132-.Lfunc_begin132   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	381                             # DW_AT_linkage_name
	.short	382                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	98                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x1961:0x6 DW_TAG_template_type_parameter
	.long	8096                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	57                              # Abbrev [57] 0x1967:0x10 DW_TAG_template_value_parameter
	.long	8096                            # DW_AT_type
	.short	375                             # DW_AT_name
	.byte	8                               # DW_AT_const_value
	.byte	2
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1978:0x26 DW_TAG_subprogram
	.ascii	"\206\001"                      # DW_AT_low_pc
	.long	.Lfunc_end133-.Lfunc_begin133   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	383                             # DW_AT_linkage_name
	.short	384                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	98                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0x1987:0x6 DW_TAG_template_type_parameter
	.long	8101                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	57                              # Abbrev [57] 0x198d:0x10 DW_TAG_template_value_parameter
	.long	8106                            # DW_AT_type
	.short	375                             # DW_AT_name
	.byte	8                               # DW_AT_const_value
	.byte	2
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x199e:0x30 DW_TAG_subprogram
	.ascii	"\207\001"                      # DW_AT_low_pc
	.long	.Lfunc_end134-.Lfunc_begin134   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	386                             # DW_AT_linkage_name
	.short	387                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x19ad:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10701                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x19b9:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10717                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x19c5:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x19c7:0x5 DW_TAG_template_type_parameter
	.long	8111                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x19ce:0x30 DW_TAG_subprogram
	.ascii	"\210\001"                      # DW_AT_low_pc
	.long	.Lfunc_end135-.Lfunc_begin135   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	388                             # DW_AT_linkage_name
	.short	389                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x19dd:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10734                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x19e9:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10750                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x19f5:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x19f7:0x5 DW_TAG_template_type_parameter
	.long	8133                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x19fe:0x30 DW_TAG_subprogram
	.ascii	"\211\001"                      # DW_AT_low_pc
	.long	.Lfunc_end136-.Lfunc_begin136   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	390                             # DW_AT_linkage_name
	.short	391                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x1a0d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10767                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1a19:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10783                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1a25:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1a27:0x5 DW_TAG_template_type_parameter
	.long	8142                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1a2e:0x30 DW_TAG_subprogram
	.ascii	"\212\001"                      # DW_AT_low_pc
	.long	.Lfunc_end137-.Lfunc_begin137   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	392                             # DW_AT_linkage_name
	.short	393                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x1a3d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10800                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1a49:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10816                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1a55:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1a57:0x5 DW_TAG_template_type_parameter
	.long	8144                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x1a5e:0x16 DW_TAG_subprogram
	.ascii	"\213\001"                      # DW_AT_low_pc
	.long	.Lfunc_end138-.Lfunc_begin138   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	394                             # DW_AT_linkage_name
	.short	395                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	96                              # DW_AT_decl_line
	.byte	34                              # Abbrev [34] 0x1a6d:0x6 DW_TAG_template_value_parameter
	.long	134                             # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x1a74:0x12 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	84                              # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	75                              # DW_AT_decl_line
	.byte	18                              # Abbrev [18] 0x1a7a:0xb DW_TAG_subprogram
	.byte	82                              # DW_AT_linkage_name
	.byte	83                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	76                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	20                              # Abbrev [20] 0x1a7f:0x5 DW_TAG_formal_parameter
	.long	6790                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x1a86:0x5 DW_TAG_pointer_type
	.long	6772                            # DW_AT_type
	.byte	58                              # Abbrev [58] 0x1a8b:0x21 DW_TAG_subprogram
	.ascii	"\214\001"                      # DW_AT_low_pc
	.long	.Lfunc_end139-.Lfunc_begin139   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6814                            # DW_AT_object_pointer
	.short	256                             # DW_AT_decl_line
	.long	6778                            # DW_AT_specification
	.byte	48                              # Abbrev [48] 0x1a9e:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	509                             # DW_AT_name
	.long	10833                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	7                               # Abbrev [7] 0x1aa8:0x3 DW_TAG_structure_type
	.short	299                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x1aac:0x30 DW_TAG_subprogram
	.ascii	"\215\001"                      # DW_AT_low_pc
	.long	.Lfunc_end140-.Lfunc_begin140   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	396                             # DW_AT_linkage_name
	.short	301                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0x1abb:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10838                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1ac7:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10854                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1ad3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1ad5:0x5 DW_TAG_template_type_parameter
	.long	6824                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1adc:0x30 DW_TAG_subprogram
	.ascii	"\216\001"                      # DW_AT_low_pc
	.long	.Lfunc_end141-.Lfunc_begin141   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	397                             # DW_AT_linkage_name
	.short	398                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x1aeb:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	10871                           # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1af7:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.long	10887                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1b03:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1b05:0x5 DW_TAG_template_type_parameter
	.long	8149                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	8                               # Abbrev [8] 0x1b0c:0x76 DW_TAG_namespace
	.byte	85                              # DW_AT_name
	.byte	59                              # Abbrev [59] 0x1b0e:0x2b DW_TAG_subprogram
	.ascii	"\217\001"                      # DW_AT_low_pc
	.long	.Lfunc_end142-.Lfunc_begin142   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	399                             # DW_AT_linkage_name
	.short	400                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	272                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	60                              # Abbrev [60] 0x1b1e:0xd DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	273                             # DW_AT_decl_line
	.long	6969                            # DW_AT_type
	.byte	60                              # Abbrev [60] 0x1b2b:0xd DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	126
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	274                             # DW_AT_decl_line
	.long	7013                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	61                              # Abbrev [61] 0x1b39:0x14 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	258                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.short	269                             # DW_AT_decl_line
	.byte	62                              # Abbrev [62] 0x1b41:0xb DW_TAG_member
	.short	412                             # DW_AT_name
	.long	6989                            # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.short	270                             # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	61                              # Abbrev [61] 0x1b4d:0x13 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	579                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.short	268                             # DW_AT_decl_line
	.byte	63                              # Abbrev [63] 0x1b55:0x5 DW_TAG_template_type_parameter
	.long	7009                            # DW_AT_type
                                        # DW_AT_default_value
	.byte	63                              # Abbrev [63] 0x1b5a:0x5 DW_TAG_template_type_parameter
	.long	7013                            # DW_AT_type
                                        # DW_AT_default_value
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x1b60:0x5 DW_TAG_namespace
	.byte	7                               # Abbrev [7] 0x1b61:0x3 DW_TAG_structure_type
	.short	576                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	64                              # Abbrev [64] 0x1b65:0xe DW_TAG_class_type
	.byte	5                               # DW_AT_calling_convention
	.short	578                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.short	267                             # DW_AT_decl_line
	.byte	63                              # Abbrev [63] 0x1b6d:0x5 DW_TAG_template_type_parameter
	.long	7027                            # DW_AT_type
                                        # DW_AT_default_value
	.byte	0                               # End Of Children Mark
	.byte	65                              # Abbrev [65] 0x1b73:0xe DW_TAG_structure_type
	.short	577                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	66                              # Abbrev [66] 0x1b76:0xa DW_TAG_template_value_parameter
	.long	7710                            # DW_AT_type
                                        # DW_AT_default_value
	.byte	4                               # DW_AT_location
	.byte	161
	.ascii	"\222\001"
	.byte	159
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	8                               # Abbrev [8] 0x1b82:0x40 DW_TAG_namespace
	.byte	86                              # DW_AT_name
	.byte	67                              # Abbrev [67] 0x1b84:0x10 DW_TAG_subprogram
	.ascii	"\220\001"                      # DW_AT_low_pc
	.long	.Lfunc_end143-.Lfunc_begin143   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	401                             # DW_AT_linkage_name
	.short	402                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	284                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	59                              # Abbrev [59] 0x1b94:0x19 DW_TAG_subprogram
	.ascii	"\221\001"                      # DW_AT_low_pc
	.long	.Lfunc_end144-.Lfunc_begin144   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	406                             # DW_AT_linkage_name
	.short	407                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	283                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	56                              # Abbrev [56] 0x1ba4:0x8 DW_TAG_template_value_parameter
	.long	8165                            # DW_AT_type
	.short	405                             # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	61                              # Abbrev [61] 0x1bad:0x14 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	404                             # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.short	280                             # DW_AT_decl_line
	.byte	62                              # Abbrev [62] 0x1bb5:0xb DW_TAG_member
	.short	403                             # DW_AT_name
	.long	54                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.short	281                             # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	68                              # Abbrev [68] 0x1bc2:0x8 DW_TAG_typedef
	.long	7114                            # DW_AT_type
	.byte	87                              # DW_AT_name
	.byte	3                               # DW_AT_decl_file
	.byte	24                              # DW_AT_decl_line
	.byte	69                              # Abbrev [69] 0x1bca:0x1 DW_TAG_structure_type
                                        # DW_AT_declaration
	.byte	68                              # Abbrev [68] 0x1bcb:0x8 DW_TAG_typedef
	.long	7123                            # DW_AT_type
	.byte	90                              # DW_AT_name
	.byte	6                               # DW_AT_decl_file
	.byte	24                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1bd3:0x8 DW_TAG_typedef
	.long	7131                            # DW_AT_type
	.byte	89                              # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x1bdb:0x4 DW_TAG_base_type
	.byte	88                              # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	68                              # Abbrev [68] 0x1bdf:0x8 DW_TAG_typedef
	.long	7143                            # DW_AT_type
	.byte	93                              # DW_AT_name
	.byte	6                               # DW_AT_decl_file
	.byte	25                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1be7:0x8 DW_TAG_typedef
	.long	7151                            # DW_AT_type
	.byte	92                              # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	39                              # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x1bef:0x4 DW_TAG_base_type
	.byte	91                              # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	2                               # DW_AT_byte_size
	.byte	68                              # Abbrev [68] 0x1bf3:0x8 DW_TAG_typedef
	.long	7163                            # DW_AT_type
	.byte	95                              # DW_AT_name
	.byte	6                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1bfb:0x8 DW_TAG_typedef
	.long	54                              # DW_AT_type
	.byte	94                              # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c03:0x8 DW_TAG_typedef
	.long	7179                            # DW_AT_type
	.byte	98                              # DW_AT_name
	.byte	6                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c0b:0x8 DW_TAG_typedef
	.long	7187                            # DW_AT_type
	.byte	97                              # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	44                              # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x1c13:0x4 DW_TAG_base_type
	.byte	96                              # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	68                              # Abbrev [68] 0x1c17:0x8 DW_TAG_typedef
	.long	7131                            # DW_AT_type
	.byte	99                              # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c1f:0x8 DW_TAG_typedef
	.long	7187                            # DW_AT_type
	.byte	100                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	60                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c27:0x8 DW_TAG_typedef
	.long	7187                            # DW_AT_type
	.byte	101                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	61                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c2f:0x8 DW_TAG_typedef
	.long	7187                            # DW_AT_type
	.byte	102                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	62                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c37:0x8 DW_TAG_typedef
	.long	7231                            # DW_AT_type
	.byte	104                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c3f:0x8 DW_TAG_typedef
	.long	7123                            # DW_AT_type
	.byte	103                             # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	52                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c47:0x8 DW_TAG_typedef
	.long	7247                            # DW_AT_type
	.byte	106                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	44                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c4f:0x8 DW_TAG_typedef
	.long	7143                            # DW_AT_type
	.byte	105                             # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	54                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c57:0x8 DW_TAG_typedef
	.long	7263                            # DW_AT_type
	.byte	108                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	45                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c5f:0x8 DW_TAG_typedef
	.long	7163                            # DW_AT_type
	.byte	107                             # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	56                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c67:0x8 DW_TAG_typedef
	.long	7279                            # DW_AT_type
	.byte	110                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	46                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c6f:0x8 DW_TAG_typedef
	.long	7179                            # DW_AT_type
	.byte	109                             # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c77:0x8 DW_TAG_typedef
	.long	7295                            # DW_AT_type
	.byte	112                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	101                             # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c7f:0x8 DW_TAG_typedef
	.long	7187                            # DW_AT_type
	.byte	111                             # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	72                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c87:0x8 DW_TAG_typedef
	.long	7187                            # DW_AT_type
	.byte	113                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	87                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c8f:0x8 DW_TAG_typedef
	.long	7319                            # DW_AT_type
	.byte	115                             # DW_AT_name
	.byte	9                               # DW_AT_decl_file
	.byte	24                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c97:0x8 DW_TAG_typedef
	.long	179                             # DW_AT_type
	.byte	114                             # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	38                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1c9f:0x8 DW_TAG_typedef
	.long	7335                            # DW_AT_type
	.byte	118                             # DW_AT_name
	.byte	9                               # DW_AT_decl_file
	.byte	25                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1ca7:0x8 DW_TAG_typedef
	.long	7343                            # DW_AT_type
	.byte	117                             # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	40                              # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x1caf:0x4 DW_TAG_base_type
	.byte	116                             # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	2                               # DW_AT_byte_size
	.byte	68                              # Abbrev [68] 0x1cb3:0x8 DW_TAG_typedef
	.long	7355                            # DW_AT_type
	.byte	120                             # DW_AT_name
	.byte	9                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1cbb:0x8 DW_TAG_typedef
	.long	76                              # DW_AT_type
	.byte	119                             # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	42                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1cc3:0x8 DW_TAG_typedef
	.long	7371                            # DW_AT_type
	.byte	122                             # DW_AT_name
	.byte	9                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1ccb:0x8 DW_TAG_typedef
	.long	4803                            # DW_AT_type
	.byte	121                             # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	45                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1cd3:0x8 DW_TAG_typedef
	.long	179                             # DW_AT_type
	.byte	123                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	71                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1cdb:0x8 DW_TAG_typedef
	.long	4803                            # DW_AT_type
	.byte	124                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1ce3:0x8 DW_TAG_typedef
	.long	4803                            # DW_AT_type
	.byte	125                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	74                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1ceb:0x8 DW_TAG_typedef
	.long	4803                            # DW_AT_type
	.byte	126                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	75                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1cf3:0x8 DW_TAG_typedef
	.long	7419                            # DW_AT_type
	.byte	128                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	49                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1cfb:0x8 DW_TAG_typedef
	.long	7319                            # DW_AT_type
	.byte	127                             # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	53                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1d03:0x8 DW_TAG_typedef
	.long	7435                            # DW_AT_type
	.byte	130                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	50                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1d0b:0x8 DW_TAG_typedef
	.long	7335                            # DW_AT_type
	.byte	129                             # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	55                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1d13:0x8 DW_TAG_typedef
	.long	7451                            # DW_AT_type
	.byte	132                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	51                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1d1b:0x8 DW_TAG_typedef
	.long	7355                            # DW_AT_type
	.byte	131                             # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	57                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1d23:0x8 DW_TAG_typedef
	.long	7467                            # DW_AT_type
	.byte	134                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	52                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1d2b:0x8 DW_TAG_typedef
	.long	7371                            # DW_AT_type
	.byte	133                             # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	59                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1d33:0x8 DW_TAG_typedef
	.long	7483                            # DW_AT_type
	.byte	136                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	102                             # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1d3b:0x8 DW_TAG_typedef
	.long	4803                            # DW_AT_type
	.byte	135                             # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
	.byte	68                              # Abbrev [68] 0x1d43:0x8 DW_TAG_typedef
	.long	4803                            # DW_AT_type
	.byte	137                             # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	90                              # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x1d4b:0x4 DW_TAG_base_type
	.byte	147                             # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x1d4f:0x4 DW_TAG_base_type
	.byte	156                             # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x1d53:0x4 DW_TAG_base_type
	.byte	159                             # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	14                              # Abbrev [14] 0x1d57:0x2 DW_TAG_structure_type
	.byte	162                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	46                              # Abbrev [46] 0x1d59:0x5 DW_TAG_pointer_type
	.long	171                             # DW_AT_type
	.byte	15                              # Abbrev [15] 0x1d5e:0xf DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	172                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x1d64:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1d66:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x1d6d:0x5 DW_TAG_pointer_type
	.long	54                              # DW_AT_type
	.byte	70                              # Abbrev [70] 0x1d72:0x5 DW_TAG_reference_type
	.long	54                              # DW_AT_type
	.byte	71                              # Abbrev [71] 0x1d77:0x5 DW_TAG_rvalue_reference_type
	.long	54                              # DW_AT_type
	.byte	72                              # Abbrev [72] 0x1d7c:0x5 DW_TAG_const_type
	.long	54                              # DW_AT_type
	.byte	73                              # Abbrev [73] 0x1d81:0xc DW_TAG_array_type
	.long	54                              # DW_AT_type
	.byte	74                              # Abbrev [74] 0x1d86:0x6 DW_TAG_subrange_type
	.long	7565                            # DW_AT_type
	.byte	3                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	75                              # Abbrev [75] 0x1d8d:0x4 DW_TAG_base_type
	.byte	185                             # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	7                               # DW_AT_encoding
	.byte	15                              # Abbrev [15] 0x1d91:0x9 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	190                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.byte	14                              # Abbrev [14] 0x1d97:0x2 DW_TAG_structure_type
	.byte	191                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0x1d9a:0x4 DW_TAG_base_type
	.byte	229                             # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x1d9e:0x4 DW_TAG_base_type
	.byte	232                             # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	16                              # DW_AT_byte_size
	.byte	76                              # Abbrev [76] 0x1da2:0x10 DW_TAG_structure_type
	.byte	239                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	16                              # Abbrev [16] 0x1da4:0x6 DW_TAG_template_type_parameter
	.long	183                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	17                              # Abbrev [17] 0x1daa:0x7 DW_TAG_template_value_parameter
	.long	203                             # DW_AT_type
	.byte	22                              # DW_AT_name
                                        # DW_AT_default_value
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x1db2:0x10 DW_TAG_structure_type
	.byte	245                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	16                              # Abbrev [16] 0x1db4:0x6 DW_TAG_template_type_parameter
	.long	7618                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	17                              # Abbrev [17] 0x1dba:0x7 DW_TAG_template_value_parameter
	.long	203                             # DW_AT_type
	.byte	22                              # DW_AT_name
                                        # DW_AT_default_value
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x1dc2:0x14 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	244                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	25                              # DW_AT_decl_line
	.byte	16                              # Abbrev [16] 0x1dc8:0x6 DW_TAG_template_type_parameter
	.long	367                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	17                              # Abbrev [17] 0x1dce:0x7 DW_TAG_template_value_parameter
	.long	203                             # DW_AT_type
	.byte	22                              # DW_AT_name
                                        # DW_AT_default_value
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	77                              # Abbrev [77] 0x1dd6:0xb DW_TAG_subroutine_type
	.long	54                              # DW_AT_type
	.byte	44                              # Abbrev [44] 0x1ddb:0x5 DW_TAG_formal_parameter
	.long	3935                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	78                              # Abbrev [78] 0x1de1:0x3 DW_TAG_subroutine_type
	.byte	79                              # Abbrev [79] 0x1de2:0x1 DW_TAG_unspecified_parameters
	.byte	0                               # End Of Children Mark
	.byte	78                              # Abbrev [78] 0x1de4:0x8 DW_TAG_subroutine_type
	.byte	44                              # Abbrev [44] 0x1de5:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	79                              # Abbrev [79] 0x1dea:0x1 DW_TAG_unspecified_parameters
	.byte	0                               # End Of Children Mark
	.byte	70                              # Abbrev [70] 0x1dec:0x5 DW_TAG_reference_type
	.long	7548                            # DW_AT_type
	.byte	70                              # Abbrev [70] 0x1df1:0x5 DW_TAG_reference_type
	.long	7670                            # DW_AT_type
	.byte	46                              # Abbrev [46] 0x1df6:0x5 DW_TAG_pointer_type
	.long	7548                            # DW_AT_type
	.byte	80                              # Abbrev [80] 0x1dfb:0x3 DW_TAG_unspecified_type
	.short	261                             # DW_AT_name
	.byte	46                              # Abbrev [46] 0x1dfe:0x5 DW_TAG_pointer_type
	.long	7187                            # DW_AT_type
	.byte	46                              # Abbrev [46] 0x1e03:0x5 DW_TAG_pointer_type
	.long	7511                            # DW_AT_type
	.byte	72                              # Abbrev [72] 0x1e08:0x5 DW_TAG_const_type
	.long	4587                            # DW_AT_type
	.byte	46                              # Abbrev [46] 0x1e0d:0x5 DW_TAG_pointer_type
	.long	7698                            # DW_AT_type
	.byte	72                              # Abbrev [72] 0x1e12:0x5 DW_TAG_const_type
	.long	7703                            # DW_AT_type
	.byte	46                              # Abbrev [46] 0x1e17:0x5 DW_TAG_pointer_type
	.long	7708                            # DW_AT_type
	.byte	81                              # Abbrev [81] 0x1e1c:0x1 DW_TAG_const_type
	.byte	82                              # Abbrev [82] 0x1e1d:0x1 DW_TAG_subroutine_type
	.byte	46                              # Abbrev [46] 0x1e1e:0x5 DW_TAG_pointer_type
	.long	7709                            # DW_AT_type
	.byte	46                              # Abbrev [46] 0x1e23:0x5 DW_TAG_pointer_type
	.long	367                             # DW_AT_type
	.byte	46                              # Abbrev [46] 0x1e28:0x5 DW_TAG_pointer_type
	.long	372                             # DW_AT_type
	.byte	46                              # Abbrev [46] 0x1e2d:0x5 DW_TAG_pointer_type
	.long	7730                            # DW_AT_type
	.byte	83                              # Abbrev [83] 0x1e32:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	294                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x1e39:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1e3b:0x5 DW_TAG_template_type_parameter
	.long	7533                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	73                              # Abbrev [73] 0x1e42:0xb DW_TAG_array_type
	.long	7533                            # DW_AT_type
	.byte	84                              # Abbrev [84] 0x1e47:0x5 DW_TAG_subrange_type
	.long	7565                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	70                              # Abbrev [70] 0x1e4d:0x5 DW_TAG_reference_type
	.long	7553                            # DW_AT_type
	.byte	46                              # Abbrev [46] 0x1e52:0x5 DW_TAG_pointer_type
	.long	7553                            # DW_AT_type
	.byte	46                              # Abbrev [46] 0x1e57:0x5 DW_TAG_pointer_type
	.long	7675                            # DW_AT_type
	.byte	83                              # Abbrev [83] 0x1e5c:0xe DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	316                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	74                              # DW_AT_decl_line
	.byte	16                              # Abbrev [16] 0x1e63:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	85                              # Abbrev [85] 0x1e6a:0x5 DW_TAG_atomic_type
	.long	54                              # DW_AT_type
	.byte	86                              # Abbrev [86] 0x1e6f:0x5 DW_TAG_volatile_type
	.long	7578                            # DW_AT_type
	.byte	87                              # Abbrev [87] 0x1e74:0xc DW_TAG_array_type
                                        # DW_AT_GNU_vector
	.long	54                              # DW_AT_type
	.byte	74                              # Abbrev [74] 0x1e79:0x6 DW_TAG_subrange_type
	.long	7565                            # DW_AT_type
	.byte	2                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	72                              # Abbrev [72] 0x1e80:0x5 DW_TAG_const_type
	.long	7813                            # DW_AT_type
	.byte	86                              # Abbrev [86] 0x1e85:0x5 DW_TAG_volatile_type
	.long	7533                            # DW_AT_type
	.byte	72                              # Abbrev [72] 0x1e8a:0x5 DW_TAG_const_type
	.long	7823                            # DW_AT_type
	.byte	88                              # Abbrev [88] 0x1e8f:0x1 DW_TAG_volatile_type
	.byte	83                              # Abbrev [83] 0x1e90:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	332                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x1e97:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1e99:0x5 DW_TAG_template_type_parameter
	.long	367                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	89                              # Abbrev [89] 0x1ea0:0x9 DW_TAG_ptr_to_member_type
	.long	7849                            # DW_AT_type
	.long	7511                            # DW_AT_containing_type
	.byte	78                              # Abbrev [78] 0x1ea9:0x7 DW_TAG_subroutine_type
	.byte	20                              # Abbrev [20] 0x1eaa:0x5 DW_TAG_formal_parameter
	.long	7856                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x1eb0:0x5 DW_TAG_pointer_type
	.long	7861                            # DW_AT_type
	.byte	72                              # Abbrev [72] 0x1eb5:0x5 DW_TAG_const_type
	.long	7511                            # DW_AT_type
	.byte	89                              # Abbrev [89] 0x1eba:0x9 DW_TAG_ptr_to_member_type
	.long	7875                            # DW_AT_type
	.long	7511                            # DW_AT_containing_type
	.byte	90                              # Abbrev [90] 0x1ec3:0x7 DW_TAG_subroutine_type
                                        # DW_AT_reference
	.byte	20                              # Abbrev [20] 0x1ec4:0x5 DW_TAG_formal_parameter
	.long	7882                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x1eca:0x5 DW_TAG_pointer_type
	.long	7887                            # DW_AT_type
	.byte	86                              # Abbrev [86] 0x1ecf:0x5 DW_TAG_volatile_type
	.long	7511                            # DW_AT_type
	.byte	89                              # Abbrev [89] 0x1ed4:0x9 DW_TAG_ptr_to_member_type
	.long	7901                            # DW_AT_type
	.long	7511                            # DW_AT_containing_type
	.byte	91                              # Abbrev [91] 0x1edd:0x7 DW_TAG_subroutine_type
                                        # DW_AT_rvalue_reference
	.byte	20                              # Abbrev [20] 0x1ede:0x5 DW_TAG_formal_parameter
	.long	7908                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x1ee4:0x5 DW_TAG_pointer_type
	.long	7913                            # DW_AT_type
	.byte	72                              # Abbrev [72] 0x1ee9:0x5 DW_TAG_const_type
	.long	7887                            # DW_AT_type
	.byte	72                              # Abbrev [72] 0x1eee:0x5 DW_TAG_const_type
	.long	7710                            # DW_AT_type
	.byte	70                              # Abbrev [70] 0x1ef3:0x5 DW_TAG_reference_type
	.long	7928                            # DW_AT_type
	.byte	72                              # Abbrev [72] 0x1ef8:0x5 DW_TAG_const_type
	.long	7933                            # DW_AT_type
	.byte	73                              # Abbrev [73] 0x1efd:0xc DW_TAG_array_type
	.long	7578                            # DW_AT_type
	.byte	74                              # Abbrev [74] 0x1f02:0x6 DW_TAG_subrange_type
	.long	7565                            # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	72                              # Abbrev [72] 0x1f09:0x5 DW_TAG_const_type
	.long	7950                            # DW_AT_type
	.byte	92                              # Abbrev [92] 0x1f0e:0x1 DW_TAG_subroutine_type
                                        # DW_AT_reference
	.byte	86                              # Abbrev [86] 0x1f0f:0x5 DW_TAG_volatile_type
	.long	7956                            # DW_AT_type
	.byte	93                              # Abbrev [93] 0x1f14:0x1 DW_TAG_subroutine_type
                                        # DW_AT_rvalue_reference
	.byte	72                              # Abbrev [72] 0x1f15:0x5 DW_TAG_const_type
	.long	7962                            # DW_AT_type
	.byte	86                              # Abbrev [86] 0x1f1a:0x5 DW_TAG_volatile_type
	.long	7709                            # DW_AT_type
	.byte	72                              # Abbrev [72] 0x1f1f:0x5 DW_TAG_const_type
	.long	7972                            # DW_AT_type
	.byte	73                              # Abbrev [73] 0x1f24:0xc DW_TAG_array_type
	.long	7533                            # DW_AT_type
	.byte	74                              # Abbrev [74] 0x1f29:0x6 DW_TAG_subrange_type
	.long	7565                            # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	70                              # Abbrev [70] 0x1f30:0x5 DW_TAG_reference_type
	.long	7967                            # DW_AT_type
	.byte	70                              # Abbrev [70] 0x1f35:0x5 DW_TAG_reference_type
	.long	7994                            # DW_AT_type
	.byte	72                              # Abbrev [72] 0x1f3a:0x5 DW_TAG_const_type
	.long	7999                            # DW_AT_type
	.byte	89                              # Abbrev [89] 0x1f3f:0x9 DW_TAG_ptr_to_member_type
	.long	8008                            # DW_AT_type
	.long	7511                            # DW_AT_containing_type
	.byte	78                              # Abbrev [78] 0x1f48:0x7 DW_TAG_subroutine_type
	.byte	20                              # Abbrev [20] 0x1f49:0x5 DW_TAG_formal_parameter
	.long	8015                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x1f4f:0x5 DW_TAG_pointer_type
	.long	7511                            # DW_AT_type
	.byte	77                              # Abbrev [77] 0x1f54:0xb DW_TAG_subroutine_type
	.long	8031                            # DW_AT_type
	.byte	44                              # Abbrev [44] 0x1f59:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x1f5f:0x5 DW_TAG_pointer_type
	.long	8036                            # DW_AT_type
	.byte	78                              # Abbrev [78] 0x1f64:0x7 DW_TAG_subroutine_type
	.byte	44                              # Abbrev [44] 0x1f65:0x5 DW_TAG_formal_parameter
	.long	3935                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	73                              # Abbrev [73] 0x1f6b:0xc DW_TAG_array_type
	.long	7518                            # DW_AT_type
	.byte	74                              # Abbrev [74] 0x1f70:0x6 DW_TAG_subrange_type
	.long	7565                            # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	78                              # Abbrev [78] 0x1f77:0x7 DW_TAG_subroutine_type
	.byte	44                              # Abbrev [44] 0x1f78:0x5 DW_TAG_formal_parameter
	.long	372                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	78                              # Abbrev [78] 0x1f7e:0xc DW_TAG_subroutine_type
	.byte	44                              # Abbrev [44] 0x1f7f:0x5 DW_TAG_formal_parameter
	.long	380                             # DW_AT_type
	.byte	44                              # Abbrev [44] 0x1f84:0x5 DW_TAG_formal_parameter
	.long	372                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	78                              # Abbrev [78] 0x1f8a:0x7 DW_TAG_subroutine_type
	.byte	44                              # Abbrev [44] 0x1f8b:0x5 DW_TAG_formal_parameter
	.long	380                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	94                              # Abbrev [94] 0x1f91:0x5 DW_TAG_base_type
	.short	374                             # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	72                              # Abbrev [72] 0x1f96:0x5 DW_TAG_const_type
	.long	8091                            # DW_AT_type
	.byte	94                              # Abbrev [94] 0x1f9b:0x5 DW_TAG_base_type
	.short	378                             # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	94                              # Abbrev [94] 0x1fa0:0x5 DW_TAG_base_type
	.short	374                             # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	16                              # DW_AT_byte_size
	.byte	72                              # Abbrev [72] 0x1fa5:0x5 DW_TAG_const_type
	.long	8106                            # DW_AT_type
	.byte	94                              # Abbrev [94] 0x1faa:0x5 DW_TAG_base_type
	.short	378                             # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	16                              # DW_AT_byte_size
	.byte	78                              # Abbrev [78] 0x1faf:0xc DW_TAG_subroutine_type
	.byte	44                              # Abbrev [44] 0x1fb0:0x5 DW_TAG_formal_parameter
	.long	8123                            # DW_AT_type
	.byte	44                              # Abbrev [44] 0x1fb5:0x5 DW_TAG_formal_parameter
	.long	8123                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x1fbb:0xa DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	385                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	43                              # Abbrev [43] 0x1fc2:0x2 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	89                              # Abbrev [89] 0x1fc5:0x9 DW_TAG_ptr_to_member_type
	.long	54                              # DW_AT_type
	.long	8123                            # DW_AT_containing_type
	.byte	95                              # Abbrev [95] 0x1fce:0x2 DW_TAG_subroutine_type
	.byte	200                             # DW_AT_calling_convention
	.byte	96                              # Abbrev [96] 0x1fd0:0x5 DW_TAG_subroutine_type
	.long	54                              # DW_AT_type
	.byte	89                              # Abbrev [89] 0x1fd5:0x9 DW_TAG_ptr_to_member_type
	.long	8158                            # DW_AT_type
	.long	6772                            # DW_AT_containing_type
	.byte	78                              # Abbrev [78] 0x1fde:0x7 DW_TAG_subroutine_type
	.byte	20                              # Abbrev [20] 0x1fdf:0x5 DW_TAG_formal_parameter
	.long	6790                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	89                              # Abbrev [89] 0x1fe5:0x9 DW_TAG_ptr_to_member_type
	.long	54                              # DW_AT_type
	.long	7085                            # DW_AT_containing_type
	.byte	83                              # Abbrev [83] 0x1fee:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	411                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	32                              # DW_AT_decl_line
	.byte	56                              # Abbrev [56] 0x1ff5:0x8 DW_TAG_template_value_parameter
	.long	76                              # DW_AT_type
	.short	410                             # DW_AT_name
	.byte	3                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	97                              # Abbrev [97] 0x1ffe:0x9 DW_TAG_typedef
	.long	7772                            # DW_AT_type
	.short	415                             # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
	.byte	83                              # Abbrev [83] 0x2007:0x12 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	419                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	92                              # DW_AT_decl_line
	.byte	98                              # Abbrev [98] 0x200e:0xa DW_TAG_member
	.short	412                             # DW_AT_name
	.long	8217                            # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	93                              # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2019:0x17 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	418                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	90                              # DW_AT_decl_line
	.byte	16                              # Abbrev [16] 0x2020:0x6 DW_TAG_template_type_parameter
	.long	59                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	29                              # Abbrev [29] 0x2026:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	198                             # DW_AT_name
	.byte	35                              # Abbrev [35] 0x2028:0x6 DW_TAG_template_value_parameter
	.long	59                              # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2030:0x5 DW_TAG_pointer_type
	.long	8245                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2035:0xc DW_TAG_structure_type
	.short	420                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2038:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x203a:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2041:0x5 DW_TAG_pointer_type
	.long	8262                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2046:0xc DW_TAG_structure_type
	.short	421                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2049:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x204b:0x5 DW_TAG_template_type_parameter
	.long	3935                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2052:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	422                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2059:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x205b:0x5 DW_TAG_template_type_parameter
	.long	203                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2062:0x5 DW_TAG_pointer_type
	.long	8295                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2067:0xc DW_TAG_structure_type
	.short	423                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x206a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x206c:0x5 DW_TAG_template_type_parameter
	.long	203                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2073:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	424                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x207a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x207c:0x5 DW_TAG_template_type_parameter
	.long	7499                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2083:0x5 DW_TAG_pointer_type
	.long	8328                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2088:0xc DW_TAG_structure_type
	.short	425                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x208b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x208d:0x5 DW_TAG_template_type_parameter
	.long	7499                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2094:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	426                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x209b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x209d:0x5 DW_TAG_template_type_parameter
	.long	7187                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x20a4:0x5 DW_TAG_pointer_type
	.long	8361                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x20a9:0xc DW_TAG_structure_type
	.short	427                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x20ac:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x20ae:0x5 DW_TAG_template_type_parameter
	.long	7187                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x20b5:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	428                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x20bc:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x20be:0x5 DW_TAG_template_type_parameter
	.long	7151                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x20c5:0x5 DW_TAG_pointer_type
	.long	8394                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x20ca:0xc DW_TAG_structure_type
	.short	429                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x20cd:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x20cf:0x5 DW_TAG_template_type_parameter
	.long	7151                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x20d6:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	430                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x20dd:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x20df:0x5 DW_TAG_template_type_parameter
	.long	76                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x20e6:0x5 DW_TAG_pointer_type
	.long	8427                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x20eb:0xc DW_TAG_structure_type
	.short	431                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x20ee:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x20f0:0x5 DW_TAG_template_type_parameter
	.long	76                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x20f7:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	432                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x20fe:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2100:0x5 DW_TAG_template_type_parameter
	.long	7503                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2107:0x5 DW_TAG_pointer_type
	.long	8460                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x210c:0xc DW_TAG_structure_type
	.short	433                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x210f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2111:0x5 DW_TAG_template_type_parameter
	.long	7503                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2118:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	434                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x211f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2121:0x5 DW_TAG_template_type_parameter
	.long	7507                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2128:0x5 DW_TAG_pointer_type
	.long	8493                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x212d:0xc DW_TAG_structure_type
	.short	435                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2130:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2132:0x5 DW_TAG_template_type_parameter
	.long	7507                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2139:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	436                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2140:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2142:0x5 DW_TAG_template_type_parameter
	.long	7511                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2149:0x5 DW_TAG_pointer_type
	.long	8526                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x214e:0xc DW_TAG_structure_type
	.short	437                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2151:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2153:0x5 DW_TAG_template_type_parameter
	.long	7511                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x215a:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	438                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2161:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2163:0x5 DW_TAG_template_type_parameter
	.long	171                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x216a:0x5 DW_TAG_pointer_type
	.long	8559                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x216f:0xc DW_TAG_structure_type
	.short	439                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2172:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2174:0x5 DW_TAG_template_type_parameter
	.long	171                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x217b:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	440                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2182:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2184:0x5 DW_TAG_template_type_parameter
	.long	7513                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x218b:0x5 DW_TAG_pointer_type
	.long	8592                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2190:0xc DW_TAG_structure_type
	.short	441                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2193:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2195:0x5 DW_TAG_template_type_parameter
	.long	7513                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x219c:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	442                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x21a3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x21a5:0x5 DW_TAG_template_type_parameter
	.long	175                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x21ac:0x5 DW_TAG_pointer_type
	.long	8625                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x21b1:0xc DW_TAG_structure_type
	.short	443                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x21b4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x21b6:0x5 DW_TAG_template_type_parameter
	.long	175                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x21bd:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	444                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x21c4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x21c6:0x5 DW_TAG_template_type_parameter
	.long	7518                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x21cd:0x5 DW_TAG_pointer_type
	.long	8658                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x21d2:0xc DW_TAG_structure_type
	.short	445                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x21d5:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x21d7:0x5 DW_TAG_template_type_parameter
	.long	7518                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x21de:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	446                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x21e5:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x21e7:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	30                              # Abbrev [30] 0x21ec:0x5 DW_TAG_template_type_parameter
	.long	3935                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x21f3:0x5 DW_TAG_pointer_type
	.long	8696                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x21f8:0x11 DW_TAG_structure_type
	.short	447                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x21fb:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x21fd:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2202:0x5 DW_TAG_template_type_parameter
	.long	3935                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2209:0x5 DW_TAG_pointer_type
	.long	8718                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x220e:0xc DW_TAG_structure_type
	.short	448                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2211:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2213:0x5 DW_TAG_template_type_parameter
	.long	7533                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x221a:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	449                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2221:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2223:0x5 DW_TAG_template_type_parameter
	.long	7538                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x222a:0x5 DW_TAG_pointer_type
	.long	8751                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x222f:0xc DW_TAG_structure_type
	.short	450                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2232:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2234:0x5 DW_TAG_template_type_parameter
	.long	7538                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x223b:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	451                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2242:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2244:0x5 DW_TAG_template_type_parameter
	.long	7543                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x224b:0x5 DW_TAG_pointer_type
	.long	8784                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2250:0xc DW_TAG_structure_type
	.short	452                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2253:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2255:0x5 DW_TAG_template_type_parameter
	.long	7543                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x225c:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	453                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2263:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2265:0x5 DW_TAG_template_type_parameter
	.long	7548                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x226c:0x5 DW_TAG_pointer_type
	.long	8817                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2271:0xc DW_TAG_structure_type
	.short	454                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2274:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2276:0x5 DW_TAG_template_type_parameter
	.long	7548                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x227d:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	455                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2284:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2286:0x5 DW_TAG_template_type_parameter
	.long	7553                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x228d:0x5 DW_TAG_pointer_type
	.long	8850                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2292:0xc DW_TAG_structure_type
	.short	456                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2295:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2297:0x5 DW_TAG_template_type_parameter
	.long	7553                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x229e:0xc DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	457                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x22a5:0x4 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x22a7:0x1 DW_TAG_template_type_parameter
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x22aa:0x5 DW_TAG_pointer_type
	.long	8879                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x22af:0x8 DW_TAG_structure_type
	.short	458                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x22b2:0x4 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x22b4:0x1 DW_TAG_template_type_parameter
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x22b7:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	459                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x22be:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x22c0:0x5 DW_TAG_template_type_parameter
	.long	7575                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x22c7:0x5 DW_TAG_pointer_type
	.long	8908                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x22cc:0xc DW_TAG_structure_type
	.short	460                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x22cf:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x22d1:0x5 DW_TAG_template_type_parameter
	.long	7575                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x22d8:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	461                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x22df:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x22e1:0x5 DW_TAG_template_type_parameter
	.long	4803                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x22e8:0x5 DW_TAG_pointer_type
	.long	8941                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x22ed:0xc DW_TAG_structure_type
	.short	462                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x22f0:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x22f2:0x5 DW_TAG_template_type_parameter
	.long	4803                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x22f9:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	463                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2300:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2302:0x5 DW_TAG_template_type_parameter
	.long	183                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2309:0x5 DW_TAG_pointer_type
	.long	8974                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x230e:0xc DW_TAG_structure_type
	.short	464                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2311:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2313:0x5 DW_TAG_template_type_parameter
	.long	183                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x231a:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	465                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2321:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2323:0x5 DW_TAG_template_type_parameter
	.long	7586                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x232a:0x5 DW_TAG_pointer_type
	.long	9007                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x232f:0xc DW_TAG_structure_type
	.short	466                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2332:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2334:0x5 DW_TAG_template_type_parameter
	.long	7586                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x233b:0x5 DW_TAG_pointer_type
	.long	9024                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2340:0xc DW_TAG_structure_type
	.short	467                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2343:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2345:0x5 DW_TAG_template_type_parameter
	.long	367                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x234c:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	468                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2353:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2355:0x5 DW_TAG_template_type_parameter
	.long	7602                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x235c:0x5 DW_TAG_pointer_type
	.long	9057                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2361:0xc DW_TAG_structure_type
	.short	469                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2364:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2366:0x5 DW_TAG_template_type_parameter
	.long	7602                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x236d:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	470                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2374:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2376:0x5 DW_TAG_template_type_parameter
	.long	7638                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x237d:0x5 DW_TAG_pointer_type
	.long	9090                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2382:0xc DW_TAG_structure_type
	.short	471                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2385:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2387:0x5 DW_TAG_template_type_parameter
	.long	7638                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x238e:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	472                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2395:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2397:0x5 DW_TAG_template_type_parameter
	.long	7649                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x239e:0x5 DW_TAG_pointer_type
	.long	9123                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x23a3:0xc DW_TAG_structure_type
	.short	473                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x23a6:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x23a8:0x5 DW_TAG_template_type_parameter
	.long	7649                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x23af:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	474                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x23b6:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x23b8:0x5 DW_TAG_template_type_parameter
	.long	7652                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x23bf:0x5 DW_TAG_pointer_type
	.long	9156                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x23c4:0xc DW_TAG_structure_type
	.short	475                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x23c7:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x23c9:0x5 DW_TAG_template_type_parameter
	.long	7652                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x23d0:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	476                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x23d7:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x23d9:0x5 DW_TAG_template_type_parameter
	.long	7660                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x23e0:0x5 DW_TAG_pointer_type
	.long	9189                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x23e5:0xc DW_TAG_structure_type
	.short	477                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x23e8:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x23ea:0x5 DW_TAG_template_type_parameter
	.long	7660                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x23f1:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	478                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x23f8:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x23fa:0x5 DW_TAG_template_type_parameter
	.long	7665                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2401:0x5 DW_TAG_pointer_type
	.long	9222                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2406:0xc DW_TAG_structure_type
	.short	479                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2409:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x240b:0x5 DW_TAG_template_type_parameter
	.long	7665                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2412:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	480                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2419:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x241b:0x5 DW_TAG_template_type_parameter
	.long	72                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2422:0x5 DW_TAG_pointer_type
	.long	9255                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2427:0xc DW_TAG_structure_type
	.short	481                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x242a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x242c:0x5 DW_TAG_template_type_parameter
	.long	72                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2433:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	482                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x243a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x243c:0x5 DW_TAG_template_type_parameter
	.long	7675                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2443:0x5 DW_TAG_pointer_type
	.long	9288                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2448:0xc DW_TAG_structure_type
	.short	483                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x244b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x244d:0x5 DW_TAG_template_type_parameter
	.long	7675                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2454:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	484                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x245b:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x245d:0x5 DW_TAG_template_type_parameter
	.long	7678                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2462:0x5 DW_TAG_template_type_parameter
	.long	7678                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2469:0x5 DW_TAG_pointer_type
	.long	9326                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x246e:0x11 DW_TAG_structure_type
	.short	485                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2471:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2473:0x5 DW_TAG_template_type_parameter
	.long	7678                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2478:0x5 DW_TAG_template_type_parameter
	.long	7678                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x247f:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	486                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2486:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2488:0x5 DW_TAG_template_type_parameter
	.long	7678                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x248d:0x5 DW_TAG_template_type_parameter
	.long	7683                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2494:0x5 DW_TAG_pointer_type
	.long	9369                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2499:0x11 DW_TAG_structure_type
	.short	487                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x249c:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x249e:0x5 DW_TAG_template_type_parameter
	.long	7678                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x24a3:0x5 DW_TAG_template_type_parameter
	.long	7683                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x24aa:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	488                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x24b1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x24b3:0x5 DW_TAG_template_type_parameter
	.long	7688                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x24ba:0x5 DW_TAG_pointer_type
	.long	9407                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x24bf:0xc DW_TAG_structure_type
	.short	489                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x24c2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x24c4:0x5 DW_TAG_template_type_parameter
	.long	7688                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x24cb:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	490                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x24d2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x24d4:0x5 DW_TAG_template_type_parameter
	.long	7693                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x24db:0x5 DW_TAG_pointer_type
	.long	9440                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x24e0:0xc DW_TAG_structure_type
	.short	491                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x24e3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x24e5:0x5 DW_TAG_template_type_parameter
	.long	7693                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x24ec:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	492                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x24f3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x24f5:0x5 DW_TAG_template_type_parameter
	.long	7709                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x24fc:0x5 DW_TAG_pointer_type
	.long	9473                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2501:0xc DW_TAG_structure_type
	.short	493                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2504:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2506:0x5 DW_TAG_template_type_parameter
	.long	7709                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x250d:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	494                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2514:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2516:0x5 DW_TAG_template_type_parameter
	.long	7710                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x251d:0x5 DW_TAG_pointer_type
	.long	9506                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2522:0xc DW_TAG_structure_type
	.short	495                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2525:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2527:0x5 DW_TAG_template_type_parameter
	.long	7710                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x252e:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	496                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2535:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2537:0x5 DW_TAG_template_type_parameter
	.long	7715                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x253e:0x5 DW_TAG_pointer_type
	.long	9539                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2543:0xc DW_TAG_structure_type
	.short	497                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2546:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2548:0x5 DW_TAG_template_type_parameter
	.long	7715                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x254f:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	498                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2556:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2558:0x5 DW_TAG_template_type_parameter
	.long	372                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x255f:0x5 DW_TAG_pointer_type
	.long	9572                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2564:0xc DW_TAG_structure_type
	.short	499                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2567:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2569:0x5 DW_TAG_template_type_parameter
	.long	372                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2570:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	500                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2577:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2579:0x5 DW_TAG_template_type_parameter
	.long	7720                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2580:0x5 DW_TAG_pointer_type
	.long	9605                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2585:0xc DW_TAG_structure_type
	.short	501                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2588:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x258a:0x5 DW_TAG_template_type_parameter
	.long	7720                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2591:0x5 DW_TAG_pointer_type
	.long	9622                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2596:0x6 DW_TAG_structure_type
	.short	502                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	43                              # Abbrev [43] 0x2599:0x2 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x259c:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	503                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x25a3:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x25a5:0x5 DW_TAG_template_type_parameter
	.long	7703                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x25aa:0x5 DW_TAG_template_type_parameter
	.long	7703                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x25b1:0x5 DW_TAG_pointer_type
	.long	9654                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x25b6:0x11 DW_TAG_structure_type
	.short	504                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x25b9:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x25bb:0x5 DW_TAG_template_type_parameter
	.long	7703                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x25c0:0x5 DW_TAG_template_type_parameter
	.long	7703                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x25c7:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	505                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x25ce:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x25d0:0x5 DW_TAG_template_type_parameter
	.long	7725                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x25d7:0x5 DW_TAG_pointer_type
	.long	9692                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x25dc:0xc DW_TAG_structure_type
	.short	506                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x25df:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x25e1:0x5 DW_TAG_template_type_parameter
	.long	7725                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x25e8:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	507                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x25ef:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x25f1:0x5 DW_TAG_template_type_parameter
	.long	7746                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x25f8:0x5 DW_TAG_pointer_type
	.long	9725                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x25fd:0xc DW_TAG_structure_type
	.short	508                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2600:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2602:0x5 DW_TAG_template_type_parameter
	.long	7746                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2609:0x5 DW_TAG_pointer_type
	.long	3255                            # DW_AT_type
	.byte	83                              # Abbrev [83] 0x260e:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	510                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2615:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2617:0x5 DW_TAG_template_type_parameter
	.long	377                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x261e:0x5 DW_TAG_pointer_type
	.long	9763                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2623:0xc DW_TAG_structure_type
	.short	511                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2626:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2628:0x5 DW_TAG_template_type_parameter
	.long	377                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x262f:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	512                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2636:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2638:0x5 DW_TAG_template_type_parameter
	.long	7757                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x263f:0x5 DW_TAG_pointer_type
	.long	9796                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2644:0xc DW_TAG_structure_type
	.short	513                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2647:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2649:0x5 DW_TAG_template_type_parameter
	.long	7757                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2650:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	514                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2657:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2659:0x5 DW_TAG_template_type_parameter
	.long	7762                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2660:0x5 DW_TAG_pointer_type
	.long	9829                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2665:0xc DW_TAG_structure_type
	.short	515                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2668:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x266a:0x5 DW_TAG_template_type_parameter
	.long	7762                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2671:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	516                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2678:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x267a:0x5 DW_TAG_template_type_parameter
	.long	7533                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x267f:0x5 DW_TAG_template_type_parameter
	.long	7767                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2686:0x5 DW_TAG_pointer_type
	.long	9867                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x268b:0x11 DW_TAG_structure_type
	.short	517                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x268e:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2690:0x5 DW_TAG_template_type_parameter
	.long	7533                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2695:0x5 DW_TAG_template_type_parameter
	.long	7767                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x269c:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	518                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x26a3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x26a5:0x5 DW_TAG_template_type_parameter
	.long	7772                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x26ac:0x5 DW_TAG_pointer_type
	.long	9905                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x26b1:0xc DW_TAG_structure_type
	.short	519                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x26b4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x26b6:0x5 DW_TAG_template_type_parameter
	.long	7772                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x26bd:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	520                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x26c4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x26c6:0x5 DW_TAG_template_type_parameter
	.long	7786                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x26cd:0x5 DW_TAG_pointer_type
	.long	9938                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x26d2:0xc DW_TAG_structure_type
	.short	521                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x26d5:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x26d7:0x5 DW_TAG_template_type_parameter
	.long	7786                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x26de:0x1a DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	522                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x26e5:0x12 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x26e7:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	30                              # Abbrev [30] 0x26ec:0x5 DW_TAG_template_type_parameter
	.long	7187                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x26f1:0x5 DW_TAG_template_type_parameter
	.long	7791                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x26f8:0x5 DW_TAG_pointer_type
	.long	9981                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x26fd:0x16 DW_TAG_structure_type
	.short	523                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2700:0x12 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2702:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2707:0x5 DW_TAG_template_type_parameter
	.long	7187                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x270c:0x5 DW_TAG_template_type_parameter
	.long	7791                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2713:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	524                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x271a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x271c:0x5 DW_TAG_template_type_parameter
	.long	7796                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2723:0x5 DW_TAG_pointer_type
	.long	10024                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2728:0xc DW_TAG_structure_type
	.short	525                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x272b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x272d:0x5 DW_TAG_template_type_parameter
	.long	7796                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2734:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	526                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x273b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x273d:0x5 DW_TAG_template_type_parameter
	.long	7808                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2744:0x5 DW_TAG_pointer_type
	.long	10057                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2749:0xc DW_TAG_structure_type
	.short	527                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x274c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x274e:0x5 DW_TAG_template_type_parameter
	.long	7808                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2755:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	528                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x275c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x275e:0x5 DW_TAG_template_type_parameter
	.long	7818                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2765:0x5 DW_TAG_pointer_type
	.long	10090                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x276a:0xc DW_TAG_structure_type
	.short	529                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x276d:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x276f:0x5 DW_TAG_template_type_parameter
	.long	7818                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2776:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	530                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x277d:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x277f:0x5 DW_TAG_template_type_parameter
	.long	7824                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2786:0x5 DW_TAG_pointer_type
	.long	10123                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x278b:0xc DW_TAG_structure_type
	.short	531                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x278e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2790:0x5 DW_TAG_template_type_parameter
	.long	7824                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2797:0x5 DW_TAG_pointer_type
	.long	207                             # DW_AT_type
	.byte	83                              # Abbrev [83] 0x279c:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	532                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x27a3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x27a5:0x5 DW_TAG_template_type_parameter
	.long	7840                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x27ac:0x5 DW_TAG_pointer_type
	.long	10161                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x27b1:0xc DW_TAG_structure_type
	.short	533                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x27b4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x27b6:0x5 DW_TAG_template_type_parameter
	.long	7840                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x27bd:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	534                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x27c4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x27c6:0x5 DW_TAG_template_type_parameter
	.long	7866                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x27cd:0x5 DW_TAG_pointer_type
	.long	10194                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x27d2:0xc DW_TAG_structure_type
	.short	535                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x27d5:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x27d7:0x5 DW_TAG_template_type_parameter
	.long	7866                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x27de:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	536                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x27e5:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x27e7:0x5 DW_TAG_template_type_parameter
	.long	7892                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x27ee:0x5 DW_TAG_pointer_type
	.long	10227                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x27f3:0xc DW_TAG_structure_type
	.short	537                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x27f6:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x27f8:0x5 DW_TAG_template_type_parameter
	.long	7892                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x27ff:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	538                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2806:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2808:0x5 DW_TAG_template_type_parameter
	.long	7918                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x280f:0x5 DW_TAG_pointer_type
	.long	10260                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2814:0xc DW_TAG_structure_type
	.short	539                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2817:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2819:0x5 DW_TAG_template_type_parameter
	.long	7918                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2820:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	540                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2827:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2829:0x5 DW_TAG_template_type_parameter
	.long	7923                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2830:0x5 DW_TAG_pointer_type
	.long	10293                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2835:0xc DW_TAG_structure_type
	.short	541                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2838:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x283a:0x5 DW_TAG_template_type_parameter
	.long	7923                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2841:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	542                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2848:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x284a:0x5 DW_TAG_template_type_parameter
	.long	7945                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2851:0x5 DW_TAG_pointer_type
	.long	10326                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2856:0xc DW_TAG_structure_type
	.short	543                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2859:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x285b:0x5 DW_TAG_template_type_parameter
	.long	7945                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2862:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	544                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2869:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x286b:0x5 DW_TAG_template_type_parameter
	.long	7951                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2872:0x5 DW_TAG_pointer_type
	.long	10359                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2877:0xc DW_TAG_structure_type
	.short	545                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x287a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x287c:0x5 DW_TAG_template_type_parameter
	.long	7951                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2883:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	546                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x288a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x288c:0x5 DW_TAG_template_type_parameter
	.long	7957                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2893:0x5 DW_TAG_pointer_type
	.long	10392                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2898:0xc DW_TAG_structure_type
	.short	547                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x289b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x289d:0x5 DW_TAG_template_type_parameter
	.long	7957                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x28a4:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	548                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x28ab:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x28ad:0x5 DW_TAG_template_type_parameter
	.long	7967                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x28b4:0x5 DW_TAG_pointer_type
	.long	10425                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x28b9:0xc DW_TAG_structure_type
	.short	549                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x28bc:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x28be:0x5 DW_TAG_template_type_parameter
	.long	7967                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x28c5:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	550                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x28cc:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x28ce:0x5 DW_TAG_template_type_parameter
	.long	7984                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x28d5:0x5 DW_TAG_pointer_type
	.long	10458                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x28da:0xc DW_TAG_structure_type
	.short	551                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x28dd:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x28df:0x5 DW_TAG_template_type_parameter
	.long	7984                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x28e6:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	552                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x28ed:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x28ef:0x5 DW_TAG_template_type_parameter
	.long	7989                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x28f6:0x5 DW_TAG_pointer_type
	.long	10491                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x28fb:0xc DW_TAG_structure_type
	.short	553                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x28fe:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2900:0x5 DW_TAG_template_type_parameter
	.long	7989                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2907:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	554                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x290e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2910:0x5 DW_TAG_template_type_parameter
	.long	8020                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2917:0x5 DW_TAG_pointer_type
	.long	10524                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x291c:0xc DW_TAG_structure_type
	.short	555                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x291f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2921:0x5 DW_TAG_template_type_parameter
	.long	8020                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2928:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	556                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x292f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2931:0x5 DW_TAG_template_type_parameter
	.long	8043                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2938:0x5 DW_TAG_pointer_type
	.long	10557                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x293d:0xc DW_TAG_structure_type
	.short	557                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2940:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2942:0x5 DW_TAG_template_type_parameter
	.long	8043                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2949:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	558                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2950:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2952:0x5 DW_TAG_template_type_parameter
	.long	7710                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2959:0x5 DW_TAG_pointer_type
	.long	10590                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x295e:0xc DW_TAG_structure_type
	.short	559                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2961:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2963:0x5 DW_TAG_template_type_parameter
	.long	7710                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x296a:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	560                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2971:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2973:0x5 DW_TAG_template_type_parameter
	.long	8055                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x297a:0x5 DW_TAG_pointer_type
	.long	10623                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x297f:0xc DW_TAG_structure_type
	.short	561                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2982:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2984:0x5 DW_TAG_template_type_parameter
	.long	8055                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x298b:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	562                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2992:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2994:0x5 DW_TAG_template_type_parameter
	.long	8062                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x299b:0x5 DW_TAG_pointer_type
	.long	10656                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x29a0:0xc DW_TAG_structure_type
	.short	563                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x29a3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x29a5:0x5 DW_TAG_template_type_parameter
	.long	8062                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x29ac:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	564                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x29b3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x29b5:0x5 DW_TAG_template_type_parameter
	.long	8074                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x29bc:0x5 DW_TAG_pointer_type
	.long	10689                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x29c1:0xc DW_TAG_structure_type
	.short	565                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x29c4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x29c6:0x5 DW_TAG_template_type_parameter
	.long	8074                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x29cd:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	566                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x29d4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x29d6:0x5 DW_TAG_template_type_parameter
	.long	8111                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x29dd:0x5 DW_TAG_pointer_type
	.long	10722                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x29e2:0xc DW_TAG_structure_type
	.short	567                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x29e5:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x29e7:0x5 DW_TAG_template_type_parameter
	.long	8111                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x29ee:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	568                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x29f5:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x29f7:0x5 DW_TAG_template_type_parameter
	.long	8133                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x29fe:0x5 DW_TAG_pointer_type
	.long	10755                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2a03:0xc DW_TAG_structure_type
	.short	569                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2a06:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2a08:0x5 DW_TAG_template_type_parameter
	.long	8133                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2a0f:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	570                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2a16:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2a18:0x5 DW_TAG_template_type_parameter
	.long	8142                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2a1f:0x5 DW_TAG_pointer_type
	.long	10788                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2a24:0xc DW_TAG_structure_type
	.short	571                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2a27:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2a29:0x5 DW_TAG_template_type_parameter
	.long	8142                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2a30:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	572                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2a37:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2a39:0x5 DW_TAG_template_type_parameter
	.long	8144                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2a40:0x5 DW_TAG_pointer_type
	.long	10821                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2a45:0xc DW_TAG_structure_type
	.short	573                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2a48:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2a4a:0x5 DW_TAG_template_type_parameter
	.long	8144                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2a51:0x5 DW_TAG_pointer_type
	.long	6772                            # DW_AT_type
	.byte	83                              # Abbrev [83] 0x2a56:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	510                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2a5d:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2a5f:0x5 DW_TAG_template_type_parameter
	.long	6824                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2a66:0x5 DW_TAG_pointer_type
	.long	10859                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2a6b:0xc DW_TAG_structure_type
	.short	511                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2a6e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2a70:0x5 DW_TAG_template_type_parameter
	.long	6824                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	83                              # Abbrev [83] 0x2a77:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	574                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2a7e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2a80:0x5 DW_TAG_template_type_parameter
	.long	8149                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x2a87:0x5 DW_TAG_pointer_type
	.long	10892                           # DW_AT_type
	.byte	65                              # Abbrev [65] 0x2a8c:0xc DW_TAG_structure_type
	.short	575                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2a8f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2a91:0x5 DW_TAG_template_type_parameter
	.long	8149                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_rnglists,"",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0 # Length
.Ldebug_list_header_start0:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	1                               # Offset entry count
.Lrnglists_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_table_base0
.Ldebug_ranges0:
	.byte	1                               # DW_RLE_base_addressx
	.byte	1                               #   base address index
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    #   starting offset
	.uleb128 .Lfunc_end1-.Lfunc_begin0      #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Lfunc_begin29-.Lfunc_begin0   #   starting offset
	.uleb128 .Lfunc_end30-.Lfunc_begin0     #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Lfunc_begin46-.Lfunc_begin0   #   starting offset
	.uleb128 .Lfunc_end47-.Lfunc_begin0     #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Lfunc_begin53-.Lfunc_begin0   #   starting offset
	.uleb128 .Lfunc_end53-.Lfunc_begin0     #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Lfunc_begin61-.Lfunc_begin0   #   starting offset
	.uleb128 .Lfunc_end63-.Lfunc_begin0     #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Lfunc_begin95-.Lfunc_begin0   #   starting offset
	.uleb128 .Lfunc_end95-.Lfunc_begin0     #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Lfunc_begin109-.Lfunc_begin0  #   starting offset
	.uleb128 .Lfunc_end109-.Lfunc_begin0    #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Lfunc_begin126-.Lfunc_begin0  #   starting offset
	.uleb128 .Lfunc_end128-.Lfunc_begin0    #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Lfunc_begin138-.Lfunc_begin0  #   starting offset
	.uleb128 .Lfunc_end140-.Lfunc_begin0    #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Lfunc_begin142-.Lfunc_begin0  #   starting offset
	.uleb128 .Lfunc_end143-.Lfunc_begin0    #   ending offset
	.byte	3                               # DW_RLE_startx_length
	.byte	3                               #   start index
	.uleb128 .Lfunc_end2-.Lfunc_begin2      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	4                               #   start index
	.uleb128 .Lfunc_end3-.Lfunc_begin3      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	5                               #   start index
	.uleb128 .Lfunc_end4-.Lfunc_begin4      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	6                               #   start index
	.uleb128 .Lfunc_end5-.Lfunc_begin5      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	7                               #   start index
	.uleb128 .Lfunc_end6-.Lfunc_begin6      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	8                               #   start index
	.uleb128 .Lfunc_end7-.Lfunc_begin7      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	9                               #   start index
	.uleb128 .Lfunc_end8-.Lfunc_begin8      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	10                              #   start index
	.uleb128 .Lfunc_end9-.Lfunc_begin9      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	11                              #   start index
	.uleb128 .Lfunc_end10-.Lfunc_begin10    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	12                              #   start index
	.uleb128 .Lfunc_end11-.Lfunc_begin11    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	13                              #   start index
	.uleb128 .Lfunc_end12-.Lfunc_begin12    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	14                              #   start index
	.uleb128 .Lfunc_end13-.Lfunc_begin13    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	15                              #   start index
	.uleb128 .Lfunc_end14-.Lfunc_begin14    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	16                              #   start index
	.uleb128 .Lfunc_end15-.Lfunc_begin15    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	17                              #   start index
	.uleb128 .Lfunc_end16-.Lfunc_begin16    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	18                              #   start index
	.uleb128 .Lfunc_end17-.Lfunc_begin17    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	19                              #   start index
	.uleb128 .Lfunc_end18-.Lfunc_begin18    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	20                              #   start index
	.uleb128 .Lfunc_end19-.Lfunc_begin19    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	21                              #   start index
	.uleb128 .Lfunc_end20-.Lfunc_begin20    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	22                              #   start index
	.uleb128 .Lfunc_end21-.Lfunc_begin21    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	23                              #   start index
	.uleb128 .Lfunc_end22-.Lfunc_begin22    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	24                              #   start index
	.uleb128 .Lfunc_end23-.Lfunc_begin23    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	25                              #   start index
	.uleb128 .Lfunc_end24-.Lfunc_begin24    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	26                              #   start index
	.uleb128 .Lfunc_end25-.Lfunc_begin25    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	27                              #   start index
	.uleb128 .Lfunc_end26-.Lfunc_begin26    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	28                              #   start index
	.uleb128 .Lfunc_end27-.Lfunc_begin27    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	29                              #   start index
	.uleb128 .Lfunc_end28-.Lfunc_begin28    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	32                              #   start index
	.uleb128 .Lfunc_end31-.Lfunc_begin31    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	33                              #   start index
	.uleb128 .Lfunc_end32-.Lfunc_begin32    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	34                              #   start index
	.uleb128 .Lfunc_end33-.Lfunc_begin33    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	35                              #   start index
	.uleb128 .Lfunc_end34-.Lfunc_begin34    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	36                              #   start index
	.uleb128 .Lfunc_end35-.Lfunc_begin35    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	37                              #   start index
	.uleb128 .Lfunc_end36-.Lfunc_begin36    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	38                              #   start index
	.uleb128 .Lfunc_end37-.Lfunc_begin37    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	39                              #   start index
	.uleb128 .Lfunc_end38-.Lfunc_begin38    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	40                              #   start index
	.uleb128 .Lfunc_end39-.Lfunc_begin39    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	41                              #   start index
	.uleb128 .Lfunc_end40-.Lfunc_begin40    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	42                              #   start index
	.uleb128 .Lfunc_end41-.Lfunc_begin41    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	43                              #   start index
	.uleb128 .Lfunc_end42-.Lfunc_begin42    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	44                              #   start index
	.uleb128 .Lfunc_end43-.Lfunc_begin43    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	45                              #   start index
	.uleb128 .Lfunc_end44-.Lfunc_begin44    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	46                              #   start index
	.uleb128 .Lfunc_end45-.Lfunc_begin45    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	49                              #   start index
	.uleb128 .Lfunc_end48-.Lfunc_begin48    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	50                              #   start index
	.uleb128 .Lfunc_end49-.Lfunc_begin49    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	51                              #   start index
	.uleb128 .Lfunc_end50-.Lfunc_begin50    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	52                              #   start index
	.uleb128 .Lfunc_end51-.Lfunc_begin51    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	53                              #   start index
	.uleb128 .Lfunc_end52-.Lfunc_begin52    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	55                              #   start index
	.uleb128 .Lfunc_end54-.Lfunc_begin54    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	56                              #   start index
	.uleb128 .Lfunc_end55-.Lfunc_begin55    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	57                              #   start index
	.uleb128 .Lfunc_end56-.Lfunc_begin56    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	58                              #   start index
	.uleb128 .Lfunc_end57-.Lfunc_begin57    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	59                              #   start index
	.uleb128 .Lfunc_end58-.Lfunc_begin58    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	60                              #   start index
	.uleb128 .Lfunc_end59-.Lfunc_begin59    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	61                              #   start index
	.uleb128 .Lfunc_end60-.Lfunc_begin60    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	65                              #   start index
	.uleb128 .Lfunc_end64-.Lfunc_begin64    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	66                              #   start index
	.uleb128 .Lfunc_end65-.Lfunc_begin65    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	67                              #   start index
	.uleb128 .Lfunc_end66-.Lfunc_begin66    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	68                              #   start index
	.uleb128 .Lfunc_end67-.Lfunc_begin67    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	69                              #   start index
	.uleb128 .Lfunc_end68-.Lfunc_begin68    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	70                              #   start index
	.uleb128 .Lfunc_end69-.Lfunc_begin69    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	71                              #   start index
	.uleb128 .Lfunc_end70-.Lfunc_begin70    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	72                              #   start index
	.uleb128 .Lfunc_end71-.Lfunc_begin71    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	73                              #   start index
	.uleb128 .Lfunc_end72-.Lfunc_begin72    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	74                              #   start index
	.uleb128 .Lfunc_end73-.Lfunc_begin73    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	75                              #   start index
	.uleb128 .Lfunc_end74-.Lfunc_begin74    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	76                              #   start index
	.uleb128 .Lfunc_end75-.Lfunc_begin75    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	77                              #   start index
	.uleb128 .Lfunc_end76-.Lfunc_begin76    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	78                              #   start index
	.uleb128 .Lfunc_end77-.Lfunc_begin77    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	79                              #   start index
	.uleb128 .Lfunc_end78-.Lfunc_begin78    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	80                              #   start index
	.uleb128 .Lfunc_end79-.Lfunc_begin79    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	81                              #   start index
	.uleb128 .Lfunc_end80-.Lfunc_begin80    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	82                              #   start index
	.uleb128 .Lfunc_end81-.Lfunc_begin81    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	83                              #   start index
	.uleb128 .Lfunc_end82-.Lfunc_begin82    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	84                              #   start index
	.uleb128 .Lfunc_end83-.Lfunc_begin83    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	85                              #   start index
	.uleb128 .Lfunc_end84-.Lfunc_begin84    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	86                              #   start index
	.uleb128 .Lfunc_end85-.Lfunc_begin85    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	87                              #   start index
	.uleb128 .Lfunc_end86-.Lfunc_begin86    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	88                              #   start index
	.uleb128 .Lfunc_end87-.Lfunc_begin87    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	89                              #   start index
	.uleb128 .Lfunc_end88-.Lfunc_begin88    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	90                              #   start index
	.uleb128 .Lfunc_end89-.Lfunc_begin89    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	91                              #   start index
	.uleb128 .Lfunc_end90-.Lfunc_begin90    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	92                              #   start index
	.uleb128 .Lfunc_end91-.Lfunc_begin91    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	93                              #   start index
	.uleb128 .Lfunc_end92-.Lfunc_begin92    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	94                              #   start index
	.uleb128 .Lfunc_end93-.Lfunc_begin93    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	95                              #   start index
	.uleb128 .Lfunc_end94-.Lfunc_begin94    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	97                              #   start index
	.uleb128 .Lfunc_end96-.Lfunc_begin96    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	98                              #   start index
	.uleb128 .Lfunc_end97-.Lfunc_begin97    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	99                              #   start index
	.uleb128 .Lfunc_end98-.Lfunc_begin98    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	100                             #   start index
	.uleb128 .Lfunc_end99-.Lfunc_begin99    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	101                             #   start index
	.uleb128 .Lfunc_end100-.Lfunc_begin100  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	102                             #   start index
	.uleb128 .Lfunc_end101-.Lfunc_begin101  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	103                             #   start index
	.uleb128 .Lfunc_end102-.Lfunc_begin102  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	104                             #   start index
	.uleb128 .Lfunc_end103-.Lfunc_begin103  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	105                             #   start index
	.uleb128 .Lfunc_end104-.Lfunc_begin104  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	106                             #   start index
	.uleb128 .Lfunc_end105-.Lfunc_begin105  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	107                             #   start index
	.uleb128 .Lfunc_end106-.Lfunc_begin106  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	108                             #   start index
	.uleb128 .Lfunc_end107-.Lfunc_begin107  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	109                             #   start index
	.uleb128 .Lfunc_end108-.Lfunc_begin108  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	111                             #   start index
	.uleb128 .Lfunc_end110-.Lfunc_begin110  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	112                             #   start index
	.uleb128 .Lfunc_end111-.Lfunc_begin111  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	113                             #   start index
	.uleb128 .Lfunc_end112-.Lfunc_begin112  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	114                             #   start index
	.uleb128 .Lfunc_end113-.Lfunc_begin113  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	115                             #   start index
	.uleb128 .Lfunc_end114-.Lfunc_begin114  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	116                             #   start index
	.uleb128 .Lfunc_end115-.Lfunc_begin115  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	117                             #   start index
	.uleb128 .Lfunc_end116-.Lfunc_begin116  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	118                             #   start index
	.uleb128 .Lfunc_end117-.Lfunc_begin117  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	119                             #   start index
	.uleb128 .Lfunc_end118-.Lfunc_begin118  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	120                             #   start index
	.uleb128 .Lfunc_end119-.Lfunc_begin119  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	121                             #   start index
	.uleb128 .Lfunc_end120-.Lfunc_begin120  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	122                             #   start index
	.uleb128 .Lfunc_end121-.Lfunc_begin121  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	123                             #   start index
	.uleb128 .Lfunc_end122-.Lfunc_begin122  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	124                             #   start index
	.uleb128 .Lfunc_end123-.Lfunc_begin123  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	125                             #   start index
	.uleb128 .Lfunc_end124-.Lfunc_begin124  #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	126                             #   start index
	.uleb128 .Lfunc_end125-.Lfunc_begin125  #   length
	.byte	3                               # DW_RLE_startx_length
	.ascii	"\202\001"                      #   start index
	.uleb128 .Lfunc_end129-.Lfunc_begin129  #   length
	.byte	3                               # DW_RLE_startx_length
	.ascii	"\203\001"                      #   start index
	.uleb128 .Lfunc_end130-.Lfunc_begin130  #   length
	.byte	3                               # DW_RLE_startx_length
	.ascii	"\204\001"                      #   start index
	.uleb128 .Lfunc_end131-.Lfunc_begin131  #   length
	.byte	3                               # DW_RLE_startx_length
	.ascii	"\205\001"                      #   start index
	.uleb128 .Lfunc_end132-.Lfunc_begin132  #   length
	.byte	3                               # DW_RLE_startx_length
	.ascii	"\206\001"                      #   start index
	.uleb128 .Lfunc_end133-.Lfunc_begin133  #   length
	.byte	3                               # DW_RLE_startx_length
	.ascii	"\207\001"                      #   start index
	.uleb128 .Lfunc_end134-.Lfunc_begin134  #   length
	.byte	3                               # DW_RLE_startx_length
	.ascii	"\210\001"                      #   start index
	.uleb128 .Lfunc_end135-.Lfunc_begin135  #   length
	.byte	3                               # DW_RLE_startx_length
	.ascii	"\211\001"                      #   start index
	.uleb128 .Lfunc_end136-.Lfunc_begin136  #   length
	.byte	3                               # DW_RLE_startx_length
	.ascii	"\212\001"                      #   start index
	.uleb128 .Lfunc_end137-.Lfunc_begin137  #   length
	.byte	3                               # DW_RLE_startx_length
	.ascii	"\216\001"                      #   start index
	.uleb128 .Lfunc_end141-.Lfunc_begin141  #   length
	.byte	3                               # DW_RLE_startx_length
	.ascii	"\221\001"                      #   start index
	.uleb128 .Lfunc_end144-.Lfunc_begin144  #   length
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	2324                            # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.byte	0                               # string offset=0
.Linfo_string1:
	.asciz	"/proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp" # string offset=1
.Linfo_string2:
	.asciz	"/proc/self/cwd"                # string offset=110
.Linfo_string3:
	.asciz	"i"                             # string offset=125
.Linfo_string4:
	.asciz	"int"                           # string offset=127
.Linfo_string5:
	.asciz	"unsigned int"                  # string offset=131
.Linfo_string6:
	.asciz	"LocalEnum1"                    # string offset=144
.Linfo_string7:
	.asciz	"LocalEnum"                     # string offset=155
.Linfo_string8:
	.asciz	"ns"                            # string offset=165
.Linfo_string9:
	.asciz	"Enumerator1"                   # string offset=168
.Linfo_string10:
	.asciz	"Enumerator2"                   # string offset=180
.Linfo_string11:
	.asciz	"Enumerator3"                   # string offset=192
.Linfo_string12:
	.asciz	"Enumeration"                   # string offset=204
.Linfo_string13:
	.asciz	"EnumerationClass"              # string offset=216
.Linfo_string14:
	.asciz	"unsigned char"                 # string offset=233
.Linfo_string15:
	.asciz	"kNeg"                          # string offset=247
.Linfo_string16:
	.asciz	"EnumerationSmall"              # string offset=252
.Linfo_string17:
	.asciz	"AnonEnum1"                     # string offset=269
.Linfo_string18:
	.asciz	"AnonEnum2"                     # string offset=279
.Linfo_string19:
	.asciz	"AnonEnum3"                     # string offset=289
.Linfo_string20:
	.asciz	"T"                             # string offset=299
.Linfo_string21:
	.asciz	"bool"                          # string offset=301
.Linfo_string22:
	.asciz	"b"                             # string offset=306
.Linfo_string23:
	.asciz	"_STN|t3|<int, false>"          # string offset=308
.Linfo_string24:
	.asciz	"t10"                           # string offset=329
.Linfo_string25:
	.asciz	"t6"                            # string offset=333
.Linfo_string26:
	.asciz	"_ZN2t6lsIiEEvi"                # string offset=336
.Linfo_string27:
	.asciz	"operator<<<int>"               # string offset=351
.Linfo_string28:
	.asciz	"_ZN2t6ltIiEEvi"                # string offset=367
.Linfo_string29:
	.asciz	"operator<<int>"                # string offset=382
.Linfo_string30:
	.asciz	"_ZN2t6leIiEEvi"                # string offset=397
.Linfo_string31:
	.asciz	"operator<=<int>"               # string offset=412
.Linfo_string32:
	.asciz	"_ZN2t6cvP2t1IJfEEIiEEv"        # string offset=428
.Linfo_string33:
	.asciz	"operator t1<float> *<int>"     # string offset=451
.Linfo_string34:
	.asciz	"Ts"                            # string offset=477
.Linfo_string35:
	.asciz	"float"                         # string offset=480
.Linfo_string36:
	.asciz	"_STN|t1|<float>"               # string offset=486
.Linfo_string37:
	.asciz	"_ZN2t6miIiEEvi"                # string offset=502
.Linfo_string38:
	.asciz	"operator-<int>"                # string offset=517
.Linfo_string39:
	.asciz	"_ZN2t6mlIiEEvi"                # string offset=532
.Linfo_string40:
	.asciz	"operator*<int>"                # string offset=547
.Linfo_string41:
	.asciz	"_ZN2t6dvIiEEvi"                # string offset=562
.Linfo_string42:
	.asciz	"operator/<int>"                # string offset=577
.Linfo_string43:
	.asciz	"_ZN2t6rmIiEEvi"                # string offset=592
.Linfo_string44:
	.asciz	"operator%<int>"                # string offset=607
.Linfo_string45:
	.asciz	"_ZN2t6eoIiEEvi"                # string offset=622
.Linfo_string46:
	.asciz	"operator^<int>"                # string offset=637
.Linfo_string47:
	.asciz	"_ZN2t6anIiEEvi"                # string offset=652
.Linfo_string48:
	.asciz	"operator&<int>"                # string offset=667
.Linfo_string49:
	.asciz	"_ZN2t6orIiEEvi"                # string offset=682
.Linfo_string50:
	.asciz	"operator|<int>"                # string offset=697
.Linfo_string51:
	.asciz	"_ZN2t6coIiEEvv"                # string offset=712
.Linfo_string52:
	.asciz	"operator~<int>"                # string offset=727
.Linfo_string53:
	.asciz	"_ZN2t6ntIiEEvv"                # string offset=742
.Linfo_string54:
	.asciz	"operator!<int>"                # string offset=757
.Linfo_string55:
	.asciz	"_ZN2t6aSIiEEvi"                # string offset=772
.Linfo_string56:
	.asciz	"operator=<int>"                # string offset=787
.Linfo_string57:
	.asciz	"_ZN2t6gtIiEEvi"                # string offset=802
.Linfo_string58:
	.asciz	"operator><int>"                # string offset=817
.Linfo_string59:
	.asciz	"_ZN2t6cmIiEEvi"                # string offset=832
.Linfo_string60:
	.asciz	"operator,<int>"                # string offset=847
.Linfo_string61:
	.asciz	"_ZN2t6clIiEEvv"                # string offset=862
.Linfo_string62:
	.asciz	"operator()<int>"               # string offset=877
.Linfo_string63:
	.asciz	"_ZN2t6ixIiEEvi"                # string offset=893
.Linfo_string64:
	.asciz	"operator[]<int>"               # string offset=908
.Linfo_string65:
	.asciz	"_ZN2t6ssIiEEvi"                # string offset=924
.Linfo_string66:
	.asciz	"operator<=><int>"              # string offset=939
.Linfo_string67:
	.asciz	"_ZN2t6nwIiEEPvmT_"             # string offset=956
.Linfo_string68:
	.asciz	"operator new<int>"             # string offset=974
.Linfo_string69:
	.asciz	"std"                           # string offset=992
.Linfo_string70:
	.asciz	"unsigned long"                 # string offset=996
.Linfo_string71:
	.asciz	"size_t"                        # string offset=1010
.Linfo_string72:
	.asciz	"_ZN2t6naIiEEPvmT_"             # string offset=1017
.Linfo_string73:
	.asciz	"operator new[]<int>"           # string offset=1035
.Linfo_string74:
	.asciz	"_ZN2t6dlIiEEvPvT_"             # string offset=1055
.Linfo_string75:
	.asciz	"operator delete<int>"          # string offset=1073
.Linfo_string76:
	.asciz	"_ZN2t6daIiEEvPvT_"             # string offset=1094
.Linfo_string77:
	.asciz	"operator delete[]<int>"        # string offset=1112
.Linfo_string78:
	.asciz	"_ZN2t6awIiEEiv"                # string offset=1135
.Linfo_string79:
	.asciz	"operator co_await<int>"        # string offset=1150
.Linfo_string80:
	.asciz	"_ZN3t10C4IvEEv"                # string offset=1173
.Linfo_string81:
	.asciz	"_STN|t10|<void>"               # string offset=1188
.Linfo_string82:
	.asciz	"_ZN2t83memEv"                  # string offset=1204
.Linfo_string83:
	.asciz	"mem"                           # string offset=1217
.Linfo_string84:
	.asciz	"t8"                            # string offset=1221
.Linfo_string85:
	.asciz	"complex_type_units"            # string offset=1224
.Linfo_string86:
	.asciz	"ptr_to_member_test"            # string offset=1243
.Linfo_string87:
	.asciz	"max_align_t"                   # string offset=1262
.Linfo_string88:
	.asciz	"signed char"                   # string offset=1274
.Linfo_string89:
	.asciz	"__int8_t"                      # string offset=1286
.Linfo_string90:
	.asciz	"int8_t"                        # string offset=1295
.Linfo_string91:
	.asciz	"short"                         # string offset=1302
.Linfo_string92:
	.asciz	"__int16_t"                     # string offset=1308
.Linfo_string93:
	.asciz	"int16_t"                       # string offset=1318
.Linfo_string94:
	.asciz	"__int32_t"                     # string offset=1326
.Linfo_string95:
	.asciz	"int32_t"                       # string offset=1336
.Linfo_string96:
	.asciz	"long"                          # string offset=1344
.Linfo_string97:
	.asciz	"__int64_t"                     # string offset=1349
.Linfo_string98:
	.asciz	"int64_t"                       # string offset=1359
.Linfo_string99:
	.asciz	"int_fast8_t"                   # string offset=1367
.Linfo_string100:
	.asciz	"int_fast16_t"                  # string offset=1379
.Linfo_string101:
	.asciz	"int_fast32_t"                  # string offset=1392
.Linfo_string102:
	.asciz	"int_fast64_t"                  # string offset=1405
.Linfo_string103:
	.asciz	"__int_least8_t"                # string offset=1418
.Linfo_string104:
	.asciz	"int_least8_t"                  # string offset=1433
.Linfo_string105:
	.asciz	"__int_least16_t"               # string offset=1446
.Linfo_string106:
	.asciz	"int_least16_t"                 # string offset=1462
.Linfo_string107:
	.asciz	"__int_least32_t"               # string offset=1476
.Linfo_string108:
	.asciz	"int_least32_t"                 # string offset=1492
.Linfo_string109:
	.asciz	"__int_least64_t"               # string offset=1506
.Linfo_string110:
	.asciz	"int_least64_t"                 # string offset=1522
.Linfo_string111:
	.asciz	"__intmax_t"                    # string offset=1536
.Linfo_string112:
	.asciz	"intmax_t"                      # string offset=1547
.Linfo_string113:
	.asciz	"intptr_t"                      # string offset=1556
.Linfo_string114:
	.asciz	"__uint8_t"                     # string offset=1565
.Linfo_string115:
	.asciz	"uint8_t"                       # string offset=1575
.Linfo_string116:
	.asciz	"unsigned short"                # string offset=1583
.Linfo_string117:
	.asciz	"__uint16_t"                    # string offset=1598
.Linfo_string118:
	.asciz	"uint16_t"                      # string offset=1609
.Linfo_string119:
	.asciz	"__uint32_t"                    # string offset=1618
.Linfo_string120:
	.asciz	"uint32_t"                      # string offset=1629
.Linfo_string121:
	.asciz	"__uint64_t"                    # string offset=1638
.Linfo_string122:
	.asciz	"uint64_t"                      # string offset=1649
.Linfo_string123:
	.asciz	"uint_fast8_t"                  # string offset=1658
.Linfo_string124:
	.asciz	"uint_fast16_t"                 # string offset=1671
.Linfo_string125:
	.asciz	"uint_fast32_t"                 # string offset=1685
.Linfo_string126:
	.asciz	"uint_fast64_t"                 # string offset=1699
.Linfo_string127:
	.asciz	"__uint_least8_t"               # string offset=1713
.Linfo_string128:
	.asciz	"uint_least8_t"                 # string offset=1729
.Linfo_string129:
	.asciz	"__uint_least16_t"              # string offset=1743
.Linfo_string130:
	.asciz	"uint_least16_t"                # string offset=1760
.Linfo_string131:
	.asciz	"__uint_least32_t"              # string offset=1775
.Linfo_string132:
	.asciz	"uint_least32_t"                # string offset=1792
.Linfo_string133:
	.asciz	"__uint_least64_t"              # string offset=1807
.Linfo_string134:
	.asciz	"uint_least64_t"                # string offset=1824
.Linfo_string135:
	.asciz	"__uintmax_t"                   # string offset=1839
.Linfo_string136:
	.asciz	"uintmax_t"                     # string offset=1851
.Linfo_string137:
	.asciz	"uintptr_t"                     # string offset=1861
.Linfo_string138:
	.asciz	"_Zli5_suffy"                   # string offset=1871
.Linfo_string139:
	.asciz	"operator\"\"_suff"             # string offset=1883
.Linfo_string140:
	.asciz	"main"                          # string offset=1899
.Linfo_string141:
	.asciz	"_Z2f1IJiEEvv"                  # string offset=1904
.Linfo_string142:
	.asciz	"_STN|f1|<int>"                 # string offset=1917
.Linfo_string143:
	.asciz	"_Z2f1IJfEEvv"                  # string offset=1931
.Linfo_string144:
	.asciz	"_STN|f1|<float>"               # string offset=1944
.Linfo_string145:
	.asciz	"_Z2f1IJbEEvv"                  # string offset=1960
.Linfo_string146:
	.asciz	"_STN|f1|<bool>"                # string offset=1973
.Linfo_string147:
	.asciz	"double"                        # string offset=1988
.Linfo_string148:
	.asciz	"_Z2f1IJdEEvv"                  # string offset=1995
.Linfo_string149:
	.asciz	"_STN|f1|<double>"              # string offset=2008
.Linfo_string150:
	.asciz	"_Z2f1IJlEEvv"                  # string offset=2025
.Linfo_string151:
	.asciz	"_STN|f1|<long>"                # string offset=2038
.Linfo_string152:
	.asciz	"_Z2f1IJsEEvv"                  # string offset=2053
.Linfo_string153:
	.asciz	"_STN|f1|<short>"               # string offset=2066
.Linfo_string154:
	.asciz	"_Z2f1IJjEEvv"                  # string offset=2082
.Linfo_string155:
	.asciz	"_STN|f1|<unsigned int>"        # string offset=2095
.Linfo_string156:
	.asciz	"unsigned long long"            # string offset=2118
.Linfo_string157:
	.asciz	"_Z2f1IJyEEvv"                  # string offset=2137
.Linfo_string158:
	.asciz	"_STN|f1|<unsigned long long>"  # string offset=2150
.Linfo_string159:
	.asciz	"long long"                     # string offset=2179
.Linfo_string160:
	.asciz	"_Z2f1IJxEEvv"                  # string offset=2189
.Linfo_string161:
	.asciz	"_STN|f1|<long long>"           # string offset=2202
.Linfo_string162:
	.asciz	"udt"                           # string offset=2222
.Linfo_string163:
	.asciz	"_Z2f1IJ3udtEEvv"               # string offset=2226
.Linfo_string164:
	.asciz	"_STN|f1|<udt>"                 # string offset=2242
.Linfo_string165:
	.asciz	"_Z2f1IJN2ns3udtEEEvv"          # string offset=2256
.Linfo_string166:
	.asciz	"_STN|f1|<ns::udt>"             # string offset=2277
.Linfo_string167:
	.asciz	"_Z2f1IJPN2ns3udtEEEvv"         # string offset=2295
.Linfo_string168:
	.asciz	"_STN|f1|<ns::udt *>"           # string offset=2317
.Linfo_string169:
	.asciz	"inner"                         # string offset=2337
.Linfo_string170:
	.asciz	"_Z2f1IJN2ns5inner3udtEEEvv"    # string offset=2343
.Linfo_string171:
	.asciz	"_STN|f1|<ns::inner::udt>"      # string offset=2370
.Linfo_string172:
	.asciz	"_STN|t1|<int>"                 # string offset=2395
.Linfo_string173:
	.asciz	"_Z2f1IJ2t1IJiEEEEvv"           # string offset=2409
.Linfo_string174:
	.asciz	"_STN|f1|<t1<int> >"            # string offset=2429
.Linfo_string175:
	.asciz	"_Z2f1IJifEEvv"                 # string offset=2448
.Linfo_string176:
	.asciz	"_STN|f1|<int, float>"          # string offset=2462
.Linfo_string177:
	.asciz	"_Z2f1IJPiEEvv"                 # string offset=2483
.Linfo_string178:
	.asciz	"_STN|f1|<int *>"               # string offset=2497
.Linfo_string179:
	.asciz	"_Z2f1IJRiEEvv"                 # string offset=2513
.Linfo_string180:
	.asciz	"_STN|f1|<int &>"               # string offset=2527
.Linfo_string181:
	.asciz	"_Z2f1IJOiEEvv"                 # string offset=2543
.Linfo_string182:
	.asciz	"_STN|f1|<int &&>"              # string offset=2557
.Linfo_string183:
	.asciz	"_Z2f1IJKiEEvv"                 # string offset=2574
.Linfo_string184:
	.asciz	"_STN|f1|<const int>"           # string offset=2588
.Linfo_string185:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=2608
.Linfo_string186:
	.asciz	"_Z2f1IJA3_iEEvv"               # string offset=2628
.Linfo_string187:
	.asciz	"_STN|f1|<int[3]>"              # string offset=2644
.Linfo_string188:
	.asciz	"_Z2f1IJvEEvv"                  # string offset=2661
.Linfo_string189:
	.asciz	"_STN|f1|<void>"                # string offset=2674
.Linfo_string190:
	.asciz	"outer_class"                   # string offset=2689
.Linfo_string191:
	.asciz	"inner_class"                   # string offset=2701
.Linfo_string192:
	.asciz	"_Z2f1IJN11outer_class11inner_classEEEvv" # string offset=2713
.Linfo_string193:
	.asciz	"_STN|f1|<outer_class::inner_class>" # string offset=2753
.Linfo_string194:
	.asciz	"_Z2f1IJmEEvv"                  # string offset=2788
.Linfo_string195:
	.asciz	"_STN|f1|<unsigned long>"       # string offset=2801
.Linfo_string196:
	.asciz	"_Z2f2ILb1ELi3EEvv"             # string offset=2825
.Linfo_string197:
	.asciz	"_STN|f2|<true, 3>"             # string offset=2843
.Linfo_string198:
	.asciz	"A"                             # string offset=2861
.Linfo_string199:
	.asciz	"_Z2f3IN2ns11EnumerationETpTnT_JLS1_1ELS1_2EEEvv" # string offset=2863
.Linfo_string200:
	.asciz	"_STN|f3|<ns::Enumeration, (ns::Enumeration)1, (ns::Enumeration)2>" # string offset=2911
.Linfo_string201:
	.asciz	"_Z2f3IN2ns16EnumerationClassETpTnT_JLS1_1ELS1_2EEEvv" # string offset=2977
.Linfo_string202:
	.asciz	"_STN|f3|<ns::EnumerationClass, (ns::EnumerationClass)1, (ns::EnumerationClass)2>" # string offset=3030
.Linfo_string203:
	.asciz	"_Z2f3IN2ns16EnumerationSmallETpTnT_JLS1_255EEEvv" # string offset=3111
.Linfo_string204:
	.asciz	"_STN|f3|<ns::EnumerationSmall, (ns::EnumerationSmall)255>" # string offset=3160
.Linfo_string205:
	.asciz	"_Z2f3IN2ns3$_0ETpTnT_JLS1_1ELS1_2EEEvv" # string offset=3218
.Linfo_string206:
	.asciz	"f3<ns::(unnamed enum at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:15:1), (ns::(unnamed enum at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:15:1))1, (ns::(unnamed enum at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:15:1))2>" # string offset=3257
.Linfo_string207:
	.asciz	"_Z2f3IN12_GLOBAL__N_19LocalEnumETpTnT_JLS1_0EEEvv" # string offset=3677
.Linfo_string208:
	.asciz	"f3<(anonymous namespace)::LocalEnum, ((anonymous namespace)::LocalEnum)0>" # string offset=3727
.Linfo_string209:
	.asciz	"_Z2f3IPiTpTnT_JXadL_Z1iEEEEvv" # string offset=3801
.Linfo_string210:
	.asciz	"f3<int *, &i>"                 # string offset=3831
.Linfo_string211:
	.asciz	"_Z2f3IPiTpTnT_JLS0_0EEEvv"     # string offset=3845
.Linfo_string212:
	.asciz	"f3<int *, nullptr>"            # string offset=3871
.Linfo_string213:
	.asciz	"_Z2f3ImTpTnT_JLm1EEEvv"        # string offset=3890
.Linfo_string214:
	.asciz	"_STN|f3|<unsigned long, 1UL>"  # string offset=3913
.Linfo_string215:
	.asciz	"_Z2f3IyTpTnT_JLy1EEEvv"        # string offset=3942
.Linfo_string216:
	.asciz	"_STN|f3|<unsigned long long, 1ULL>" # string offset=3965
.Linfo_string217:
	.asciz	"_Z2f3IlTpTnT_JLl1EEEvv"        # string offset=4000
.Linfo_string218:
	.asciz	"_STN|f3|<long, 1L>"            # string offset=4023
.Linfo_string219:
	.asciz	"_Z2f3IjTpTnT_JLj1EEEvv"        # string offset=4042
.Linfo_string220:
	.asciz	"_STN|f3|<unsigned int, 1U>"    # string offset=4065
.Linfo_string221:
	.asciz	"_Z2f3IsTpTnT_JLs1EEEvv"        # string offset=4092
.Linfo_string222:
	.asciz	"_STN|f3|<short, (short)1>"     # string offset=4115
.Linfo_string223:
	.asciz	"_Z2f3IhTpTnT_JLh0EEEvv"        # string offset=4141
.Linfo_string224:
	.asciz	"_STN|f3|<unsigned char, (unsigned char)'\\x00'>" # string offset=4164
.Linfo_string225:
	.asciz	"_Z2f3IaTpTnT_JLa0EEEvv"        # string offset=4211
.Linfo_string226:
	.asciz	"_STN|f3|<signed char, (signed char)'\\x00'>" # string offset=4234
.Linfo_string227:
	.asciz	"_Z2f3ItTpTnT_JLt1ELt2EEEvv"    # string offset=4277
.Linfo_string228:
	.asciz	"_STN|f3|<unsigned short, (unsigned short)1, (unsigned short)2>" # string offset=4304
.Linfo_string229:
	.asciz	"char"                          # string offset=4367
.Linfo_string230:
	.asciz	"_Z2f3IcTpTnT_JLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv" # string offset=4372
.Linfo_string231:
	.asciz	"_STN|f3|<char, '\\x00', '\\x01', '\\x06', '\\a', '\\r', '\\x0e', '\\x1f', ' ', '!', '\\x7f', '\\x80'>" # string offset=4445
.Linfo_string232:
	.asciz	"__int128"                      # string offset=4538
.Linfo_string233:
	.asciz	"_Z2f3InTpTnT_JLn18446744073709551614EEEvv" # string offset=4547
.Linfo_string234:
	.asciz	"f3<__int128, (__int128)18446744073709551614>" # string offset=4589
.Linfo_string235:
	.asciz	"_Z2f4IjLj3EEvv"                # string offset=4634
.Linfo_string236:
	.asciz	"_STN|f4|<unsigned int, 3U>"    # string offset=4649
.Linfo_string237:
	.asciz	"_Z2f1IJ2t3IiLb0EEEEvv"         # string offset=4676
.Linfo_string238:
	.asciz	"_STN|f1|<t3<int, false> >"     # string offset=4698
.Linfo_string239:
	.asciz	"_STN|t3|<t3<int, false>, false>" # string offset=4724
.Linfo_string240:
	.asciz	"_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv" # string offset=4756
.Linfo_string241:
	.asciz	"_STN|f1|<t3<t3<int, false>, false> >" # string offset=4787
.Linfo_string242:
	.asciz	"_Z2f1IJZ4mainE3$_0EEvv"        # string offset=4824
.Linfo_string243:
	.asciz	"f1<(lambda at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:103:12)>" # string offset=4847
.Linfo_string244:
	.asciz	"t3<(lambda at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:103:12), false>" # string offset=4979
.Linfo_string245:
	.asciz	"t3<t3<(lambda at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:103:12), false>, false>" # string offset=5118
.Linfo_string246:
	.asciz	"_Z2f1IJ2t3IS0_IZ4mainE3$_0Lb0EELb0EEEEvv" # string offset=5268
.Linfo_string247:
	.asciz	"f1<t3<t3<(lambda at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:103:12), false>, false> >" # string offset=5309
.Linfo_string248:
	.asciz	"_Z2f1IJFifEEEvv"               # string offset=5464
.Linfo_string249:
	.asciz	"_STN|f1|<int (float)>"         # string offset=5480
.Linfo_string250:
	.asciz	"_Z2f1IJFvzEEEvv"               # string offset=5502
.Linfo_string251:
	.asciz	"_STN|f1|<void (...)>"          # string offset=5518
.Linfo_string252:
	.asciz	"_Z2f1IJFvizEEEvv"              # string offset=5539
.Linfo_string253:
	.asciz	"_STN|f1|<void (int, ...)>"     # string offset=5556
.Linfo_string254:
	.asciz	"_Z2f1IJRKiEEvv"                # string offset=5582
.Linfo_string255:
	.asciz	"_STN|f1|<const int &>"         # string offset=5597
.Linfo_string256:
	.asciz	"_Z2f1IJRPKiEEvv"               # string offset=5619
.Linfo_string257:
	.asciz	"_STN|f1|<const int *&>"        # string offset=5635
.Linfo_string258:
	.asciz	"t5"                            # string offset=5658
.Linfo_string259:
	.asciz	"_Z2f1IJN12_GLOBAL__N_12t5EEEvv" # string offset=5661
.Linfo_string260:
	.asciz	"_STN|f1|<(anonymous namespace)::t5>" # string offset=5692
.Linfo_string261:
	.asciz	"decltype(nullptr)"             # string offset=5728
.Linfo_string262:
	.asciz	"_Z2f1IJDnEEvv"                 # string offset=5746
.Linfo_string263:
	.asciz	"_STN|f1|<std::nullptr_t>"      # string offset=5760
.Linfo_string264:
	.asciz	"_Z2f1IJPlS0_EEvv"              # string offset=5785
.Linfo_string265:
	.asciz	"_STN|f1|<long *, long *>"      # string offset=5802
.Linfo_string266:
	.asciz	"_Z2f1IJPlP3udtEEvv"            # string offset=5827
.Linfo_string267:
	.asciz	"_STN|f1|<long *, udt *>"       # string offset=5846
.Linfo_string268:
	.asciz	"_Z2f1IJKPvEEvv"                # string offset=5870
.Linfo_string269:
	.asciz	"_STN|f1|<void *const>"         # string offset=5885
.Linfo_string270:
	.asciz	"_Z2f1IJPKPKvEEvv"              # string offset=5907
.Linfo_string271:
	.asciz	"_STN|f1|<const void *const *>" # string offset=5924
.Linfo_string272:
	.asciz	"_Z2f1IJFvvEEEvv"               # string offset=5954
.Linfo_string273:
	.asciz	"_STN|f1|<void ()>"             # string offset=5970
.Linfo_string274:
	.asciz	"_Z2f1IJPFvvEEEvv"              # string offset=5988
.Linfo_string275:
	.asciz	"_STN|f1|<void (*)()>"          # string offset=6005
.Linfo_string276:
	.asciz	"_Z2f1IJPZ4mainE3$_0EEvv"       # string offset=6026
.Linfo_string277:
	.asciz	"f1<(lambda at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:103:12) *>" # string offset=6050
.Linfo_string278:
	.asciz	"_Z2f1IJZ4mainE3$_1EEvv"        # string offset=6184
.Linfo_string279:
	.asciz	"f1<(unnamed struct at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:101:3)>" # string offset=6207
.Linfo_string280:
	.asciz	"_Z2f1IJPZ4mainE3$_1EEvv"       # string offset=6346
.Linfo_string281:
	.asciz	"f1<(unnamed struct at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:101:3) *>" # string offset=6370
.Linfo_string282:
	.asciz	"T1"                            # string offset=6511
.Linfo_string283:
	.asciz	"T2"                            # string offset=6514
.Linfo_string284:
	.asciz	"_Z2f5IJ2t1IJiEEEiEvv"          # string offset=6517
.Linfo_string285:
	.asciz	"_STN|f5|<t1<int>, int>"        # string offset=6538
.Linfo_string286:
	.asciz	"_Z2f5IJEiEvv"                  # string offset=6561
.Linfo_string287:
	.asciz	"_STN|f5|<int>"                 # string offset=6574
.Linfo_string288:
	.asciz	"_Z2f6I2t1IJiEEJEEvv"           # string offset=6588
.Linfo_string289:
	.asciz	"_STN|f6|<t1<int> >"            # string offset=6608
.Linfo_string290:
	.asciz	"_Z2f1IJEEvv"                   # string offset=6627
.Linfo_string291:
	.asciz	"_STN|f1|<>"                    # string offset=6639
.Linfo_string292:
	.asciz	"_Z2f1IJPKvS1_EEvv"             # string offset=6650
.Linfo_string293:
	.asciz	"_STN|f1|<const void *, const void *>" # string offset=6668
.Linfo_string294:
	.asciz	"_STN|t1|<int *>"               # string offset=6705
.Linfo_string295:
	.asciz	"_Z2f1IJP2t1IJPiEEEEvv"         # string offset=6721
.Linfo_string296:
	.asciz	"_STN|f1|<t1<int *> *>"         # string offset=6743
.Linfo_string297:
	.asciz	"_Z2f1IJA_PiEEvv"               # string offset=6765
.Linfo_string298:
	.asciz	"_STN|f1|<int *[]>"             # string offset=6781
.Linfo_string299:
	.asciz	"t7"                            # string offset=6799
.Linfo_string300:
	.asciz	"_Z2f1IJZ4mainE2t7EEvv"         # string offset=6802
.Linfo_string301:
	.asciz	"_STN|f1|<t7>"                  # string offset=6824
.Linfo_string302:
	.asciz	"_Z2f1IJRA3_iEEvv"              # string offset=6837
.Linfo_string303:
	.asciz	"_STN|f1|<int (&)[3]>"          # string offset=6854
.Linfo_string304:
	.asciz	"_Z2f1IJPA3_iEEvv"              # string offset=6875
.Linfo_string305:
	.asciz	"_STN|f1|<int (*)[3]>"          # string offset=6892
.Linfo_string306:
	.asciz	"t1"                            # string offset=6913
.Linfo_string307:
	.asciz	"_Z2f7I2t1Evv"                  # string offset=6916
.Linfo_string308:
	.asciz	"_STN|f7|<t1>"                  # string offset=6929
.Linfo_string309:
	.asciz	"_Z2f8I2t1iEvv"                 # string offset=6942
.Linfo_string310:
	.asciz	"_STN|f8|<t1, int>"             # string offset=6956
.Linfo_string311:
	.asciz	"ns::inner::ttp"                # string offset=6974
.Linfo_string312:
	.asciz	"_ZN2ns8ttp_userINS_5inner3ttpEEEvv" # string offset=6989
.Linfo_string313:
	.asciz	"_STN|ttp_user|<ns::inner::ttp>" # string offset=7024
.Linfo_string314:
	.asciz	"_Z2f1IJPiPDnEEvv"              # string offset=7055
.Linfo_string315:
	.asciz	"_STN|f1|<int *, std::nullptr_t *>" # string offset=7072
.Linfo_string316:
	.asciz	"_STN|t7|<int>"                 # string offset=7106
.Linfo_string317:
	.asciz	"_Z2f1IJ2t7IiEEEvv"             # string offset=7120
.Linfo_string318:
	.asciz	"_STN|f1|<t7<int> >"            # string offset=7138
.Linfo_string319:
	.asciz	"ns::inl::t9"                   # string offset=7157
.Linfo_string320:
	.asciz	"_Z2f7ITtTpTyEN2ns3inl2t9EEvv"  # string offset=7169
.Linfo_string321:
	.asciz	"_STN|f7|<ns::inl::t9>"         # string offset=7198
.Linfo_string322:
	.asciz	"_Z2f1IJU7_AtomiciEEvv"         # string offset=7220
.Linfo_string323:
	.asciz	"f1<_Atomic(int)>"              # string offset=7242
.Linfo_string324:
	.asciz	"_Z2f1IJilVcEEvv"               # string offset=7259
.Linfo_string325:
	.asciz	"_STN|f1|<int, long, volatile char>" # string offset=7275
.Linfo_string326:
	.asciz	"_Z2f1IJDv2_iEEvv"              # string offset=7310
.Linfo_string327:
	.asciz	"f1<__attribute__((__vector_size__(2 * sizeof(int)))) int>" # string offset=7327
.Linfo_string328:
	.asciz	"_Z2f1IJVKPiEEvv"               # string offset=7385
.Linfo_string329:
	.asciz	"_STN|f1|<int *const volatile>" # string offset=7401
.Linfo_string330:
	.asciz	"_Z2f1IJVKvEEvv"                # string offset=7431
.Linfo_string331:
	.asciz	"_STN|f1|<const volatile void>" # string offset=7446
.Linfo_string332:
	.asciz	"t1<(lambda at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:103:12)>" # string offset=7476
.Linfo_string333:
	.asciz	"_Z2f1IJ2t1IJZ4mainE3$_0EEEEvv" # string offset=7608
.Linfo_string334:
	.asciz	"f1<t1<(lambda at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:103:12)> >" # string offset=7638
.Linfo_string335:
	.asciz	"_ZN3t10C2IvEEv"                # string offset=7775
.Linfo_string336:
	.asciz	"_Z2f1IJM3udtKFvvEEEvv"         # string offset=7790
.Linfo_string337:
	.asciz	"_STN|f1|<void (udt::*)() const>" # string offset=7812
.Linfo_string338:
	.asciz	"_Z2f1IJM3udtVFvvREEEvv"        # string offset=7844
.Linfo_string339:
	.asciz	"_STN|f1|<void (udt::*)() volatile &>" # string offset=7867
.Linfo_string340:
	.asciz	"_Z2f1IJM3udtVKFvvOEEEvv"       # string offset=7904
.Linfo_string341:
	.asciz	"_STN|f1|<void (udt::*)() const volatile &&>" # string offset=7928
.Linfo_string342:
	.asciz	"_Z2f9IiEPFvvEv"                # string offset=7972
.Linfo_string343:
	.asciz	"_STN|f9|<int>"                 # string offset=7987
.Linfo_string344:
	.asciz	"_Z2f1IJKPFvvEEEvv"             # string offset=8001
.Linfo_string345:
	.asciz	"_STN|f1|<void (*const)()>"     # string offset=8019
.Linfo_string346:
	.asciz	"_Z2f1IJRA1_KcEEvv"             # string offset=8045
.Linfo_string347:
	.asciz	"_STN|f1|<const char (&)[1]>"   # string offset=8063
.Linfo_string348:
	.asciz	"_Z2f1IJKFvvREEEvv"             # string offset=8091
.Linfo_string349:
	.asciz	"_STN|f1|<void () const &>"     # string offset=8109
.Linfo_string350:
	.asciz	"_Z2f1IJVFvvOEEEvv"             # string offset=8135
.Linfo_string351:
	.asciz	"_STN|f1|<void () volatile &&>" # string offset=8153
.Linfo_string352:
	.asciz	"_Z2f1IJVKFvvEEEvv"             # string offset=8183
.Linfo_string353:
	.asciz	"_STN|f1|<void () const volatile>" # string offset=8201
.Linfo_string354:
	.asciz	"_Z2f1IJA1_KPiEEvv"             # string offset=8234
.Linfo_string355:
	.asciz	"_STN|f1|<int *const[1]>"       # string offset=8252
.Linfo_string356:
	.asciz	"_Z2f1IJRA1_KPiEEvv"            # string offset=8276
.Linfo_string357:
	.asciz	"_STN|f1|<int *const (&)[1]>"   # string offset=8295
.Linfo_string358:
	.asciz	"_Z2f1IJRKM3udtFvvEEEvv"        # string offset=8323
.Linfo_string359:
	.asciz	"_STN|f1|<void (udt::*const &)()>" # string offset=8346
.Linfo_string360:
	.asciz	"_Z2f1IJFPFvfEiEEEvv"           # string offset=8379
.Linfo_string361:
	.asciz	"_STN|f1|<void (*(int))(float)>" # string offset=8399
.Linfo_string362:
	.asciz	"_Z2f1IJA1_2t1IJiEEEEvv"        # string offset=8430
.Linfo_string363:
	.asciz	"_STN|f1|<t1<int>[1]>"          # string offset=8453
.Linfo_string364:
	.asciz	"_Z2f1IJPDoFvvEEEvv"            # string offset=8474
.Linfo_string365:
	.asciz	"f1<void (*)() noexcept>"       # string offset=8493
.Linfo_string366:
	.asciz	"_Z2f1IJFvZ4mainE3$_1EEEvv"     # string offset=8517
.Linfo_string367:
	.asciz	"f1<void ((unnamed struct at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:101:3))>" # string offset=8543
.Linfo_string368:
	.asciz	"_Z2f1IJFvZ4mainE2t8Z4mainE3$_1EEEvv" # string offset=8689
.Linfo_string369:
	.asciz	"f1<void (t8, (unnamed struct at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:101:3))>" # string offset=8725
.Linfo_string370:
	.asciz	"_Z2f1IJFvZ4mainE2t8EEEvv"      # string offset=8875
.Linfo_string371:
	.asciz	"_STN|f1|<void (t8)>"           # string offset=8900
.Linfo_string372:
	.asciz	"_Z19operator_not_reallyIiEvv"  # string offset=8920
.Linfo_string373:
	.asciz	"_STN|operator_not_really|<int>" # string offset=8949
.Linfo_string374:
	.asciz	"_BitInt"                       # string offset=8980
.Linfo_string375:
	.asciz	"V"                             # string offset=8988
.Linfo_string376:
	.asciz	"_Z3f11IDB3_TnT_LS0_2EEvv"      # string offset=8990
.Linfo_string377:
	.asciz	"f11<_BitInt(3), (_BitInt(3))2>" # string offset=9015
.Linfo_string378:
	.asciz	"unsigned _BitInt"              # string offset=9046
.Linfo_string379:
	.asciz	"_Z3f11IKDU5_TnT_LS0_2EEvv"     # string offset=9063
.Linfo_string380:
	.asciz	"f11<const unsigned _BitInt(5), (unsigned _BitInt(5))2>" # string offset=9089
.Linfo_string381:
	.asciz	"_Z3f11IDB65_TnT_LS0_2EEvv"     # string offset=9144
.Linfo_string382:
	.asciz	"f11<_BitInt(65), (_BitInt(65))2>" # string offset=9170
.Linfo_string383:
	.asciz	"_Z3f11IKDU65_TnT_LS0_2EEvv"    # string offset=9203
.Linfo_string384:
	.asciz	"f11<const unsigned _BitInt(65), (unsigned _BitInt(65))2>" # string offset=9230
.Linfo_string385:
	.asciz	"_STN|t1|<>"                    # string offset=9287
.Linfo_string386:
	.asciz	"_Z2f1IJFv2t1IJEES1_EEEvv"      # string offset=9298
.Linfo_string387:
	.asciz	"_STN|f1|<void (t1<>, t1<>)>"   # string offset=9323
.Linfo_string388:
	.asciz	"_Z2f1IJM2t1IJEEiEEvv"          # string offset=9351
.Linfo_string389:
	.asciz	"_STN|f1|<int t1<>::*>"         # string offset=9372
.Linfo_string390:
	.asciz	"_Z2f1IJU9swiftcallFvvEEEvv"    # string offset=9394
.Linfo_string391:
	.asciz	"_STN|f1|<void () __attribute__((swiftcall))>" # string offset=9421
.Linfo_string392:
	.asciz	"_Z2f1IJFivEEEvv"               # string offset=9466
.Linfo_string393:
	.asciz	"f1<int () __attribute__((noreturn))>" # string offset=9482
.Linfo_string394:
	.asciz	"_Z3f10ILN2ns3$_0E0EEvv"        # string offset=9519
.Linfo_string395:
	.asciz	"f10<(ns::(unnamed enum at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:15:1))0>" # string offset=9542
.Linfo_string396:
	.asciz	"_Z2f1IJZN2t83memEvE2t7EEvv"    # string offset=9686
.Linfo_string397:
	.asciz	"_Z2f1IJM2t8FvvEEEvv"           # string offset=9713
.Linfo_string398:
	.asciz	"_STN|f1|<void (t8::*)()>"      # string offset=9733
.Linfo_string399:
	.asciz	"_ZN18complex_type_units2f1Ev"  # string offset=9758
.Linfo_string400:
	.asciz	"f1"                            # string offset=9787
.Linfo_string401:
	.asciz	"_ZN18ptr_to_member_test4testEv" # string offset=9790
.Linfo_string402:
	.asciz	"test"                          # string offset=9821
.Linfo_string403:
	.asciz	"data_mem"                      # string offset=9826
.Linfo_string404:
	.asciz	"S"                             # string offset=9835
.Linfo_string405:
	.asciz	"P"                             # string offset=9837
.Linfo_string406:
	.asciz	"_ZN18ptr_to_member_test1fIXadL_ZNS_1S8data_memEEEEEvv" # string offset=9839
.Linfo_string407:
	.asciz	"f<&ptr_to_member_test::S::data_mem>" # string offset=9893
.Linfo_string408:
	.asciz	"L"                             # string offset=9929
.Linfo_string409:
	.asciz	"v2"                            # string offset=9931
.Linfo_string410:
	.asciz	"N"                             # string offset=9934
.Linfo_string411:
	.asciz	"_STN|t4|<3U>"                  # string offset=9936
.Linfo_string412:
	.asciz	"v1"                            # string offset=9949
.Linfo_string413:
	.asciz	"v6"                            # string offset=9952
.Linfo_string414:
	.asciz	"x"                             # string offset=9955
.Linfo_string415:
	.asciz	"t7i"                           # string offset=9957
.Linfo_string416:
	.asciz	"v3"                            # string offset=9961
.Linfo_string417:
	.asciz	"v4"                            # string offset=9964
.Linfo_string418:
	.asciz	"t11<(anonymous namespace)::LocalEnum, ((anonymous namespace)::LocalEnum)0>" # string offset=9967
.Linfo_string419:
	.asciz	"t12"                           # string offset=10042
.Linfo_string420:
	.asciz	"_STN|t2|<int>"                 # string offset=10046
.Linfo_string421:
	.asciz	"_STN|t2|<float>"               # string offset=10060
.Linfo_string422:
	.asciz	"_STN|t1|<bool>"                # string offset=10076
.Linfo_string423:
	.asciz	"_STN|t2|<bool>"                # string offset=10091
.Linfo_string424:
	.asciz	"_STN|t1|<double>"              # string offset=10106
.Linfo_string425:
	.asciz	"_STN|t2|<double>"              # string offset=10123
.Linfo_string426:
	.asciz	"_STN|t1|<long>"                # string offset=10140
.Linfo_string427:
	.asciz	"_STN|t2|<long>"                # string offset=10155
.Linfo_string428:
	.asciz	"_STN|t1|<short>"               # string offset=10170
.Linfo_string429:
	.asciz	"_STN|t2|<short>"               # string offset=10186
.Linfo_string430:
	.asciz	"_STN|t1|<unsigned int>"        # string offset=10202
.Linfo_string431:
	.asciz	"_STN|t2|<unsigned int>"        # string offset=10225
.Linfo_string432:
	.asciz	"_STN|t1|<unsigned long long>"  # string offset=10248
.Linfo_string433:
	.asciz	"_STN|t2|<unsigned long long>"  # string offset=10277
.Linfo_string434:
	.asciz	"_STN|t1|<long long>"           # string offset=10306
.Linfo_string435:
	.asciz	"_STN|t2|<long long>"           # string offset=10326
.Linfo_string436:
	.asciz	"_STN|t1|<udt>"                 # string offset=10346
.Linfo_string437:
	.asciz	"_STN|t2|<udt>"                 # string offset=10360
.Linfo_string438:
	.asciz	"_STN|t1|<ns::udt>"             # string offset=10374
.Linfo_string439:
	.asciz	"_STN|t2|<ns::udt>"             # string offset=10392
.Linfo_string440:
	.asciz	"_STN|t1|<ns::udt *>"           # string offset=10410
.Linfo_string441:
	.asciz	"_STN|t2|<ns::udt *>"           # string offset=10430
.Linfo_string442:
	.asciz	"_STN|t1|<ns::inner::udt>"      # string offset=10450
.Linfo_string443:
	.asciz	"_STN|t2|<ns::inner::udt>"      # string offset=10475
.Linfo_string444:
	.asciz	"_STN|t1|<t1<int> >"            # string offset=10500
.Linfo_string445:
	.asciz	"_STN|t2|<t1<int> >"            # string offset=10519
.Linfo_string446:
	.asciz	"_STN|t1|<int, float>"          # string offset=10538
.Linfo_string447:
	.asciz	"_STN|t2|<int, float>"          # string offset=10559
.Linfo_string448:
	.asciz	"_STN|t2|<int *>"               # string offset=10580
.Linfo_string449:
	.asciz	"_STN|t1|<int &>"               # string offset=10596
.Linfo_string450:
	.asciz	"_STN|t2|<int &>"               # string offset=10612
.Linfo_string451:
	.asciz	"_STN|t1|<int &&>"              # string offset=10628
.Linfo_string452:
	.asciz	"_STN|t2|<int &&>"              # string offset=10645
.Linfo_string453:
	.asciz	"_STN|t1|<const int>"           # string offset=10662
.Linfo_string454:
	.asciz	"_STN|t2|<const int>"           # string offset=10682
.Linfo_string455:
	.asciz	"_STN|t1|<int[3]>"              # string offset=10702
.Linfo_string456:
	.asciz	"_STN|t2|<int[3]>"              # string offset=10719
.Linfo_string457:
	.asciz	"_STN|t1|<void>"                # string offset=10736
.Linfo_string458:
	.asciz	"_STN|t2|<void>"                # string offset=10751
.Linfo_string459:
	.asciz	"_STN|t1|<outer_class::inner_class>" # string offset=10766
.Linfo_string460:
	.asciz	"_STN|t2|<outer_class::inner_class>" # string offset=10801
.Linfo_string461:
	.asciz	"_STN|t1|<unsigned long>"       # string offset=10836
.Linfo_string462:
	.asciz	"_STN|t2|<unsigned long>"       # string offset=10860
.Linfo_string463:
	.asciz	"_STN|t1|<t3<int, false> >"     # string offset=10884
.Linfo_string464:
	.asciz	"_STN|t2|<t3<int, false> >"     # string offset=10910
.Linfo_string465:
	.asciz	"_STN|t1|<t3<t3<int, false>, false> >" # string offset=10936
.Linfo_string466:
	.asciz	"_STN|t2|<t3<t3<int, false>, false> >" # string offset=10973
.Linfo_string467:
	.asciz	"t2<(lambda at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:103:12)>" # string offset=11010
.Linfo_string468:
	.asciz	"t1<t3<t3<(lambda at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:103:12), false>, false> >" # string offset=11142
.Linfo_string469:
	.asciz	"t2<t3<t3<(lambda at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:103:12), false>, false> >" # string offset=11297
.Linfo_string470:
	.asciz	"_STN|t1|<int (float)>"         # string offset=11452
.Linfo_string471:
	.asciz	"_STN|t2|<int (float)>"         # string offset=11474
.Linfo_string472:
	.asciz	"_STN|t1|<void (...)>"          # string offset=11496
.Linfo_string473:
	.asciz	"_STN|t2|<void (...)>"          # string offset=11517
.Linfo_string474:
	.asciz	"_STN|t1|<void (int, ...)>"     # string offset=11538
.Linfo_string475:
	.asciz	"_STN|t2|<void (int, ...)>"     # string offset=11564
.Linfo_string476:
	.asciz	"_STN|t1|<const int &>"         # string offset=11590
.Linfo_string477:
	.asciz	"_STN|t2|<const int &>"         # string offset=11612
.Linfo_string478:
	.asciz	"_STN|t1|<const int *&>"        # string offset=11634
.Linfo_string479:
	.asciz	"_STN|t2|<const int *&>"        # string offset=11657
.Linfo_string480:
	.asciz	"_STN|t1|<(anonymous namespace)::t5>" # string offset=11680
.Linfo_string481:
	.asciz	"_STN|t2|<(anonymous namespace)::t5>" # string offset=11716
.Linfo_string482:
	.asciz	"_STN|t1|<std::nullptr_t>"      # string offset=11752
.Linfo_string483:
	.asciz	"_STN|t2|<std::nullptr_t>"      # string offset=11777
.Linfo_string484:
	.asciz	"_STN|t1|<long *, long *>"      # string offset=11802
.Linfo_string485:
	.asciz	"_STN|t2|<long *, long *>"      # string offset=11827
.Linfo_string486:
	.asciz	"_STN|t1|<long *, udt *>"       # string offset=11852
.Linfo_string487:
	.asciz	"_STN|t2|<long *, udt *>"       # string offset=11876
.Linfo_string488:
	.asciz	"_STN|t1|<void *const>"         # string offset=11900
.Linfo_string489:
	.asciz	"_STN|t2|<void *const>"         # string offset=11922
.Linfo_string490:
	.asciz	"_STN|t1|<const void *const *>" # string offset=11944
.Linfo_string491:
	.asciz	"_STN|t2|<const void *const *>" # string offset=11974
.Linfo_string492:
	.asciz	"_STN|t1|<void ()>"             # string offset=12004
.Linfo_string493:
	.asciz	"_STN|t2|<void ()>"             # string offset=12022
.Linfo_string494:
	.asciz	"_STN|t1|<void (*)()>"          # string offset=12040
.Linfo_string495:
	.asciz	"_STN|t2|<void (*)()>"          # string offset=12061
.Linfo_string496:
	.asciz	"t1<(lambda at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:103:12) *>" # string offset=12082
.Linfo_string497:
	.asciz	"t2<(lambda at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:103:12) *>" # string offset=12216
.Linfo_string498:
	.asciz	"t1<(unnamed struct at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:101:3)>" # string offset=12350
.Linfo_string499:
	.asciz	"t2<(unnamed struct at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:101:3)>" # string offset=12489
.Linfo_string500:
	.asciz	"t1<(unnamed struct at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:101:3) *>" # string offset=12628
.Linfo_string501:
	.asciz	"t2<(unnamed struct at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:101:3) *>" # string offset=12769
.Linfo_string502:
	.asciz	"_STN|t2|<>"                    # string offset=12910
.Linfo_string503:
	.asciz	"_STN|t1|<const void *, const void *>" # string offset=12921
.Linfo_string504:
	.asciz	"_STN|t2|<const void *, const void *>" # string offset=12958
.Linfo_string505:
	.asciz	"_STN|t1|<t1<int *> *>"         # string offset=12995
.Linfo_string506:
	.asciz	"_STN|t2|<t1<int *> *>"         # string offset=13017
.Linfo_string507:
	.asciz	"_STN|t1|<int *[]>"             # string offset=13039
.Linfo_string508:
	.asciz	"_STN|t2|<int *[]>"             # string offset=13057
.Linfo_string509:
	.asciz	"this"                          # string offset=13075
.Linfo_string510:
	.asciz	"_STN|t1|<t7>"                  # string offset=13080
.Linfo_string511:
	.asciz	"_STN|t2|<t7>"                  # string offset=13093
.Linfo_string512:
	.asciz	"_STN|t1|<int (&)[3]>"          # string offset=13106
.Linfo_string513:
	.asciz	"_STN|t2|<int (&)[3]>"          # string offset=13127
.Linfo_string514:
	.asciz	"_STN|t1|<int (*)[3]>"          # string offset=13148
.Linfo_string515:
	.asciz	"_STN|t2|<int (*)[3]>"          # string offset=13169
.Linfo_string516:
	.asciz	"_STN|t1|<int *, std::nullptr_t *>" # string offset=13190
.Linfo_string517:
	.asciz	"_STN|t2|<int *, std::nullptr_t *>" # string offset=13224
.Linfo_string518:
	.asciz	"_STN|t1|<t7<int> >"            # string offset=13258
.Linfo_string519:
	.asciz	"_STN|t2|<t7<int> >"            # string offset=13277
.Linfo_string520:
	.asciz	"t1<_Atomic(int)>"              # string offset=13296
.Linfo_string521:
	.asciz	"t2<_Atomic(int)>"              # string offset=13313
.Linfo_string522:
	.asciz	"_STN|t1|<int, long, volatile char>" # string offset=13330
.Linfo_string523:
	.asciz	"_STN|t2|<int, long, volatile char>" # string offset=13365
.Linfo_string524:
	.asciz	"t1<__attribute__((__vector_size__(2 * sizeof(int)))) int>" # string offset=13400
.Linfo_string525:
	.asciz	"t2<__attribute__((__vector_size__(2 * sizeof(int)))) int>" # string offset=13458
.Linfo_string526:
	.asciz	"_STN|t1|<int *const volatile>" # string offset=13516
.Linfo_string527:
	.asciz	"_STN|t2|<int *const volatile>" # string offset=13546
.Linfo_string528:
	.asciz	"_STN|t1|<const volatile void>" # string offset=13576
.Linfo_string529:
	.asciz	"_STN|t2|<const volatile void>" # string offset=13606
.Linfo_string530:
	.asciz	"t1<t1<(lambda at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:103:12)> >" # string offset=13636
.Linfo_string531:
	.asciz	"t2<t1<(lambda at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:103:12)> >" # string offset=13773
.Linfo_string532:
	.asciz	"_STN|t1|<void (udt::*)() const>" # string offset=13910
.Linfo_string533:
	.asciz	"_STN|t2|<void (udt::*)() const>" # string offset=13942
.Linfo_string534:
	.asciz	"_STN|t1|<void (udt::*)() volatile &>" # string offset=13974
.Linfo_string535:
	.asciz	"_STN|t2|<void (udt::*)() volatile &>" # string offset=14011
.Linfo_string536:
	.asciz	"_STN|t1|<void (udt::*)() const volatile &&>" # string offset=14048
.Linfo_string537:
	.asciz	"_STN|t2|<void (udt::*)() const volatile &&>" # string offset=14092
.Linfo_string538:
	.asciz	"_STN|t1|<void (*const)()>"     # string offset=14136
.Linfo_string539:
	.asciz	"_STN|t2|<void (*const)()>"     # string offset=14162
.Linfo_string540:
	.asciz	"_STN|t1|<const char (&)[1]>"   # string offset=14188
.Linfo_string541:
	.asciz	"_STN|t2|<const char (&)[1]>"   # string offset=14216
.Linfo_string542:
	.asciz	"_STN|t1|<void () const &>"     # string offset=14244
.Linfo_string543:
	.asciz	"_STN|t2|<void () const &>"     # string offset=14270
.Linfo_string544:
	.asciz	"_STN|t1|<void () volatile &&>" # string offset=14296
.Linfo_string545:
	.asciz	"_STN|t2|<void () volatile &&>" # string offset=14326
.Linfo_string546:
	.asciz	"_STN|t1|<void () const volatile>" # string offset=14356
.Linfo_string547:
	.asciz	"_STN|t2|<void () const volatile>" # string offset=14389
.Linfo_string548:
	.asciz	"_STN|t1|<int *const[1]>"       # string offset=14422
.Linfo_string549:
	.asciz	"_STN|t2|<int *const[1]>"       # string offset=14446
.Linfo_string550:
	.asciz	"_STN|t1|<int *const (&)[1]>"   # string offset=14470
.Linfo_string551:
	.asciz	"_STN|t2|<int *const (&)[1]>"   # string offset=14498
.Linfo_string552:
	.asciz	"_STN|t1|<void (udt::*const &)()>" # string offset=14526
.Linfo_string553:
	.asciz	"_STN|t2|<void (udt::*const &)()>" # string offset=14559
.Linfo_string554:
	.asciz	"_STN|t1|<void (*(int))(float)>" # string offset=14592
.Linfo_string555:
	.asciz	"_STN|t2|<void (*(int))(float)>" # string offset=14623
.Linfo_string556:
	.asciz	"_STN|t1|<t1<int>[1]>"          # string offset=14654
.Linfo_string557:
	.asciz	"_STN|t2|<t1<int>[1]>"          # string offset=14675
.Linfo_string558:
	.asciz	"t1<void (*)() noexcept>"       # string offset=14696
.Linfo_string559:
	.asciz	"t2<void (*)() noexcept>"       # string offset=14720
.Linfo_string560:
	.asciz	"t1<void ((unnamed struct at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:101:3))>" # string offset=14744
.Linfo_string561:
	.asciz	"t2<void ((unnamed struct at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:101:3))>" # string offset=14890
.Linfo_string562:
	.asciz	"t1<void (t8, (unnamed struct at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:101:3))>" # string offset=15036
.Linfo_string563:
	.asciz	"t2<void (t8, (unnamed struct at /proc/self/cwd/cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/Inputs/simplified_template_names.cpp:101:3))>" # string offset=15186
.Linfo_string564:
	.asciz	"_STN|t1|<void (t8)>"           # string offset=15336
.Linfo_string565:
	.asciz	"_STN|t2|<void (t8)>"           # string offset=15356
.Linfo_string566:
	.asciz	"_STN|t1|<void (t1<>, t1<>)>"   # string offset=15376
.Linfo_string567:
	.asciz	"_STN|t2|<void (t1<>, t1<>)>"   # string offset=15404
.Linfo_string568:
	.asciz	"_STN|t1|<int t1<>::*>"         # string offset=15432
.Linfo_string569:
	.asciz	"_STN|t2|<int t1<>::*>"         # string offset=15454
.Linfo_string570:
	.asciz	"_STN|t1|<void () __attribute__((swiftcall))>" # string offset=15476
.Linfo_string571:
	.asciz	"_STN|t2|<void () __attribute__((swiftcall))>" # string offset=15521
.Linfo_string572:
	.asciz	"t1<int () __attribute__((noreturn))>" # string offset=15566
.Linfo_string573:
	.asciz	"t2<int () __attribute__((noreturn))>" # string offset=15603
.Linfo_string574:
	.asciz	"_STN|t1|<void (t8::*)()>"      # string offset=15640
.Linfo_string575:
	.asciz	"_STN|t2|<void (t8::*)()>"      # string offset=15665
.Linfo_string576:
	.asciz	"internal_type"                 # string offset=15690
.Linfo_string577:
	.asciz	"t2<&complex_type_units::external_function>" # string offset=15704
.Linfo_string578:
	.asciz	"_STN|t3|<complex_type_units::t2<&complex_type_units::external_function> >" # string offset=15747
.Linfo_string579:
	.asciz	"_STN|t4|<complex_type_units::(anonymous namespace)::internal_type, complex_type_units::t3<complex_type_units::t2<&complex_type_units::external_function> > >" # string offset=15821
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string8
	.long	.Linfo_string9
	.long	.Linfo_string10
	.long	.Linfo_string11
	.long	.Linfo_string12
	.long	.Linfo_string13
	.long	.Linfo_string14
	.long	.Linfo_string15
	.long	.Linfo_string16
	.long	.Linfo_string17
	.long	.Linfo_string18
	.long	.Linfo_string19
	.long	.Linfo_string20
	.long	.Linfo_string21
	.long	.Linfo_string22
	.long	.Linfo_string23
	.long	.Linfo_string24
	.long	.Linfo_string25
	.long	.Linfo_string26
	.long	.Linfo_string27
	.long	.Linfo_string28
	.long	.Linfo_string29
	.long	.Linfo_string30
	.long	.Linfo_string31
	.long	.Linfo_string32
	.long	.Linfo_string33
	.long	.Linfo_string34
	.long	.Linfo_string35
	.long	.Linfo_string36
	.long	.Linfo_string37
	.long	.Linfo_string38
	.long	.Linfo_string39
	.long	.Linfo_string40
	.long	.Linfo_string41
	.long	.Linfo_string42
	.long	.Linfo_string43
	.long	.Linfo_string44
	.long	.Linfo_string45
	.long	.Linfo_string46
	.long	.Linfo_string47
	.long	.Linfo_string48
	.long	.Linfo_string49
	.long	.Linfo_string50
	.long	.Linfo_string51
	.long	.Linfo_string52
	.long	.Linfo_string53
	.long	.Linfo_string54
	.long	.Linfo_string55
	.long	.Linfo_string56
	.long	.Linfo_string57
	.long	.Linfo_string58
	.long	.Linfo_string59
	.long	.Linfo_string60
	.long	.Linfo_string61
	.long	.Linfo_string62
	.long	.Linfo_string63
	.long	.Linfo_string64
	.long	.Linfo_string65
	.long	.Linfo_string66
	.long	.Linfo_string67
	.long	.Linfo_string68
	.long	.Linfo_string69
	.long	.Linfo_string70
	.long	.Linfo_string71
	.long	.Linfo_string72
	.long	.Linfo_string73
	.long	.Linfo_string74
	.long	.Linfo_string75
	.long	.Linfo_string76
	.long	.Linfo_string77
	.long	.Linfo_string78
	.long	.Linfo_string79
	.long	.Linfo_string80
	.long	.Linfo_string81
	.long	.Linfo_string82
	.long	.Linfo_string83
	.long	.Linfo_string84
	.long	.Linfo_string85
	.long	.Linfo_string86
	.long	.Linfo_string87
	.long	.Linfo_string88
	.long	.Linfo_string89
	.long	.Linfo_string90
	.long	.Linfo_string91
	.long	.Linfo_string92
	.long	.Linfo_string93
	.long	.Linfo_string94
	.long	.Linfo_string95
	.long	.Linfo_string96
	.long	.Linfo_string97
	.long	.Linfo_string98
	.long	.Linfo_string99
	.long	.Linfo_string100
	.long	.Linfo_string101
	.long	.Linfo_string102
	.long	.Linfo_string103
	.long	.Linfo_string104
	.long	.Linfo_string105
	.long	.Linfo_string106
	.long	.Linfo_string107
	.long	.Linfo_string108
	.long	.Linfo_string109
	.long	.Linfo_string110
	.long	.Linfo_string111
	.long	.Linfo_string112
	.long	.Linfo_string113
	.long	.Linfo_string114
	.long	.Linfo_string115
	.long	.Linfo_string116
	.long	.Linfo_string117
	.long	.Linfo_string118
	.long	.Linfo_string119
	.long	.Linfo_string120
	.long	.Linfo_string121
	.long	.Linfo_string122
	.long	.Linfo_string123
	.long	.Linfo_string124
	.long	.Linfo_string125
	.long	.Linfo_string126
	.long	.Linfo_string127
	.long	.Linfo_string128
	.long	.Linfo_string129
	.long	.Linfo_string130
	.long	.Linfo_string131
	.long	.Linfo_string132
	.long	.Linfo_string133
	.long	.Linfo_string134
	.long	.Linfo_string135
	.long	.Linfo_string136
	.long	.Linfo_string137
	.long	.Linfo_string138
	.long	.Linfo_string139
	.long	.Linfo_string140
	.long	.Linfo_string141
	.long	.Linfo_string142
	.long	.Linfo_string143
	.long	.Linfo_string144
	.long	.Linfo_string145
	.long	.Linfo_string146
	.long	.Linfo_string147
	.long	.Linfo_string148
	.long	.Linfo_string149
	.long	.Linfo_string150
	.long	.Linfo_string151
	.long	.Linfo_string152
	.long	.Linfo_string153
	.long	.Linfo_string154
	.long	.Linfo_string155
	.long	.Linfo_string156
	.long	.Linfo_string157
	.long	.Linfo_string158
	.long	.Linfo_string159
	.long	.Linfo_string160
	.long	.Linfo_string161
	.long	.Linfo_string162
	.long	.Linfo_string163
	.long	.Linfo_string164
	.long	.Linfo_string165
	.long	.Linfo_string166
	.long	.Linfo_string167
	.long	.Linfo_string168
	.long	.Linfo_string169
	.long	.Linfo_string170
	.long	.Linfo_string171
	.long	.Linfo_string172
	.long	.Linfo_string173
	.long	.Linfo_string174
	.long	.Linfo_string175
	.long	.Linfo_string176
	.long	.Linfo_string177
	.long	.Linfo_string178
	.long	.Linfo_string179
	.long	.Linfo_string180
	.long	.Linfo_string181
	.long	.Linfo_string182
	.long	.Linfo_string183
	.long	.Linfo_string184
	.long	.Linfo_string185
	.long	.Linfo_string186
	.long	.Linfo_string187
	.long	.Linfo_string188
	.long	.Linfo_string189
	.long	.Linfo_string190
	.long	.Linfo_string191
	.long	.Linfo_string192
	.long	.Linfo_string193
	.long	.Linfo_string194
	.long	.Linfo_string195
	.long	.Linfo_string196
	.long	.Linfo_string197
	.long	.Linfo_string198
	.long	.Linfo_string199
	.long	.Linfo_string200
	.long	.Linfo_string201
	.long	.Linfo_string202
	.long	.Linfo_string203
	.long	.Linfo_string204
	.long	.Linfo_string205
	.long	.Linfo_string206
	.long	.Linfo_string207
	.long	.Linfo_string208
	.long	.Linfo_string209
	.long	.Linfo_string210
	.long	.Linfo_string211
	.long	.Linfo_string212
	.long	.Linfo_string213
	.long	.Linfo_string214
	.long	.Linfo_string215
	.long	.Linfo_string216
	.long	.Linfo_string217
	.long	.Linfo_string218
	.long	.Linfo_string219
	.long	.Linfo_string220
	.long	.Linfo_string221
	.long	.Linfo_string222
	.long	.Linfo_string223
	.long	.Linfo_string224
	.long	.Linfo_string225
	.long	.Linfo_string226
	.long	.Linfo_string227
	.long	.Linfo_string228
	.long	.Linfo_string229
	.long	.Linfo_string230
	.long	.Linfo_string231
	.long	.Linfo_string232
	.long	.Linfo_string233
	.long	.Linfo_string234
	.long	.Linfo_string235
	.long	.Linfo_string236
	.long	.Linfo_string237
	.long	.Linfo_string238
	.long	.Linfo_string239
	.long	.Linfo_string240
	.long	.Linfo_string241
	.long	.Linfo_string242
	.long	.Linfo_string243
	.long	.Linfo_string244
	.long	.Linfo_string245
	.long	.Linfo_string246
	.long	.Linfo_string247
	.long	.Linfo_string248
	.long	.Linfo_string249
	.long	.Linfo_string250
	.long	.Linfo_string251
	.long	.Linfo_string252
	.long	.Linfo_string253
	.long	.Linfo_string254
	.long	.Linfo_string255
	.long	.Linfo_string256
	.long	.Linfo_string257
	.long	.Linfo_string258
	.long	.Linfo_string259
	.long	.Linfo_string260
	.long	.Linfo_string261
	.long	.Linfo_string262
	.long	.Linfo_string263
	.long	.Linfo_string264
	.long	.Linfo_string265
	.long	.Linfo_string266
	.long	.Linfo_string267
	.long	.Linfo_string268
	.long	.Linfo_string269
	.long	.Linfo_string270
	.long	.Linfo_string271
	.long	.Linfo_string272
	.long	.Linfo_string273
	.long	.Linfo_string274
	.long	.Linfo_string275
	.long	.Linfo_string276
	.long	.Linfo_string277
	.long	.Linfo_string278
	.long	.Linfo_string279
	.long	.Linfo_string280
	.long	.Linfo_string281
	.long	.Linfo_string282
	.long	.Linfo_string283
	.long	.Linfo_string284
	.long	.Linfo_string285
	.long	.Linfo_string286
	.long	.Linfo_string287
	.long	.Linfo_string288
	.long	.Linfo_string289
	.long	.Linfo_string290
	.long	.Linfo_string291
	.long	.Linfo_string292
	.long	.Linfo_string293
	.long	.Linfo_string294
	.long	.Linfo_string295
	.long	.Linfo_string296
	.long	.Linfo_string297
	.long	.Linfo_string298
	.long	.Linfo_string299
	.long	.Linfo_string300
	.long	.Linfo_string301
	.long	.Linfo_string302
	.long	.Linfo_string303
	.long	.Linfo_string304
	.long	.Linfo_string305
	.long	.Linfo_string306
	.long	.Linfo_string307
	.long	.Linfo_string308
	.long	.Linfo_string309
	.long	.Linfo_string310
	.long	.Linfo_string311
	.long	.Linfo_string312
	.long	.Linfo_string313
	.long	.Linfo_string314
	.long	.Linfo_string315
	.long	.Linfo_string316
	.long	.Linfo_string317
	.long	.Linfo_string318
	.long	.Linfo_string319
	.long	.Linfo_string320
	.long	.Linfo_string321
	.long	.Linfo_string322
	.long	.Linfo_string323
	.long	.Linfo_string324
	.long	.Linfo_string325
	.long	.Linfo_string326
	.long	.Linfo_string327
	.long	.Linfo_string328
	.long	.Linfo_string329
	.long	.Linfo_string330
	.long	.Linfo_string331
	.long	.Linfo_string332
	.long	.Linfo_string333
	.long	.Linfo_string334
	.long	.Linfo_string335
	.long	.Linfo_string336
	.long	.Linfo_string337
	.long	.Linfo_string338
	.long	.Linfo_string339
	.long	.Linfo_string340
	.long	.Linfo_string341
	.long	.Linfo_string342
	.long	.Linfo_string343
	.long	.Linfo_string344
	.long	.Linfo_string345
	.long	.Linfo_string346
	.long	.Linfo_string347
	.long	.Linfo_string348
	.long	.Linfo_string349
	.long	.Linfo_string350
	.long	.Linfo_string351
	.long	.Linfo_string352
	.long	.Linfo_string353
	.long	.Linfo_string354
	.long	.Linfo_string355
	.long	.Linfo_string356
	.long	.Linfo_string357
	.long	.Linfo_string358
	.long	.Linfo_string359
	.long	.Linfo_string360
	.long	.Linfo_string361
	.long	.Linfo_string362
	.long	.Linfo_string363
	.long	.Linfo_string364
	.long	.Linfo_string365
	.long	.Linfo_string366
	.long	.Linfo_string367
	.long	.Linfo_string368
	.long	.Linfo_string369
	.long	.Linfo_string370
	.long	.Linfo_string371
	.long	.Linfo_string372
	.long	.Linfo_string373
	.long	.Linfo_string374
	.long	.Linfo_string375
	.long	.Linfo_string376
	.long	.Linfo_string377
	.long	.Linfo_string378
	.long	.Linfo_string379
	.long	.Linfo_string380
	.long	.Linfo_string381
	.long	.Linfo_string382
	.long	.Linfo_string383
	.long	.Linfo_string384
	.long	.Linfo_string385
	.long	.Linfo_string386
	.long	.Linfo_string387
	.long	.Linfo_string388
	.long	.Linfo_string389
	.long	.Linfo_string390
	.long	.Linfo_string391
	.long	.Linfo_string392
	.long	.Linfo_string393
	.long	.Linfo_string394
	.long	.Linfo_string395
	.long	.Linfo_string396
	.long	.Linfo_string397
	.long	.Linfo_string398
	.long	.Linfo_string399
	.long	.Linfo_string400
	.long	.Linfo_string401
	.long	.Linfo_string402
	.long	.Linfo_string403
	.long	.Linfo_string404
	.long	.Linfo_string405
	.long	.Linfo_string406
	.long	.Linfo_string407
	.long	.Linfo_string408
	.long	.Linfo_string409
	.long	.Linfo_string410
	.long	.Linfo_string411
	.long	.Linfo_string412
	.long	.Linfo_string413
	.long	.Linfo_string414
	.long	.Linfo_string415
	.long	.Linfo_string416
	.long	.Linfo_string417
	.long	.Linfo_string418
	.long	.Linfo_string419
	.long	.Linfo_string420
	.long	.Linfo_string421
	.long	.Linfo_string422
	.long	.Linfo_string423
	.long	.Linfo_string424
	.long	.Linfo_string425
	.long	.Linfo_string426
	.long	.Linfo_string427
	.long	.Linfo_string428
	.long	.Linfo_string429
	.long	.Linfo_string430
	.long	.Linfo_string431
	.long	.Linfo_string432
	.long	.Linfo_string433
	.long	.Linfo_string434
	.long	.Linfo_string435
	.long	.Linfo_string436
	.long	.Linfo_string437
	.long	.Linfo_string438
	.long	.Linfo_string439
	.long	.Linfo_string440
	.long	.Linfo_string441
	.long	.Linfo_string442
	.long	.Linfo_string443
	.long	.Linfo_string444
	.long	.Linfo_string445
	.long	.Linfo_string446
	.long	.Linfo_string447
	.long	.Linfo_string448
	.long	.Linfo_string449
	.long	.Linfo_string450
	.long	.Linfo_string451
	.long	.Linfo_string452
	.long	.Linfo_string453
	.long	.Linfo_string454
	.long	.Linfo_string455
	.long	.Linfo_string456
	.long	.Linfo_string457
	.long	.Linfo_string458
	.long	.Linfo_string459
	.long	.Linfo_string460
	.long	.Linfo_string461
	.long	.Linfo_string462
	.long	.Linfo_string463
	.long	.Linfo_string464
	.long	.Linfo_string465
	.long	.Linfo_string466
	.long	.Linfo_string467
	.long	.Linfo_string468
	.long	.Linfo_string469
	.long	.Linfo_string470
	.long	.Linfo_string471
	.long	.Linfo_string472
	.long	.Linfo_string473
	.long	.Linfo_string474
	.long	.Linfo_string475
	.long	.Linfo_string476
	.long	.Linfo_string477
	.long	.Linfo_string478
	.long	.Linfo_string479
	.long	.Linfo_string480
	.long	.Linfo_string481
	.long	.Linfo_string482
	.long	.Linfo_string483
	.long	.Linfo_string484
	.long	.Linfo_string485
	.long	.Linfo_string486
	.long	.Linfo_string487
	.long	.Linfo_string488
	.long	.Linfo_string489
	.long	.Linfo_string490
	.long	.Linfo_string491
	.long	.Linfo_string492
	.long	.Linfo_string493
	.long	.Linfo_string494
	.long	.Linfo_string495
	.long	.Linfo_string496
	.long	.Linfo_string497
	.long	.Linfo_string498
	.long	.Linfo_string499
	.long	.Linfo_string500
	.long	.Linfo_string501
	.long	.Linfo_string502
	.long	.Linfo_string503
	.long	.Linfo_string504
	.long	.Linfo_string505
	.long	.Linfo_string506
	.long	.Linfo_string507
	.long	.Linfo_string508
	.long	.Linfo_string509
	.long	.Linfo_string510
	.long	.Linfo_string511
	.long	.Linfo_string512
	.long	.Linfo_string513
	.long	.Linfo_string514
	.long	.Linfo_string515
	.long	.Linfo_string516
	.long	.Linfo_string517
	.long	.Linfo_string518
	.long	.Linfo_string519
	.long	.Linfo_string520
	.long	.Linfo_string521
	.long	.Linfo_string522
	.long	.Linfo_string523
	.long	.Linfo_string524
	.long	.Linfo_string525
	.long	.Linfo_string526
	.long	.Linfo_string527
	.long	.Linfo_string528
	.long	.Linfo_string529
	.long	.Linfo_string530
	.long	.Linfo_string531
	.long	.Linfo_string532
	.long	.Linfo_string533
	.long	.Linfo_string534
	.long	.Linfo_string535
	.long	.Linfo_string536
	.long	.Linfo_string537
	.long	.Linfo_string538
	.long	.Linfo_string539
	.long	.Linfo_string540
	.long	.Linfo_string541
	.long	.Linfo_string542
	.long	.Linfo_string543
	.long	.Linfo_string544
	.long	.Linfo_string545
	.long	.Linfo_string546
	.long	.Linfo_string547
	.long	.Linfo_string548
	.long	.Linfo_string549
	.long	.Linfo_string550
	.long	.Linfo_string551
	.long	.Linfo_string552
	.long	.Linfo_string553
	.long	.Linfo_string554
	.long	.Linfo_string555
	.long	.Linfo_string556
	.long	.Linfo_string557
	.long	.Linfo_string558
	.long	.Linfo_string559
	.long	.Linfo_string560
	.long	.Linfo_string561
	.long	.Linfo_string562
	.long	.Linfo_string563
	.long	.Linfo_string564
	.long	.Linfo_string565
	.long	.Linfo_string566
	.long	.Linfo_string567
	.long	.Linfo_string568
	.long	.Linfo_string569
	.long	.Linfo_string570
	.long	.Linfo_string571
	.long	.Linfo_string572
	.long	.Linfo_string573
	.long	.Linfo_string574
	.long	.Linfo_string575
	.long	.Linfo_string576
	.long	.Linfo_string577
	.long	.Linfo_string578
	.long	.Linfo_string579
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	i
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin1
	.quad	.Lfunc_begin2
	.quad	.Lfunc_begin3
	.quad	.Lfunc_begin4
	.quad	.Lfunc_begin5
	.quad	.Lfunc_begin6
	.quad	.Lfunc_begin7
	.quad	.Lfunc_begin8
	.quad	.Lfunc_begin9
	.quad	.Lfunc_begin10
	.quad	.Lfunc_begin11
	.quad	.Lfunc_begin12
	.quad	.Lfunc_begin13
	.quad	.Lfunc_begin14
	.quad	.Lfunc_begin15
	.quad	.Lfunc_begin16
	.quad	.Lfunc_begin17
	.quad	.Lfunc_begin18
	.quad	.Lfunc_begin19
	.quad	.Lfunc_begin20
	.quad	.Lfunc_begin21
	.quad	.Lfunc_begin22
	.quad	.Lfunc_begin23
	.quad	.Lfunc_begin24
	.quad	.Lfunc_begin25
	.quad	.Lfunc_begin26
	.quad	.Lfunc_begin27
	.quad	.Lfunc_begin28
	.quad	.Lfunc_begin29
	.quad	.Lfunc_begin30
	.quad	.Lfunc_begin31
	.quad	.Lfunc_begin32
	.quad	.Lfunc_begin33
	.quad	.Lfunc_begin34
	.quad	.Lfunc_begin35
	.quad	.Lfunc_begin36
	.quad	.Lfunc_begin37
	.quad	.Lfunc_begin38
	.quad	.Lfunc_begin39
	.quad	.Lfunc_begin40
	.quad	.Lfunc_begin41
	.quad	.Lfunc_begin42
	.quad	.Lfunc_begin43
	.quad	.Lfunc_begin44
	.quad	.Lfunc_begin45
	.quad	.Lfunc_begin46
	.quad	.Lfunc_begin47
	.quad	.Lfunc_begin48
	.quad	.Lfunc_begin49
	.quad	.Lfunc_begin50
	.quad	.Lfunc_begin51
	.quad	.Lfunc_begin52
	.quad	.Lfunc_begin53
	.quad	.Lfunc_begin54
	.quad	.Lfunc_begin55
	.quad	.Lfunc_begin56
	.quad	.Lfunc_begin57
	.quad	.Lfunc_begin58
	.quad	.Lfunc_begin59
	.quad	.Lfunc_begin60
	.quad	.Lfunc_begin61
	.quad	.Lfunc_begin62
	.quad	.Lfunc_begin63
	.quad	.Lfunc_begin64
	.quad	.Lfunc_begin65
	.quad	.Lfunc_begin66
	.quad	.Lfunc_begin67
	.quad	.Lfunc_begin68
	.quad	.Lfunc_begin69
	.quad	.Lfunc_begin70
	.quad	.Lfunc_begin71
	.quad	.Lfunc_begin72
	.quad	.Lfunc_begin73
	.quad	.Lfunc_begin74
	.quad	.Lfunc_begin75
	.quad	.Lfunc_begin76
	.quad	.Lfunc_begin77
	.quad	.Lfunc_begin78
	.quad	.Lfunc_begin79
	.quad	.Lfunc_begin80
	.quad	.Lfunc_begin81
	.quad	.Lfunc_begin82
	.quad	.Lfunc_begin83
	.quad	.Lfunc_begin84
	.quad	.Lfunc_begin85
	.quad	.Lfunc_begin86
	.quad	.Lfunc_begin87
	.quad	.Lfunc_begin88
	.quad	.Lfunc_begin89
	.quad	.Lfunc_begin90
	.quad	.Lfunc_begin91
	.quad	.Lfunc_begin92
	.quad	.Lfunc_begin93
	.quad	.Lfunc_begin94
	.quad	.Lfunc_begin95
	.quad	.Lfunc_begin96
	.quad	.Lfunc_begin97
	.quad	.Lfunc_begin98
	.quad	.Lfunc_begin99
	.quad	.Lfunc_begin100
	.quad	.Lfunc_begin101
	.quad	.Lfunc_begin102
	.quad	.Lfunc_begin103
	.quad	.Lfunc_begin104
	.quad	.Lfunc_begin105
	.quad	.Lfunc_begin106
	.quad	.Lfunc_begin107
	.quad	.Lfunc_begin108
	.quad	.Lfunc_begin109
	.quad	.Lfunc_begin110
	.quad	.Lfunc_begin111
	.quad	.Lfunc_begin112
	.quad	.Lfunc_begin113
	.quad	.Lfunc_begin114
	.quad	.Lfunc_begin115
	.quad	.Lfunc_begin116
	.quad	.Lfunc_begin117
	.quad	.Lfunc_begin118
	.quad	.Lfunc_begin119
	.quad	.Lfunc_begin120
	.quad	.Lfunc_begin121
	.quad	.Lfunc_begin122
	.quad	.Lfunc_begin123
	.quad	.Lfunc_begin124
	.quad	.Lfunc_begin125
	.quad	.Lfunc_begin126
	.quad	.Lfunc_begin127
	.quad	.Lfunc_begin128
	.quad	.Lfunc_begin129
	.quad	.Lfunc_begin130
	.quad	.Lfunc_begin131
	.quad	.Lfunc_begin132
	.quad	.Lfunc_begin133
	.quad	.Lfunc_begin134
	.quad	.Lfunc_begin135
	.quad	.Lfunc_begin136
	.quad	.Lfunc_begin137
	.quad	.Lfunc_begin138
	.quad	.Lfunc_begin139
	.quad	.Lfunc_begin140
	.quad	.Lfunc_begin141
	.quad	.Lfunc_begin142
	.quad	.Lfunc_begin143
	.quad	.Lfunc_begin144
	.quad	_ZN18complex_type_units17external_functionEv
.Ldebug_addr_end0:
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Zli5_suffy
	.addrsig_sym _Z2f1IJiEEvv
	.addrsig_sym _Z2f1IJfEEvv
	.addrsig_sym _Z2f1IJbEEvv
	.addrsig_sym _Z2f1IJdEEvv
	.addrsig_sym _Z2f1IJlEEvv
	.addrsig_sym _Z2f1IJsEEvv
	.addrsig_sym _Z2f1IJjEEvv
	.addrsig_sym _Z2f1IJyEEvv
	.addrsig_sym _Z2f1IJxEEvv
	.addrsig_sym _Z2f1IJ3udtEEvv
	.addrsig_sym _Z2f1IJN2ns3udtEEEvv
	.addrsig_sym _Z2f1IJPN2ns3udtEEEvv
	.addrsig_sym _Z2f1IJN2ns5inner3udtEEEvv
	.addrsig_sym _Z2f1IJ2t1IJiEEEEvv
	.addrsig_sym _Z2f1IJifEEvv
	.addrsig_sym _Z2f1IJPiEEvv
	.addrsig_sym _Z2f1IJRiEEvv
	.addrsig_sym _Z2f1IJOiEEvv
	.addrsig_sym _Z2f1IJKiEEvv
	.addrsig_sym _Z2f1IJA3_iEEvv
	.addrsig_sym _Z2f1IJvEEvv
	.addrsig_sym _Z2f1IJN11outer_class11inner_classEEEvv
	.addrsig_sym _Z2f1IJmEEvv
	.addrsig_sym _Z2f2ILb1ELi3EEvv
	.addrsig_sym _Z2f3IN2ns11EnumerationETpTnT_JLS1_1ELS1_2EEEvv
	.addrsig_sym _Z2f3IN2ns16EnumerationClassETpTnT_JLS1_1ELS1_2EEEvv
	.addrsig_sym _Z2f3IN2ns16EnumerationSmallETpTnT_JLS1_255EEEvv
	.addrsig_sym _Z2f3IN2ns3$_0ETpTnT_JLS1_1ELS1_2EEEvv
	.addrsig_sym _Z2f3IN12_GLOBAL__N_19LocalEnumETpTnT_JLS1_0EEEvv
	.addrsig_sym _Z2f3IPiTpTnT_JXadL_Z1iEEEEvv
	.addrsig_sym _Z2f3IPiTpTnT_JLS0_0EEEvv
	.addrsig_sym _Z2f3ImTpTnT_JLm1EEEvv
	.addrsig_sym _Z2f3IyTpTnT_JLy1EEEvv
	.addrsig_sym _Z2f3IlTpTnT_JLl1EEEvv
	.addrsig_sym _Z2f3IjTpTnT_JLj1EEEvv
	.addrsig_sym _Z2f3IsTpTnT_JLs1EEEvv
	.addrsig_sym _Z2f3IhTpTnT_JLh0EEEvv
	.addrsig_sym _Z2f3IaTpTnT_JLa0EEEvv
	.addrsig_sym _Z2f3ItTpTnT_JLt1ELt2EEEvv
	.addrsig_sym _Z2f3IcTpTnT_JLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv
	.addrsig_sym _Z2f3InTpTnT_JLn18446744073709551614EEEvv
	.addrsig_sym _Z2f4IjLj3EEvv
	.addrsig_sym _Z2f1IJ2t3IiLb0EEEEvv
	.addrsig_sym _Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv
	.addrsig_sym _Z2f1IJZ4mainE3$_0EEvv
	.addrsig_sym _Z2f1IJ2t3IS0_IZ4mainE3$_0Lb0EELb0EEEEvv
	.addrsig_sym _Z2f1IJFifEEEvv
	.addrsig_sym _Z2f1IJFvzEEEvv
	.addrsig_sym _Z2f1IJFvizEEEvv
	.addrsig_sym _Z2f1IJRKiEEvv
	.addrsig_sym _Z2f1IJRPKiEEvv
	.addrsig_sym _Z2f1IJN12_GLOBAL__N_12t5EEEvv
	.addrsig_sym _Z2f1IJDnEEvv
	.addrsig_sym _Z2f1IJPlS0_EEvv
	.addrsig_sym _Z2f1IJPlP3udtEEvv
	.addrsig_sym _Z2f1IJKPvEEvv
	.addrsig_sym _Z2f1IJPKPKvEEvv
	.addrsig_sym _Z2f1IJFvvEEEvv
	.addrsig_sym _Z2f1IJPFvvEEEvv
	.addrsig_sym _Z2f1IJPZ4mainE3$_0EEvv
	.addrsig_sym _Z2f1IJZ4mainE3$_1EEvv
	.addrsig_sym _Z2f1IJPZ4mainE3$_1EEvv
	.addrsig_sym _Z2f5IJ2t1IJiEEEiEvv
	.addrsig_sym _Z2f5IJEiEvv
	.addrsig_sym _Z2f6I2t1IJiEEJEEvv
	.addrsig_sym _Z2f1IJEEvv
	.addrsig_sym _Z2f1IJPKvS1_EEvv
	.addrsig_sym _Z2f1IJP2t1IJPiEEEEvv
	.addrsig_sym _Z2f1IJA_PiEEvv
	.addrsig_sym _ZN2t6lsIiEEvi
	.addrsig_sym _ZN2t6ltIiEEvi
	.addrsig_sym _ZN2t6leIiEEvi
	.addrsig_sym _ZN2t6cvP2t1IJfEEIiEEv
	.addrsig_sym _ZN2t6miIiEEvi
	.addrsig_sym _ZN2t6mlIiEEvi
	.addrsig_sym _ZN2t6dvIiEEvi
	.addrsig_sym _ZN2t6rmIiEEvi
	.addrsig_sym _ZN2t6eoIiEEvi
	.addrsig_sym _ZN2t6anIiEEvi
	.addrsig_sym _ZN2t6orIiEEvi
	.addrsig_sym _ZN2t6coIiEEvv
	.addrsig_sym _ZN2t6ntIiEEvv
	.addrsig_sym _ZN2t6aSIiEEvi
	.addrsig_sym _ZN2t6gtIiEEvi
	.addrsig_sym _ZN2t6cmIiEEvi
	.addrsig_sym _ZN2t6clIiEEvv
	.addrsig_sym _ZN2t6ixIiEEvi
	.addrsig_sym _ZN2t6ssIiEEvi
	.addrsig_sym _ZN2t6nwIiEEPvmT_
	.addrsig_sym _ZN2t6naIiEEPvmT_
	.addrsig_sym _ZN2t6dlIiEEvPvT_
	.addrsig_sym _ZN2t6daIiEEvPvT_
	.addrsig_sym _ZN2t6awIiEEiv
	.addrsig_sym _Z2f1IJZ4mainE2t7EEvv
	.addrsig_sym _Z2f1IJRA3_iEEvv
	.addrsig_sym _Z2f1IJPA3_iEEvv
	.addrsig_sym _Z2f7I2t1Evv
	.addrsig_sym _Z2f8I2t1iEvv
	.addrsig_sym _ZN2ns8ttp_userINS_5inner3ttpEEEvv
	.addrsig_sym _Z2f1IJPiPDnEEvv
	.addrsig_sym _Z2f1IJ2t7IiEEEvv
	.addrsig_sym _Z2f7ITtTpTyEN2ns3inl2t9EEvv
	.addrsig_sym _Z2f1IJU7_AtomiciEEvv
	.addrsig_sym _Z2f1IJilVcEEvv
	.addrsig_sym _Z2f1IJDv2_iEEvv
	.addrsig_sym _Z2f1IJVKPiEEvv
	.addrsig_sym _Z2f1IJVKvEEvv
	.addrsig_sym _Z2f1IJ2t1IJZ4mainE3$_0EEEEvv
	.addrsig_sym _Z2f1IJM3udtKFvvEEEvv
	.addrsig_sym _Z2f1IJM3udtVFvvREEEvv
	.addrsig_sym _Z2f1IJM3udtVKFvvOEEEvv
	.addrsig_sym _Z2f9IiEPFvvEv
	.addrsig_sym _Z2f1IJKPFvvEEEvv
	.addrsig_sym _Z2f1IJRA1_KcEEvv
	.addrsig_sym _Z2f1IJKFvvREEEvv
	.addrsig_sym _Z2f1IJVFvvOEEEvv
	.addrsig_sym _Z2f1IJVKFvvEEEvv
	.addrsig_sym _Z2f1IJA1_KPiEEvv
	.addrsig_sym _Z2f1IJRA1_KPiEEvv
	.addrsig_sym _Z2f1IJRKM3udtFvvEEEvv
	.addrsig_sym _Z2f1IJFPFvfEiEEEvv
	.addrsig_sym _Z2f1IJA1_2t1IJiEEEEvv
	.addrsig_sym _Z2f1IJPDoFvvEEEvv
	.addrsig_sym _Z2f1IJFvZ4mainE3$_1EEEvv
	.addrsig_sym _Z2f1IJFvZ4mainE2t8Z4mainE3$_1EEEvv
	.addrsig_sym _Z2f1IJFvZ4mainE2t8EEEvv
	.addrsig_sym _Z19operator_not_reallyIiEvv
	.addrsig_sym _Z3f11IDB3_TnT_LS0_2EEvv
	.addrsig_sym _Z3f11IKDU5_TnT_LS0_2EEvv
	.addrsig_sym _Z3f11IDB65_TnT_LS0_2EEvv
	.addrsig_sym _Z3f11IKDU65_TnT_LS0_2EEvv
	.addrsig_sym _Z2f1IJFv2t1IJEES1_EEEvv
	.addrsig_sym _Z2f1IJM2t1IJEEiEEvv
	.addrsig_sym _Z2f1IJU9swiftcallFvvEEEvv
	.addrsig_sym _Z2f1IJFivEEEvv
	.addrsig_sym _Z3f10ILN2ns3$_0E0EEvv
	.addrsig_sym _Z2f1IJZN2t83memEvE2t7EEvv
	.addrsig_sym _Z2f1IJM2t8FvvEEEvv
	.addrsig_sym _ZN18ptr_to_member_test1fIXadL_ZNS_1S8data_memEEEEEvv
	.section	.debug_line,"",@progbits
.Lline_table_start0:
