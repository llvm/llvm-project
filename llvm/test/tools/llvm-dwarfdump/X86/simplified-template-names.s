# RUN: llvm-mc < %s -filetype obj -triple x86_64 -o - \
# RUN:   | llvm-dwarfdump --verify - | FileCheck %s

# Checking the LLVM side of cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp
# Compile that file with `-g -Xclang -gsimple-template-names=mangled -Xclang -debug-forward-template-params -S -std=c++20`
# to (re)generate this assembly file - while it might be slightly overkill in
# some ways, it seems small/simple enough to keep this as an exact match for
# that end to end test.

# CHECK: No errors.
	.text
	.file	"simplified_template_names.cpp"
	.file	0 "/usr/local/google/home/blaikie/dev/llvm/src" "cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp" md5 0xeb7736932083f3bb70e031e86d00362e
	.file	1 "/usr" "include/x86_64-linux-gnu/bits/types.h" md5 0x58b79843d97f4309eefa4aa722dac91e
	.file	2 "/usr" "include/x86_64-linux-gnu/bits/stdint-intn.h" md5 0xb26974ec56196748bbc399ee826d2a0e
	.file	3 "/usr/local/google/home/blaikie" "install/bin/../lib/gcc/x86_64-pc-linux-gnu/10.0.0/../../../../include/c++/10.0.0/cstdint"
	.file	4 "/usr" "include/stdint.h" md5 0x8e56ab3ccd56760d8ae9848ebf326071
	.file	5 "/usr" "include/x86_64-linux-gnu/bits/stdint-uintn.h" md5 0x3d2fbc5d847dd222c2fbd70457568436
	.globl	_Zli5_suffy                     # -- Begin function _Zli5_suffy
	.p2align	4, 0x90
	.type	_Zli5_suffy,@function
_Zli5_suffy:                            # @_Zli5_suffy
.Lfunc_begin0:
	.loc	0 142 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:142:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp0:
	.loc	0 142 44 prologue_end           # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:142:44
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	_Zli5_suffy, .Lfunc_end0-_Zli5_suffy
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin1:
	.loc	0 182 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:182:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
.Ltmp2:
	.loc	0 184 8 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:184:8
	movb	.L__const.main.L, %al
	movb	%al, -16(%rbp)
	.loc	0 185 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:185:3
	callq	_Z2f1IJiEEvv
	.loc	0 186 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:186:3
	callq	_Z2f1IJfEEvv
	.loc	0 187 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:187:3
	callq	_Z2f1IJbEEvv
	.loc	0 188 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:188:3
	callq	_Z2f1IJdEEvv
	.loc	0 189 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:189:3
	callq	_Z2f1IJlEEvv
	.loc	0 190 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:190:3
	callq	_Z2f1IJsEEvv
	.loc	0 191 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:191:3
	callq	_Z2f1IJjEEvv
	.loc	0 192 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:192:3
	callq	_Z2f1IJyEEvv
	.loc	0 193 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:193:3
	callq	_Z2f1IJxEEvv
	.loc	0 194 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:194:3
	callq	_Z2f1IJ3udtEEvv
	.loc	0 195 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:195:3
	callq	_Z2f1IJN2ns3udtEEEvv
	.loc	0 196 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:196:3
	callq	_Z2f1IJPN2ns3udtEEEvv
	.loc	0 197 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:197:3
	callq	_Z2f1IJN2ns5inner3udtEEEvv
	.loc	0 198 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:198:3
	callq	_Z2f1IJ2t1IJiEEEEvv
	.loc	0 199 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:199:3
	callq	_Z2f1IJifEEvv
	.loc	0 200 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:200:3
	callq	_Z2f1IJPiEEvv
	.loc	0 201 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:201:3
	callq	_Z2f1IJRiEEvv
	.loc	0 202 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:202:3
	callq	_Z2f1IJOiEEvv
	.loc	0 203 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:203:3
	callq	_Z2f1IJKiEEvv
	.loc	0 204 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:204:3
	callq	_Z2f1IJA3_iEEvv
	.loc	0 205 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:205:3
	callq	_Z2f1IJvEEvv
	.loc	0 206 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:206:3
	callq	_Z2f1IJN11outer_class11inner_classEEEvv
	.loc	0 207 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:207:3
	callq	_Z2f1IJmEEvv
	.loc	0 208 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:208:3
	callq	_Z2f2ILb1ELi3EEvv
	.loc	0 209 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:209:3
	callq	_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv
	.loc	0 210 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:210:3
	callq	_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv
	.loc	0 211 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:211:3
	callq	_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv
	.loc	0 212 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:212:3
	callq	_Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv
	.loc	0 213 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:213:3
	callq	_Z2f3IN12_GLOBAL__N_19LocalEnumEJLS1_0EEEvv
	.loc	0 214 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:214:3
	callq	_Z2f3IPiJXadL_Z1iEEEEvv
	.loc	0 215 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:215:3
	callq	_Z2f3IPiJLS0_0EEEvv
	.loc	0 217 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:217:3
	callq	_Z2f3ImJLm1EEEvv
	.loc	0 218 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:218:3
	callq	_Z2f3IyJLy1EEEvv
	.loc	0 219 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:219:3
	callq	_Z2f3IlJLl1EEEvv
	.loc	0 220 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:220:3
	callq	_Z2f3IjJLj1EEEvv
	.loc	0 221 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:221:3
	callq	_Z2f3IsJLs1EEEvv
	.loc	0 222 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:222:3
	callq	_Z2f3IhJLh0EEEvv
	.loc	0 223 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:223:3
	callq	_Z2f3IaJLa0EEEvv
	.loc	0 224 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:224:3
	callq	_Z2f3ItJLt1ELt2EEEvv
	.loc	0 225 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:225:3
	callq	_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv
	.loc	0 226 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:226:3
	callq	_Z2f3InJLn18446744073709551614EEEvv
	.loc	0 227 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:227:3
	callq	_Z2f4IjLj3EEvv
	.loc	0 228 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:228:3
	callq	_Z2f1IJ2t3IiLb0EEEEvv
	.loc	0 229 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:229:3
	callq	_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv
	.loc	0 230 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:230:3
	callq	_Z2f1IJZ4mainE3$_1EEvv
	.loc	0 232 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:232:3
	callq	_Z2f1IJ2t3IS0_IZ4mainE3$_1Lb0EELb0EEEEvv
	.loc	0 233 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:233:3
	callq	_Z2f1IJFifEEEvv
	.loc	0 234 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:234:3
	callq	_Z2f1IJFvzEEEvv
	.loc	0 235 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:235:3
	callq	_Z2f1IJFvizEEEvv
	.loc	0 236 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:236:3
	callq	_Z2f1IJRKiEEvv
	.loc	0 237 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:237:3
	callq	_Z2f1IJRPKiEEvv
	.loc	0 238 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:238:3
	callq	_Z2f1IJN12_GLOBAL__N_12t5EEEvv
	.loc	0 239 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:239:3
	callq	_Z2f1IJDnEEvv
	.loc	0 240 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:240:3
	callq	_Z2f1IJPlS0_EEvv
	.loc	0 241 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:241:3
	callq	_Z2f1IJPlP3udtEEvv
	.loc	0 242 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:242:3
	callq	_Z2f1IJKPvEEvv
	.loc	0 243 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:243:3
	callq	_Z2f1IJPKPKvEEvv
	.loc	0 244 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:244:3
	callq	_Z2f1IJFvvEEEvv
	.loc	0 245 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:245:3
	callq	_Z2f1IJPFvvEEEvv
	.loc	0 246 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:246:3
	callq	_Z2f1IJPZ4mainE3$_1EEvv
	.loc	0 247 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:247:3
	callq	_Z2f1IJZ4mainE3$_2EEvv
	.loc	0 248 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:248:3
	callq	_Z2f1IJPZ4mainE3$_2EEvv
	.loc	0 249 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:249:3
	callq	_Z2f5IJ2t1IJiEEEiEvv
	.loc	0 250 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:250:3
	callq	_Z2f5IJEiEvv
	.loc	0 251 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:251:3
	callq	_Z2f6I2t1IJiEEJEEvv
	.loc	0 252 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:252:3
	callq	_Z2f1IJEEvv
	.loc	0 253 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:253:3
	callq	_Z2f1IJPKvS1_EEvv
	.loc	0 254 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:254:3
	callq	_Z2f1IJP2t1IJPiEEEEvv
	.loc	0 255 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:255:3
	callq	_Z2f1IJA_PiEEvv
	.loc	0 257 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:257:6
	leaq	-40(%rbp), %rdi
	movl	$1, %esi
	callq	_ZN2t6lsIiEEvi
	.loc	0 258 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:258:6
	leaq	-40(%rbp), %rdi
	movl	$1, %esi
	callq	_ZN2t6ltIiEEvi
	.loc	0 259 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:259:6
	leaq	-40(%rbp), %rdi
	movl	$1, %esi
	callq	_ZN2t6leIiEEvi
	.loc	0 260 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:260:6
	leaq	-40(%rbp), %rdi
	callq	_ZN2t6cvP2t1IJfEEIiEEv
	.loc	0 261 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:261:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6miIiEEvi
	.loc	0 262 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:262:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6mlIiEEvi
	.loc	0 263 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:263:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6dvIiEEvi
	.loc	0 264 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:264:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6rmIiEEvi
	.loc	0 265 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:265:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6eoIiEEvi
	.loc	0 266 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:266:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6anIiEEvi
	.loc	0 267 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:267:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6orIiEEvi
	.loc	0 268 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:268:6
	leaq	-40(%rbp), %rdi
	callq	_ZN2t6coIiEEvv
	.loc	0 269 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:269:6
	leaq	-40(%rbp), %rdi
	callq	_ZN2t6ntIiEEvv
	.loc	0 270 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:270:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6aSIiEEvi
	.loc	0 271 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:271:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6gtIiEEvi
	.loc	0 272 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:272:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6cmIiEEvi
	.loc	0 273 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:273:6
	leaq	-40(%rbp), %rdi
	callq	_ZN2t6clIiEEvv
	.loc	0 274 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:274:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6ixIiEEvi
	.loc	0 275 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:275:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6ssIiEEvi
	.loc	0 276 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:276:3
	xorl	%eax, %eax
	movl	%eax, %edi
	xorl	%esi, %esi
	callq	_ZN2t6nwIiEEPvmT_
	.loc	0 277 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:277:3
	xorl	%eax, %eax
	movl	%eax, %edi
	xorl	%esi, %esi
	callq	_ZN2t6naIiEEPvmT_
	.loc	0 278 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:278:3
	xorl	%eax, %eax
	movl	%eax, %edi
	xorl	%esi, %esi
	callq	_ZN2t6dlIiEEvPvT_
	.loc	0 279 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:279:3
	xorl	%eax, %eax
	movl	%eax, %edi
	xorl	%esi, %esi
	callq	_ZN2t6daIiEEvPvT_
	.loc	0 280 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:280:6
	leaq	-40(%rbp), %rdi
	callq	_ZN2t6awIiEEiv
	.loc	0 281 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:281:3
	movl	$42, %edi
	callq	_Zli5_suffy
	.loc	0 283 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:283:3
	callq	_Z2f1IJZ4mainE2t7EEvv
	.loc	0 284 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:284:3
	callq	_Z2f1IJRA3_iEEvv
	.loc	0 285 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:285:3
	callq	_Z2f1IJPA3_iEEvv
	.loc	0 286 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:286:3
	callq	_Z2f7I2t1Evv
	.loc	0 287 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:287:3
	callq	_Z2f8I2t1iEvv
	.loc	0 289 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:289:3
	callq	_ZN2ns8ttp_userINS_5inner3ttpEEEvv
	.loc	0 290 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:290:3
	callq	_Z2f1IJPiPDnEEvv
	.loc	0 292 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:292:3
	callq	_Z2f1IJ2t7IiEEEvv
	.loc	0 293 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:293:3
	callq	_Z2f7IN2ns3inl2t9EEvv
	.loc	0 294 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:294:3
	callq	_Z2f1IJU7_AtomiciEEvv
	.loc	0 295 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:295:3
	callq	_Z2f1IJilVcEEvv
	.loc	0 296 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:296:3
	callq	_Z2f1IJDv2_iEEvv
	.loc	0 297 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:297:3
	callq	_Z2f1IJVKPiEEvv
	.loc	0 298 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:298:3
	callq	_Z2f1IJVKvEEvv
	.loc	0 299 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:299:3
	callq	_Z2f1IJ2t1IJZ4mainE3$_1EEEEvv
	.loc	0 300 7                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:300:7
	leaq	-56(%rbp), %rdi
	callq	_ZN3t10C2IvEEv
	.loc	0 301 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:301:3
	callq	_Z2f1IJM3udtKFvvEEEvv
	.loc	0 302 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:302:3
	callq	_Z2f1IJM3udtVFvvREEEvv
	.loc	0 303 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:303:3
	callq	_Z2f1IJM3udtVKFvvOEEEvv
	.loc	0 304 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:304:3
	callq	_Z2f9IiEPFvvEv
	.loc	0 305 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:305:3
	callq	_Z2f1IJKPFvvEEEvv
	.loc	0 306 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:306:3
	callq	_Z2f1IJRA1_KcEEvv
	.loc	0 307 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:307:3
	callq	_Z2f1IJKFvvREEEvv
	.loc	0 308 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:308:3
	callq	_Z2f1IJVFvvOEEEvv
	.loc	0 309 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:309:3
	callq	_Z2f1IJVKFvvEEEvv
	.loc	0 310 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:310:3
	callq	_Z2f1IJA1_KPiEEvv
	.loc	0 311 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:311:3
	callq	_Z2f1IJRA1_KPiEEvv
	.loc	0 312 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:312:3
	callq	_Z2f1IJRKM3udtFvvEEEvv
	.loc	0 313 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:313:3
	callq	_Z2f1IJFPFvfEiEEEvv
	.loc	0 314 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:314:3
	callq	_Z2f1IJA1_2t1IJiEEEEvv
	.loc	0 315 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:315:3
	callq	_Z2f1IJPDoFvvEEEvv
	.loc	0 316 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:316:3
	callq	_Z2f1IJFvZ4mainE3$_2EEEvv
	.loc	0 318 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:318:3
	callq	_Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv
	.loc	0 319 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:319:3
	callq	_Z2f1IJFvZ4mainE2t8EEEvv
	.loc	0 320 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:320:3
	callq	_Z19operator_not_reallyIiEvv
	.loc	0 322 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:322:3
	callq	_Z2f1IJDB3_EEvv
	.loc	0 323 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:323:3
	callq	_Z2f1IJKDU5_EEvv
	.loc	0 324 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:324:3
	callq	_Z2f1IJFv2t1IJEES1_EEEvv
	.loc	0 325 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:325:3
	callq	_Z2f1IJM2t1IJEEiEEvv
	.loc	0 326 1                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:326:1
	xorl	%eax, %eax
	addq	$64, %rsp
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
	.p2align	4, 0x90
	.type	_Z2f1IJiEEvv,@function
_Z2f1IJiEEvv:                           # @_Z2f1IJiEEvv
.Lfunc_begin2:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp4:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJfEEvv,@function
_Z2f1IJfEEvv:                           # @_Z2f1IJfEEvv
.Lfunc_begin3:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp6:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJbEEvv,@function
_Z2f1IJbEEvv:                           # @_Z2f1IJbEEvv
.Lfunc_begin4:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp8:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJdEEvv,@function
_Z2f1IJdEEvv:                           # @_Z2f1IJdEEvv
.Lfunc_begin5:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp10:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJlEEvv,@function
_Z2f1IJlEEvv:                           # @_Z2f1IJlEEvv
.Lfunc_begin6:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp12:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJsEEvv,@function
_Z2f1IJsEEvv:                           # @_Z2f1IJsEEvv
.Lfunc_begin7:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp14:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJjEEvv,@function
_Z2f1IJjEEvv:                           # @_Z2f1IJjEEvv
.Lfunc_begin8:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp16:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJyEEvv,@function
_Z2f1IJyEEvv:                           # @_Z2f1IJyEEvv
.Lfunc_begin9:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp18:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJxEEvv,@function
_Z2f1IJxEEvv:                           # @_Z2f1IJxEEvv
.Lfunc_begin10:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp20:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJ3udtEEvv,@function
_Z2f1IJ3udtEEvv:                        # @_Z2f1IJ3udtEEvv
.Lfunc_begin11:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp22:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJN2ns3udtEEEvv,@function
_Z2f1IJN2ns3udtEEEvv:                   # @_Z2f1IJN2ns3udtEEEvv
.Lfunc_begin12:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp24:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJPN2ns3udtEEEvv,@function
_Z2f1IJPN2ns3udtEEEvv:                  # @_Z2f1IJPN2ns3udtEEEvv
.Lfunc_begin13:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp26:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJN2ns5inner3udtEEEvv,@function
_Z2f1IJN2ns5inner3udtEEEvv:             # @_Z2f1IJN2ns5inner3udtEEEvv
.Lfunc_begin14:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp28:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJ2t1IJiEEEEvv,@function
_Z2f1IJ2t1IJiEEEEvv:                    # @_Z2f1IJ2t1IJiEEEEvv
.Lfunc_begin15:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp30:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJifEEvv,@function
_Z2f1IJifEEvv:                          # @_Z2f1IJifEEvv
.Lfunc_begin16:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp32:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJPiEEvv,@function
_Z2f1IJPiEEvv:                          # @_Z2f1IJPiEEvv
.Lfunc_begin17:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp34:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJRiEEvv,@function
_Z2f1IJRiEEvv:                          # @_Z2f1IJRiEEvv
.Lfunc_begin18:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp36:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJOiEEvv,@function
_Z2f1IJOiEEvv:                          # @_Z2f1IJOiEEvv
.Lfunc_begin19:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp38:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJKiEEvv,@function
_Z2f1IJKiEEvv:                          # @_Z2f1IJKiEEvv
.Lfunc_begin20:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp40:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJA3_iEEvv,@function
_Z2f1IJA3_iEEvv:                        # @_Z2f1IJA3_iEEvv
.Lfunc_begin21:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp42:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJvEEvv,@function
_Z2f1IJvEEvv:                           # @_Z2f1IJvEEvv
.Lfunc_begin22:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp44:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJN11outer_class11inner_classEEEvv,@function
_Z2f1IJN11outer_class11inner_classEEEvv: # @_Z2f1IJN11outer_class11inner_classEEEvv
.Lfunc_begin23:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp46:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJmEEvv,@function
_Z2f1IJmEEvv:                           # @_Z2f1IJmEEvv
.Lfunc_begin24:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp48:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f2ILb1ELi3EEvv,@function
_Z2f2ILb1ELi3EEvv:                      # @_Z2f2ILb1ELi3EEvv
.Lfunc_begin25:
	.loc	0 38 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp50:
	.loc	0 39 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:39:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp51:
.Lfunc_end25:
	.size	_Z2f2ILb1ELi3EEvv, .Lfunc_end25-_Z2f2ILb1ELi3EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv,"axG",@progbits,_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv,comdat
	.weak	_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv # -- Begin function _Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv,@function
_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv: # @_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv
.Lfunc_begin26:
	.loc	0 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp52:
	.loc	0 42 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:42:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp53:
.Lfunc_end26:
	.size	_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv, .Lfunc_end26-_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv,"axG",@progbits,_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv,comdat
	.weak	_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv # -- Begin function _Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv,@function
_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv: # @_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv
.Lfunc_begin27:
	.loc	0 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp54:
	.loc	0 42 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:42:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp55:
.Lfunc_end27:
	.size	_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv, .Lfunc_end27-_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv,"axG",@progbits,_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv,comdat
	.weak	_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv # -- Begin function _Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv,@function
_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv: # @_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv
.Lfunc_begin28:
	.loc	0 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp56:
	.loc	0 42 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:42:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp57:
.Lfunc_end28:
	.size	_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv, .Lfunc_end28-_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function _Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv
	.type	_Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv,@function
_Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv:       # @"_Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv"
.Lfunc_begin29:
	.loc	0 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp58:
	.loc	0 42 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:42:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp59:
.Lfunc_end29:
	.size	_Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv, .Lfunc_end29-_Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _Z2f3IN12_GLOBAL__N_19LocalEnumEJLS1_0EEEvv
	.type	_Z2f3IN12_GLOBAL__N_19LocalEnumEJLS1_0EEEvv,@function
_Z2f3IN12_GLOBAL__N_19LocalEnumEJLS1_0EEEvv: # @_Z2f3IN12_GLOBAL__N_19LocalEnumEJLS1_0EEEvv
.Lfunc_begin30:
	.loc	0 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp60:
	.loc	0 42 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:42:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp61:
.Lfunc_end30:
	.size	_Z2f3IN12_GLOBAL__N_19LocalEnumEJLS1_0EEEvv, .Lfunc_end30-_Z2f3IN12_GLOBAL__N_19LocalEnumEJLS1_0EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IPiJXadL_Z1iEEEEvv,"axG",@progbits,_Z2f3IPiJXadL_Z1iEEEEvv,comdat
	.weak	_Z2f3IPiJXadL_Z1iEEEEvv         # -- Begin function _Z2f3IPiJXadL_Z1iEEEEvv
	.p2align	4, 0x90
	.type	_Z2f3IPiJXadL_Z1iEEEEvv,@function
_Z2f3IPiJXadL_Z1iEEEEvv:                # @_Z2f3IPiJXadL_Z1iEEEEvv
.Lfunc_begin31:
	.loc	0 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp62:
	.loc	0 42 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:42:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp63:
.Lfunc_end31:
	.size	_Z2f3IPiJXadL_Z1iEEEEvv, .Lfunc_end31-_Z2f3IPiJXadL_Z1iEEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IPiJLS0_0EEEvv,"axG",@progbits,_Z2f3IPiJLS0_0EEEvv,comdat
	.weak	_Z2f3IPiJLS0_0EEEvv             # -- Begin function _Z2f3IPiJLS0_0EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IPiJLS0_0EEEvv,@function
_Z2f3IPiJLS0_0EEEvv:                    # @_Z2f3IPiJLS0_0EEEvv
.Lfunc_begin32:
	.loc	0 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp64:
	.loc	0 42 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:42:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp65:
.Lfunc_end32:
	.size	_Z2f3IPiJLS0_0EEEvv, .Lfunc_end32-_Z2f3IPiJLS0_0EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3ImJLm1EEEvv,"axG",@progbits,_Z2f3ImJLm1EEEvv,comdat
	.weak	_Z2f3ImJLm1EEEvv                # -- Begin function _Z2f3ImJLm1EEEvv
	.p2align	4, 0x90
	.type	_Z2f3ImJLm1EEEvv,@function
_Z2f3ImJLm1EEEvv:                       # @_Z2f3ImJLm1EEEvv
.Lfunc_begin33:
	.loc	0 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp66:
	.loc	0 42 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:42:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp67:
.Lfunc_end33:
	.size	_Z2f3ImJLm1EEEvv, .Lfunc_end33-_Z2f3ImJLm1EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IyJLy1EEEvv,"axG",@progbits,_Z2f3IyJLy1EEEvv,comdat
	.weak	_Z2f3IyJLy1EEEvv                # -- Begin function _Z2f3IyJLy1EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IyJLy1EEEvv,@function
_Z2f3IyJLy1EEEvv:                       # @_Z2f3IyJLy1EEEvv
.Lfunc_begin34:
	.loc	0 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp68:
	.loc	0 42 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:42:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp69:
.Lfunc_end34:
	.size	_Z2f3IyJLy1EEEvv, .Lfunc_end34-_Z2f3IyJLy1EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IlJLl1EEEvv,"axG",@progbits,_Z2f3IlJLl1EEEvv,comdat
	.weak	_Z2f3IlJLl1EEEvv                # -- Begin function _Z2f3IlJLl1EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IlJLl1EEEvv,@function
_Z2f3IlJLl1EEEvv:                       # @_Z2f3IlJLl1EEEvv
.Lfunc_begin35:
	.loc	0 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp70:
	.loc	0 42 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:42:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp71:
.Lfunc_end35:
	.size	_Z2f3IlJLl1EEEvv, .Lfunc_end35-_Z2f3IlJLl1EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IjJLj1EEEvv,"axG",@progbits,_Z2f3IjJLj1EEEvv,comdat
	.weak	_Z2f3IjJLj1EEEvv                # -- Begin function _Z2f3IjJLj1EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IjJLj1EEEvv,@function
_Z2f3IjJLj1EEEvv:                       # @_Z2f3IjJLj1EEEvv
.Lfunc_begin36:
	.loc	0 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp72:
	.loc	0 42 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:42:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp73:
.Lfunc_end36:
	.size	_Z2f3IjJLj1EEEvv, .Lfunc_end36-_Z2f3IjJLj1EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IsJLs1EEEvv,"axG",@progbits,_Z2f3IsJLs1EEEvv,comdat
	.weak	_Z2f3IsJLs1EEEvv                # -- Begin function _Z2f3IsJLs1EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IsJLs1EEEvv,@function
_Z2f3IsJLs1EEEvv:                       # @_Z2f3IsJLs1EEEvv
.Lfunc_begin37:
	.loc	0 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp74:
	.loc	0 42 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:42:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp75:
.Lfunc_end37:
	.size	_Z2f3IsJLs1EEEvv, .Lfunc_end37-_Z2f3IsJLs1EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IhJLh0EEEvv,"axG",@progbits,_Z2f3IhJLh0EEEvv,comdat
	.weak	_Z2f3IhJLh0EEEvv                # -- Begin function _Z2f3IhJLh0EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IhJLh0EEEvv,@function
_Z2f3IhJLh0EEEvv:                       # @_Z2f3IhJLh0EEEvv
.Lfunc_begin38:
	.loc	0 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp76:
	.loc	0 42 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:42:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp77:
.Lfunc_end38:
	.size	_Z2f3IhJLh0EEEvv, .Lfunc_end38-_Z2f3IhJLh0EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IaJLa0EEEvv,"axG",@progbits,_Z2f3IaJLa0EEEvv,comdat
	.weak	_Z2f3IaJLa0EEEvv                # -- Begin function _Z2f3IaJLa0EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IaJLa0EEEvv,@function
_Z2f3IaJLa0EEEvv:                       # @_Z2f3IaJLa0EEEvv
.Lfunc_begin39:
	.loc	0 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp78:
	.loc	0 42 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:42:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp79:
.Lfunc_end39:
	.size	_Z2f3IaJLa0EEEvv, .Lfunc_end39-_Z2f3IaJLa0EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3ItJLt1ELt2EEEvv,"axG",@progbits,_Z2f3ItJLt1ELt2EEEvv,comdat
	.weak	_Z2f3ItJLt1ELt2EEEvv            # -- Begin function _Z2f3ItJLt1ELt2EEEvv
	.p2align	4, 0x90
	.type	_Z2f3ItJLt1ELt2EEEvv,@function
_Z2f3ItJLt1ELt2EEEvv:                   # @_Z2f3ItJLt1ELt2EEEvv
.Lfunc_begin40:
	.loc	0 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp80:
	.loc	0 42 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:42:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp81:
.Lfunc_end40:
	.size	_Z2f3ItJLt1ELt2EEEvv, .Lfunc_end40-_Z2f3ItJLt1ELt2EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv,"axG",@progbits,_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv,comdat
	.weak	_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv # -- Begin function _Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv,@function
_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv: # @_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv
.Lfunc_begin41:
	.loc	0 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp82:
	.loc	0 42 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:42:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp83:
.Lfunc_end41:
	.size	_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv, .Lfunc_end41-_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3InJLn18446744073709551614EEEvv,"axG",@progbits,_Z2f3InJLn18446744073709551614EEEvv,comdat
	.weak	_Z2f3InJLn18446744073709551614EEEvv # -- Begin function _Z2f3InJLn18446744073709551614EEEvv
	.p2align	4, 0x90
	.type	_Z2f3InJLn18446744073709551614EEEvv,@function
_Z2f3InJLn18446744073709551614EEEvv:    # @_Z2f3InJLn18446744073709551614EEEvv
.Lfunc_begin42:
	.loc	0 41 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp84:
	.loc	0 42 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:42:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp85:
.Lfunc_end42:
	.size	_Z2f3InJLn18446744073709551614EEEvv, .Lfunc_end42-_Z2f3InJLn18446744073709551614EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f4IjLj3EEvv,"axG",@progbits,_Z2f4IjLj3EEvv,comdat
	.weak	_Z2f4IjLj3EEvv                  # -- Begin function _Z2f4IjLj3EEvv
	.p2align	4, 0x90
	.type	_Z2f4IjLj3EEvv,@function
_Z2f4IjLj3EEvv:                         # @_Z2f4IjLj3EEvv
.Lfunc_begin43:
	.loc	0 44 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp86:
	.loc	0 45 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:45:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJ2t3IiLb0EEEEvv,@function
_Z2f1IJ2t3IiLb0EEEEvv:                  # @_Z2f1IJ2t3IiLb0EEEEvv
.Lfunc_begin44:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp88:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv,@function
_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv:         # @_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv
.Lfunc_begin45:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp90:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp91:
.Lfunc_end45:
	.size	_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv, .Lfunc_end45-_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJZ4mainE3$_1EEvv
	.type	_Z2f1IJZ4mainE3$_1EEvv,@function
_Z2f1IJZ4mainE3$_1EEvv:                 # @"_Z2f1IJZ4mainE3$_1EEvv"
.Lfunc_begin46:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp92:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp93:
.Lfunc_end46:
	.size	_Z2f1IJZ4mainE3$_1EEvv, .Lfunc_end46-_Z2f1IJZ4mainE3$_1EEvv
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJ2t3IS0_IZ4mainE3$_1Lb0EELb0EEEEvv
	.type	_Z2f1IJ2t3IS0_IZ4mainE3$_1Lb0EELb0EEEEvv,@function
_Z2f1IJ2t3IS0_IZ4mainE3$_1Lb0EELb0EEEEvv: # @"_Z2f1IJ2t3IS0_IZ4mainE3$_1Lb0EELb0EEEEvv"
.Lfunc_begin47:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp94:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp95:
.Lfunc_end47:
	.size	_Z2f1IJ2t3IS0_IZ4mainE3$_1Lb0EELb0EEEEvv, .Lfunc_end47-_Z2f1IJ2t3IS0_IZ4mainE3$_1Lb0EELb0EEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJFifEEEvv,"axG",@progbits,_Z2f1IJFifEEEvv,comdat
	.weak	_Z2f1IJFifEEEvv                 # -- Begin function _Z2f1IJFifEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJFifEEEvv,@function
_Z2f1IJFifEEEvv:                        # @_Z2f1IJFifEEEvv
.Lfunc_begin48:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp96:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJFvzEEEvv,@function
_Z2f1IJFvzEEEvv:                        # @_Z2f1IJFvzEEEvv
.Lfunc_begin49:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp98:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJFvizEEEvv,@function
_Z2f1IJFvizEEEvv:                       # @_Z2f1IJFvizEEEvv
.Lfunc_begin50:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp100:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJRKiEEvv,@function
_Z2f1IJRKiEEvv:                         # @_Z2f1IJRKiEEvv
.Lfunc_begin51:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp102:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJRPKiEEvv,@function
_Z2f1IJRPKiEEvv:                        # @_Z2f1IJRPKiEEvv
.Lfunc_begin52:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp104:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp105:
.Lfunc_end52:
	.size	_Z2f1IJRPKiEEvv, .Lfunc_end52-_Z2f1IJRPKiEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJN12_GLOBAL__N_12t5EEEvv
	.type	_Z2f1IJN12_GLOBAL__N_12t5EEEvv,@function
_Z2f1IJN12_GLOBAL__N_12t5EEEvv:         # @_Z2f1IJN12_GLOBAL__N_12t5EEEvv
.Lfunc_begin53:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp106:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJDnEEvv,@function
_Z2f1IJDnEEvv:                          # @_Z2f1IJDnEEvv
.Lfunc_begin54:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp108:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJPlS0_EEvv,@function
_Z2f1IJPlS0_EEvv:                       # @_Z2f1IJPlS0_EEvv
.Lfunc_begin55:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp110:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJPlP3udtEEvv,@function
_Z2f1IJPlP3udtEEvv:                     # @_Z2f1IJPlP3udtEEvv
.Lfunc_begin56:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp112:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJKPvEEvv,@function
_Z2f1IJKPvEEvv:                         # @_Z2f1IJKPvEEvv
.Lfunc_begin57:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp114:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJPKPKvEEvv,@function
_Z2f1IJPKPKvEEvv:                       # @_Z2f1IJPKPKvEEvv
.Lfunc_begin58:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp116:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJFvvEEEvv,@function
_Z2f1IJFvvEEEvv:                        # @_Z2f1IJFvvEEEvv
.Lfunc_begin59:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp118:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJPFvvEEEvv,@function
_Z2f1IJPFvvEEEvv:                       # @_Z2f1IJPFvvEEEvv
.Lfunc_begin60:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp120:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp121:
.Lfunc_end60:
	.size	_Z2f1IJPFvvEEEvv, .Lfunc_end60-_Z2f1IJPFvvEEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJPZ4mainE3$_1EEvv
	.type	_Z2f1IJPZ4mainE3$_1EEvv,@function
_Z2f1IJPZ4mainE3$_1EEvv:                # @"_Z2f1IJPZ4mainE3$_1EEvv"
.Lfunc_begin61:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp122:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp123:
.Lfunc_end61:
	.size	_Z2f1IJPZ4mainE3$_1EEvv, .Lfunc_end61-_Z2f1IJPZ4mainE3$_1EEvv
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJZ4mainE3$_2EEvv
	.type	_Z2f1IJZ4mainE3$_2EEvv,@function
_Z2f1IJZ4mainE3$_2EEvv:                 # @"_Z2f1IJZ4mainE3$_2EEvv"
.Lfunc_begin62:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp124:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp125:
.Lfunc_end62:
	.size	_Z2f1IJZ4mainE3$_2EEvv, .Lfunc_end62-_Z2f1IJZ4mainE3$_2EEvv
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJPZ4mainE3$_2EEvv
	.type	_Z2f1IJPZ4mainE3$_2EEvv,@function
_Z2f1IJPZ4mainE3$_2EEvv:                # @"_Z2f1IJPZ4mainE3$_2EEvv"
.Lfunc_begin63:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp126:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp127:
.Lfunc_end63:
	.size	_Z2f1IJPZ4mainE3$_2EEvv, .Lfunc_end63-_Z2f1IJPZ4mainE3$_2EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f5IJ2t1IJiEEEiEvv,"axG",@progbits,_Z2f5IJ2t1IJiEEEiEvv,comdat
	.weak	_Z2f5IJ2t1IJiEEEiEvv            # -- Begin function _Z2f5IJ2t1IJiEEEiEvv
	.p2align	4, 0x90
	.type	_Z2f5IJ2t1IJiEEEiEvv,@function
_Z2f5IJ2t1IJiEEEiEvv:                   # @_Z2f5IJ2t1IJiEEEiEvv
.Lfunc_begin64:
	.loc	0 62 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:62:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp128:
	.loc	0 62 13 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:62:13
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
	.p2align	4, 0x90
	.type	_Z2f5IJEiEvv,@function
_Z2f5IJEiEvv:                           # @_Z2f5IJEiEvv
.Lfunc_begin65:
	.loc	0 62 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:62:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp130:
	.loc	0 62 13 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:62:13
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
	.p2align	4, 0x90
	.type	_Z2f6I2t1IJiEEJEEvv,@function
_Z2f6I2t1IJiEEJEEvv:                    # @_Z2f6I2t1IJiEEJEEvv
.Lfunc_begin66:
	.loc	0 64 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:64:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp132:
	.loc	0 64 13 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:64:13
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
	.p2align	4, 0x90
	.type	_Z2f1IJEEvv,@function
_Z2f1IJEEvv:                            # @_Z2f1IJEEvv
.Lfunc_begin67:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp134:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJPKvS1_EEvv,@function
_Z2f1IJPKvS1_EEvv:                      # @_Z2f1IJPKvS1_EEvv
.Lfunc_begin68:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp136:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJP2t1IJPiEEEEvv,@function
_Z2f1IJP2t1IJPiEEEEvv:                  # @_Z2f1IJP2t1IJPiEEEEvv
.Lfunc_begin69:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp138:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJA_PiEEvv,@function
_Z2f1IJA_PiEEvv:                        # @_Z2f1IJA_PiEEvv
.Lfunc_begin70:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp140:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_ZN2t6lsIiEEvi,@function
_ZN2t6lsIiEEvi:                         # @_ZN2t6lsIiEEvi
.Lfunc_begin71:
	.loc	0 67 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:67:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp142:
	.loc	0 68 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:68:3
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
	.p2align	4, 0x90
	.type	_ZN2t6ltIiEEvi,@function
_ZN2t6ltIiEEvi:                         # @_ZN2t6ltIiEEvi
.Lfunc_begin72:
	.loc	0 70 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:70:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp144:
	.loc	0 71 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:71:3
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
	.p2align	4, 0x90
	.type	_ZN2t6leIiEEvi,@function
_ZN2t6leIiEEvi:                         # @_ZN2t6leIiEEvi
.Lfunc_begin73:
	.loc	0 73 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:73:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp146:
	.loc	0 74 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:74:3
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
	.p2align	4, 0x90
	.type	_ZN2t6cvP2t1IJfEEIiEEv,@function
_ZN2t6cvP2t1IJfEEIiEEv:                 # @_ZN2t6cvP2t1IJfEEIiEEv
.Lfunc_begin74:
	.loc	0 76 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:76:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp148:
	.loc	0 77 5 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:77:5
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
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
	.p2align	4, 0x90
	.type	_ZN2t6miIiEEvi,@function
_ZN2t6miIiEEvi:                         # @_ZN2t6miIiEEvi
.Lfunc_begin75:
	.loc	0 80 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:80:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp150:
	.loc	0 81 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:81:3
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
	.p2align	4, 0x90
	.type	_ZN2t6mlIiEEvi,@function
_ZN2t6mlIiEEvi:                         # @_ZN2t6mlIiEEvi
.Lfunc_begin76:
	.loc	0 83 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:83:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp152:
	.loc	0 84 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:84:3
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
	.p2align	4, 0x90
	.type	_ZN2t6dvIiEEvi,@function
_ZN2t6dvIiEEvi:                         # @_ZN2t6dvIiEEvi
.Lfunc_begin77:
	.loc	0 86 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:86:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp154:
	.loc	0 87 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:87:3
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
	.p2align	4, 0x90
	.type	_ZN2t6rmIiEEvi,@function
_ZN2t6rmIiEEvi:                         # @_ZN2t6rmIiEEvi
.Lfunc_begin78:
	.loc	0 89 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:89:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp156:
	.loc	0 90 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:90:3
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
	.p2align	4, 0x90
	.type	_ZN2t6eoIiEEvi,@function
_ZN2t6eoIiEEvi:                         # @_ZN2t6eoIiEEvi
.Lfunc_begin79:
	.loc	0 92 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:92:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp158:
	.loc	0 93 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:93:3
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
	.p2align	4, 0x90
	.type	_ZN2t6anIiEEvi,@function
_ZN2t6anIiEEvi:                         # @_ZN2t6anIiEEvi
.Lfunc_begin80:
	.loc	0 95 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:95:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp160:
	.loc	0 96 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:96:3
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
	.p2align	4, 0x90
	.type	_ZN2t6orIiEEvi,@function
_ZN2t6orIiEEvi:                         # @_ZN2t6orIiEEvi
.Lfunc_begin81:
	.loc	0 98 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:98:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp162:
	.loc	0 99 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:99:3
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
	.p2align	4, 0x90
	.type	_ZN2t6coIiEEvv,@function
_ZN2t6coIiEEvv:                         # @_ZN2t6coIiEEvv
.Lfunc_begin82:
	.loc	0 101 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:101:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp164:
	.loc	0 102 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:102:3
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
	.p2align	4, 0x90
	.type	_ZN2t6ntIiEEvv,@function
_ZN2t6ntIiEEvv:                         # @_ZN2t6ntIiEEvv
.Lfunc_begin83:
	.loc	0 104 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:104:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp166:
	.loc	0 105 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:105:3
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
	.p2align	4, 0x90
	.type	_ZN2t6aSIiEEvi,@function
_ZN2t6aSIiEEvi:                         # @_ZN2t6aSIiEEvi
.Lfunc_begin84:
	.loc	0 107 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:107:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp168:
	.loc	0 108 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:108:3
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
	.p2align	4, 0x90
	.type	_ZN2t6gtIiEEvi,@function
_ZN2t6gtIiEEvi:                         # @_ZN2t6gtIiEEvi
.Lfunc_begin85:
	.loc	0 110 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:110:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp170:
	.loc	0 111 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:111:3
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
	.p2align	4, 0x90
	.type	_ZN2t6cmIiEEvi,@function
_ZN2t6cmIiEEvi:                         # @_ZN2t6cmIiEEvi
.Lfunc_begin86:
	.loc	0 113 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:113:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp172:
	.loc	0 114 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:114:3
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
	.p2align	4, 0x90
	.type	_ZN2t6clIiEEvv,@function
_ZN2t6clIiEEvv:                         # @_ZN2t6clIiEEvv
.Lfunc_begin87:
	.loc	0 116 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:116:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp174:
	.loc	0 117 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:117:3
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
	.p2align	4, 0x90
	.type	_ZN2t6ixIiEEvi,@function
_ZN2t6ixIiEEvi:                         # @_ZN2t6ixIiEEvi
.Lfunc_begin88:
	.loc	0 119 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:119:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp176:
	.loc	0 120 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:120:3
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
	.p2align	4, 0x90
	.type	_ZN2t6ssIiEEvi,@function
_ZN2t6ssIiEEvi:                         # @_ZN2t6ssIiEEvi
.Lfunc_begin89:
	.loc	0 122 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:122:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp178:
	.loc	0 123 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:123:3
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
	.p2align	4, 0x90
	.type	_ZN2t6nwIiEEPvmT_,@function
_ZN2t6nwIiEEPvmT_:                      # @_ZN2t6nwIiEEPvmT_
.Lfunc_begin90:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Lfunc_end90:
	.size	_ZN2t6nwIiEEPvmT_, .Lfunc_end90-_ZN2t6nwIiEEPvmT_
	.cfi_endproc
	.file	6 "/usr/local/google/home/blaikie" "install/bin/../lib/gcc/x86_64-pc-linux-gnu/10.0.0/../../../../include/c++/10.0.0/x86_64-pc-linux-gnu/bits/c++config.h" md5 0x13aeea7494b5171d821b8ca5f04a3bc5
                                        # -- End function
	.section	.text._ZN2t6naIiEEPvmT_,"axG",@progbits,_ZN2t6naIiEEPvmT_,comdat
	.weak	_ZN2t6naIiEEPvmT_               # -- Begin function _ZN2t6naIiEEPvmT_
	.p2align	4, 0x90
	.type	_ZN2t6naIiEEPvmT_,@function
_ZN2t6naIiEEPvmT_:                      # @_ZN2t6naIiEEPvmT_
.Lfunc_begin91:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Lfunc_end91:
	.size	_ZN2t6naIiEEPvmT_, .Lfunc_end91-_ZN2t6naIiEEPvmT_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6dlIiEEvPvT_,"axG",@progbits,_ZN2t6dlIiEEvPvT_,comdat
	.weak	_ZN2t6dlIiEEvPvT_               # -- Begin function _ZN2t6dlIiEEvPvT_
	.p2align	4, 0x90
	.type	_ZN2t6dlIiEEvPvT_,@function
_ZN2t6dlIiEEvPvT_:                      # @_ZN2t6dlIiEEvPvT_
.Lfunc_begin92:
	.loc	0 129 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:129:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp180:
	.loc	0 130 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:130:3
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
	.p2align	4, 0x90
	.type	_ZN2t6daIiEEvPvT_,@function
_ZN2t6daIiEEvPvT_:                      # @_ZN2t6daIiEEvPvT_
.Lfunc_begin93:
	.loc	0 136 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:136:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp182:
	.loc	0 137 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:137:3
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
	.p2align	4, 0x90
	.type	_ZN2t6awIiEEiv,@function
_ZN2t6awIiEEiv:                         # @_ZN2t6awIiEEiv
.Lfunc_begin94:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Lfunc_end94:
	.size	_ZN2t6awIiEEiv, .Lfunc_end94-_ZN2t6awIiEEiv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJZ4mainE2t7EEvv
	.type	_Z2f1IJZ4mainE2t7EEvv,@function
_Z2f1IJZ4mainE2t7EEvv:                  # @_Z2f1IJZ4mainE2t7EEvv
.Lfunc_begin95:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp184:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJRA3_iEEvv,@function
_Z2f1IJRA3_iEEvv:                       # @_Z2f1IJRA3_iEEvv
.Lfunc_begin96:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp186:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJPA3_iEEvv,@function
_Z2f1IJPA3_iEEvv:                       # @_Z2f1IJPA3_iEEvv
.Lfunc_begin97:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp188:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f7I2t1Evv,@function
_Z2f7I2t1Evv:                           # @_Z2f7I2t1Evv
.Lfunc_begin98:
	.loc	0 143 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:143:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp190:
	.loc	0 143 53 prologue_end           # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:143:53
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
	.p2align	4, 0x90
	.type	_Z2f8I2t1iEvv,@function
_Z2f8I2t1iEvv:                          # @_Z2f8I2t1iEvv
.Lfunc_begin99:
	.loc	0 144 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:144:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp192:
	.loc	0 144 66 prologue_end           # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:144:66
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
	.p2align	4, 0x90
	.type	_ZN2ns8ttp_userINS_5inner3ttpEEEvv,@function
_ZN2ns8ttp_userINS_5inner3ttpEEEvv:     # @_ZN2ns8ttp_userINS_5inner3ttpEEEvv
.Lfunc_begin100:
	.loc	0 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp194:
	.loc	0 26 19 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:19
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
	.p2align	4, 0x90
	.type	_Z2f1IJPiPDnEEvv,@function
_Z2f1IJPiPDnEEvv:                       # @_Z2f1IJPiPDnEEvv
.Lfunc_begin101:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp196:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJ2t7IiEEEvv,@function
_Z2f1IJ2t7IiEEEvv:                      # @_Z2f1IJ2t7IiEEEvv
.Lfunc_begin102:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp198:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp199:
.Lfunc_end102:
	.size	_Z2f1IJ2t7IiEEEvv, .Lfunc_end102-_Z2f1IJ2t7IiEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f7IN2ns3inl2t9EEvv,"axG",@progbits,_Z2f7IN2ns3inl2t9EEvv,comdat
	.weak	_Z2f7IN2ns3inl2t9EEvv           # -- Begin function _Z2f7IN2ns3inl2t9EEvv
	.p2align	4, 0x90
	.type	_Z2f7IN2ns3inl2t9EEvv,@function
_Z2f7IN2ns3inl2t9EEvv:                  # @_Z2f7IN2ns3inl2t9EEvv
.Lfunc_begin103:
	.loc	0 143 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:143:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp200:
	.loc	0 143 53 prologue_end           # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:143:53
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp201:
.Lfunc_end103:
	.size	_Z2f7IN2ns3inl2t9EEvv, .Lfunc_end103-_Z2f7IN2ns3inl2t9EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJU7_AtomiciEEvv,"axG",@progbits,_Z2f1IJU7_AtomiciEEvv,comdat
	.weak	_Z2f1IJU7_AtomiciEEvv           # -- Begin function _Z2f1IJU7_AtomiciEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJU7_AtomiciEEvv,@function
_Z2f1IJU7_AtomiciEEvv:                  # @_Z2f1IJU7_AtomiciEEvv
.Lfunc_begin104:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp202:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJilVcEEvv,@function
_Z2f1IJilVcEEvv:                        # @_Z2f1IJilVcEEvv
.Lfunc_begin105:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp204:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJDv2_iEEvv,@function
_Z2f1IJDv2_iEEvv:                       # @_Z2f1IJDv2_iEEvv
.Lfunc_begin106:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp206:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJVKPiEEvv,@function
_Z2f1IJVKPiEEvv:                        # @_Z2f1IJVKPiEEvv
.Lfunc_begin107:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp208:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJVKvEEvv,@function
_Z2f1IJVKvEEvv:                         # @_Z2f1IJVKvEEvv
.Lfunc_begin108:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp210:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp211:
.Lfunc_end108:
	.size	_Z2f1IJVKvEEvv, .Lfunc_end108-_Z2f1IJVKvEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJ2t1IJZ4mainE3$_1EEEEvv
	.type	_Z2f1IJ2t1IJZ4mainE3$_1EEEEvv,@function
_Z2f1IJ2t1IJZ4mainE3$_1EEEEvv:          # @"_Z2f1IJ2t1IJZ4mainE3$_1EEEEvv"
.Lfunc_begin109:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp212:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp213:
.Lfunc_end109:
	.size	_Z2f1IJ2t1IJZ4mainE3$_1EEEEvv, .Lfunc_end109-_Z2f1IJ2t1IJZ4mainE3$_1EEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3t10C2IvEEv,"axG",@progbits,_ZN3t10C2IvEEv,comdat
	.weak	_ZN3t10C2IvEEv                  # -- Begin function _ZN3t10C2IvEEv
	.p2align	4, 0x90
	.type	_ZN3t10C2IvEEv,@function
_ZN3t10C2IvEEv:                         # @_ZN3t10C2IvEEv
.Lfunc_begin110:
	.loc	0 167 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:167:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp214:
	.loc	0 167 11 prologue_end           # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:167:11
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
	.p2align	4, 0x90
	.type	_Z2f1IJM3udtKFvvEEEvv,@function
_Z2f1IJM3udtKFvvEEEvv:                  # @_Z2f1IJM3udtKFvvEEEvv
.Lfunc_begin111:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp216:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJM3udtVFvvREEEvv,@function
_Z2f1IJM3udtVFvvREEEvv:                 # @_Z2f1IJM3udtVFvvREEEvv
.Lfunc_begin112:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp218:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJM3udtVKFvvOEEEvv,@function
_Z2f1IJM3udtVKFvvOEEEvv:                # @_Z2f1IJM3udtVKFvvOEEEvv
.Lfunc_begin113:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp220:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f9IiEPFvvEv,@function
_Z2f9IiEPFvvEv:                         # @_Z2f9IiEPFvvEv
.Lfunc_begin114:
	.loc	0 162 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:162:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp222:
	.loc	0 163 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:163:3
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
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
	.p2align	4, 0x90
	.type	_Z2f1IJKPFvvEEEvv,@function
_Z2f1IJKPFvvEEEvv:                      # @_Z2f1IJKPFvvEEEvv
.Lfunc_begin115:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp224:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJRA1_KcEEvv,@function
_Z2f1IJRA1_KcEEvv:                      # @_Z2f1IJRA1_KcEEvv
.Lfunc_begin116:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp226:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJKFvvREEEvv,@function
_Z2f1IJKFvvREEEvv:                      # @_Z2f1IJKFvvREEEvv
.Lfunc_begin117:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp228:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJVFvvOEEEvv,@function
_Z2f1IJVFvvOEEEvv:                      # @_Z2f1IJVFvvOEEEvv
.Lfunc_begin118:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp230:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJVKFvvEEEvv,@function
_Z2f1IJVKFvvEEEvv:                      # @_Z2f1IJVKFvvEEEvv
.Lfunc_begin119:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp232:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJA1_KPiEEvv,@function
_Z2f1IJA1_KPiEEvv:                      # @_Z2f1IJA1_KPiEEvv
.Lfunc_begin120:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp234:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJRA1_KPiEEvv,@function
_Z2f1IJRA1_KPiEEvv:                     # @_Z2f1IJRA1_KPiEEvv
.Lfunc_begin121:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp236:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJRKM3udtFvvEEEvv,@function
_Z2f1IJRKM3udtFvvEEEvv:                 # @_Z2f1IJRKM3udtFvvEEEvv
.Lfunc_begin122:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp238:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJFPFvfEiEEEvv,@function
_Z2f1IJFPFvfEiEEEvv:                    # @_Z2f1IJFPFvfEiEEEvv
.Lfunc_begin123:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp240:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJA1_2t1IJiEEEEvv,@function
_Z2f1IJA1_2t1IJiEEEEvv:                 # @_Z2f1IJA1_2t1IJiEEEEvv
.Lfunc_begin124:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp242:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z2f1IJPDoFvvEEEvv,@function
_Z2f1IJPDoFvvEEEvv:                     # @_Z2f1IJPDoFvvEEEvv
.Lfunc_begin125:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp244:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp245:
.Lfunc_end125:
	.size	_Z2f1IJPDoFvvEEEvv, .Lfunc_end125-_Z2f1IJPDoFvvEEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJFvZ4mainE3$_2EEEvv
	.type	_Z2f1IJFvZ4mainE3$_2EEEvv,@function
_Z2f1IJFvZ4mainE3$_2EEEvv:              # @"_Z2f1IJFvZ4mainE3$_2EEEvv"
.Lfunc_begin126:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp246:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp247:
.Lfunc_end126:
	.size	_Z2f1IJFvZ4mainE3$_2EEEvv, .Lfunc_end126-_Z2f1IJFvZ4mainE3$_2EEEvv
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv
	.type	_Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv,@function
_Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv:    # @"_Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv"
.Lfunc_begin127:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp248:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp249:
.Lfunc_end127:
	.size	_Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv, .Lfunc_end127-_Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJFvZ4mainE2t8EEEvv
	.type	_Z2f1IJFvZ4mainE2t8EEEvv,@function
_Z2f1IJFvZ4mainE2t8EEEvv:               # @_Z2f1IJFvZ4mainE2t8EEEvv
.Lfunc_begin128:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp250:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
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
	.p2align	4, 0x90
	.type	_Z19operator_not_reallyIiEvv,@function
_Z19operator_not_reallyIiEvv:           # @_Z19operator_not_reallyIiEvv
.Lfunc_begin129:
	.loc	0 171 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:171:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp252:
	.loc	0 172 1 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:172:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp253:
.Lfunc_end129:
	.size	_Z19operator_not_reallyIiEvv, .Lfunc_end129-_Z19operator_not_reallyIiEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJDB3_EEvv,"axG",@progbits,_Z2f1IJDB3_EEvv,comdat
	.weak	_Z2f1IJDB3_EEvv                 # -- Begin function _Z2f1IJDB3_EEvv
	.p2align	4, 0x90
	.type	_Z2f1IJDB3_EEvv,@function
_Z2f1IJDB3_EEvv:                        # @_Z2f1IJDB3_EEvv
.Lfunc_begin130:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp254:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp255:
.Lfunc_end130:
	.size	_Z2f1IJDB3_EEvv, .Lfunc_end130-_Z2f1IJDB3_EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJKDU5_EEvv,"axG",@progbits,_Z2f1IJKDU5_EEvv,comdat
	.weak	_Z2f1IJKDU5_EEvv                # -- Begin function _Z2f1IJKDU5_EEvv
	.p2align	4, 0x90
	.type	_Z2f1IJKDU5_EEvv,@function
_Z2f1IJKDU5_EEvv:                       # @_Z2f1IJKDU5_EEvv
.Lfunc_begin131:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp256:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp257:
.Lfunc_end131:
	.size	_Z2f1IJKDU5_EEvv, .Lfunc_end131-_Z2f1IJKDU5_EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJFv2t1IJEES1_EEEvv,"axG",@progbits,_Z2f1IJFv2t1IJEES1_EEEvv,comdat
	.weak	_Z2f1IJFv2t1IJEES1_EEEvv        # -- Begin function _Z2f1IJFv2t1IJEES1_EEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJFv2t1IJEES1_EEEvv,@function
_Z2f1IJFv2t1IJEES1_EEEvv:               # @_Z2f1IJFv2t1IJEES1_EEEvv
.Lfunc_begin132:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp258:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp259:
.Lfunc_end132:
	.size	_Z2f1IJFv2t1IJEES1_EEEvv, .Lfunc_end132-_Z2f1IJFv2t1IJEES1_EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJM2t1IJEEiEEvv,"axG",@progbits,_Z2f1IJM2t1IJEEiEEvv,comdat
	.weak	_Z2f1IJM2t1IJEEiEEvv            # -- Begin function _Z2f1IJM2t1IJEEiEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJM2t1IJEEiEEvv,@function
_Z2f1IJM2t1IJEEiEEvv:                   # @_Z2f1IJM2t1IJEEiEEvv
.Lfunc_begin133:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp260:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp261:
.Lfunc_end133:
	.size	_Z2f1IJM2t1IJEEiEEvv, .Lfunc_end133-_Z2f1IJM2t1IJEEiEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.globl	_ZN2t83memEv                    # -- Begin function _ZN2t83memEv
	.p2align	4, 0x90
	.type	_ZN2t83memEv,@function
_ZN2t83memEv:                           # @_ZN2t83memEv
.Lfunc_begin134:
	.loc	0 327 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:327:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
.Ltmp262:
	.loc	0 329 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:329:3
	callq	_Z2f1IJZN2t83memEvE2t7EEvv
	.loc	0 330 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:330:3
	callq	_Z2f1IJM2t8FvvEEEvv
	.loc	0 331 1                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:331:1
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp263:
.Lfunc_end134:
	.size	_ZN2t83memEv, .Lfunc_end134-_ZN2t83memEv
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJZN2t83memEvE2t7EEvv
	.type	_Z2f1IJZN2t83memEvE2t7EEvv,@function
_Z2f1IJZN2t83memEvE2t7EEvv:             # @_Z2f1IJZN2t83memEvE2t7EEvv
.Lfunc_begin135:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp264:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp265:
.Lfunc_end135:
	.size	_Z2f1IJZN2t83memEvE2t7EEvv, .Lfunc_end135-_Z2f1IJZN2t83memEvE2t7EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJM2t8FvvEEEvv,"axG",@progbits,_Z2f1IJM2t8FvvEEEvv,comdat
	.weak	_Z2f1IJM2t8FvvEEEvv             # -- Begin function _Z2f1IJM2t8FvvEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJM2t8FvvEEEvv,@function
_Z2f1IJM2t8FvvEEEvv:                    # @_Z2f1IJM2t8FvvEEEvv
.Lfunc_begin136:
	.loc	0 33 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:33:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp266:
	.loc	0 36 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:36:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp267:
.Lfunc_end136:
	.size	_Z2f1IJM2t8FvvEEEvv, .Lfunc_end136-_Z2f1IJM2t8FvvEEEvv
	.cfi_endproc
                                        # -- End function
	.type	i,@object                       # @i
	.data
	.globl	i
	.p2align	2
i:
	.long	3                               # 0x3
	.size	i, 4

	.type	.L__const.main.L,@object        # @__const.main.L
	.section	.rodata,"a",@progbits
.L__const.main.L:
	.zero	1
	.size	.L__const.main.L, 1

	.section	".linker-options","e",@llvm_linker_options
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
	.byte	37                              # DW_FORM_strx1
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
	.byte	10                              # Abbreviation Code
	.byte	40                              # DW_TAG_enumerator
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	28                              # DW_AT_const_value
	.byte	13                              # DW_FORM_sdata
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
	.byte	15                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	16                              # Abbreviation Code
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
	.byte	17                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
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
	.byte	18                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	30                              # DW_AT_default_value
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	19                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	52                              # DW_AT_artificial
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	20                              # Abbreviation Code
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
	.byte	21                              # Abbreviation Code
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
	.byte	22                              # Abbreviation Code
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
	.byte	23                              # Abbreviation Code
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
	.byte	24                              # Abbreviation Code
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
	.byte	25                              # Abbreviation Code
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
	.byte	26                              # Abbreviation Code
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
	.byte	27                              # Abbreviation Code
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
	.byte	28                              # Abbreviation Code
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
	.byte	29                              # Abbreviation Code
	.byte	58                              # DW_TAG_imported_module
	.byte	0                               # DW_CHILDREN_no
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	5                               # DW_FORM_data2
	.byte	24                              # DW_AT_import
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	30                              # Abbreviation Code
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
	.byte	31                              # Abbreviation Code
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
	.byte	32                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	33                              # Abbreviation Code
	.ascii	"\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	34                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	35                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	36                              # Abbreviation Code
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
	.byte	37                              # Abbreviation Code
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
	.byte	38                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	28                              # DW_AT_const_value
	.byte	15                              # DW_FORM_udata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	39                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	28                              # DW_AT_const_value
	.byte	13                              # DW_FORM_sdata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	40                              # Abbreviation Code
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
	.byte	41                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	42                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	28                              # DW_AT_const_value
	.byte	10                              # DW_FORM_block1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	43                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	30                              # DW_AT_default_value
	.byte	25                              # DW_FORM_flag_present
	.byte	28                              # DW_AT_const_value
	.byte	15                              # DW_FORM_udata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	44                              # Abbreviation Code
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
	.byte	38                              # DW_FORM_strx2
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	45                              # Abbreviation Code
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
	.byte	46                              # Abbreviation Code
	.ascii	"\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	47                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	30                              # DW_AT_default_value
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	48                              # Abbreviation Code
	.ascii	"\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	49                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	50                              # Abbreviation Code
	.ascii	"\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	51                              # Abbreviation Code
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
	.byte	52                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	53                              # Abbreviation Code
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
	.byte	54                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	30                              # DW_AT_default_value
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	55                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	56                              # Abbreviation Code
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
	.byte	57                              # Abbreviation Code
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
	.byte	58                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
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
	.byte	71                              # DW_AT_specification
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	60                              # Abbreviation Code
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
	.byte	61                              # Abbreviation Code
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
	.byte	62                              # Abbreviation Code
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
	.byte	63                              # Abbreviation Code
	.byte	16                              # DW_TAG_reference_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	64                              # Abbreviation Code
	.byte	66                              # DW_TAG_rvalue_reference_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	65                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	66                              # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	67                              # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	55                              # DW_AT_count
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	68                              # Abbreviation Code
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
	.byte	69                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	70                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	71                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	72                              # Abbreviation Code
	.byte	24                              # DW_TAG_unspecified_parameters
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	73                              # Abbreviation Code
	.byte	59                              # DW_TAG_unspecified_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	74                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	75                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	76                              # Abbreviation Code
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
	.byte	77                              # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	78                              # Abbreviation Code
	.byte	71                              # DW_TAG_atomic_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	79                              # Abbreviation Code
	.byte	53                              # DW_TAG_volatile_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	80                              # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.ascii	"\207B"                         # DW_AT_GNU_vector
	.byte	25                              # DW_FORM_flag_present
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	81                              # Abbreviation Code
	.byte	53                              # DW_TAG_volatile_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	82                              # Abbreviation Code
	.byte	31                              # DW_TAG_ptr_to_member_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	29                              # DW_AT_containing_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	83                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	119                             # DW_AT_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	84                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	120                             # DW_AT_rvalue_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	85                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	119                             # DW_AT_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	86                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	120                             # DW_AT_rvalue_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	87                              # Abbreviation Code
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
	.byte	88                              # Abbreviation Code
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
	.byte	89                              # Abbreviation Code
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
	.byte	90                              # Abbreviation Code
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
	.byte	91                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
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
	.byte	1                               # Abbrev [1] 0xc:0x290a DW_TAG_compile_unit
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
	.byte	0                               # DW_AT_decl_file
	.byte	54                              # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	161
	.byte	0
	.byte	3                               # Abbrev [3] 0x36:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	4                               # Abbrev [4] 0x3a:0x11 DW_TAG_namespace
	.byte	5                               # Abbrev [5] 0x3b:0xd DW_TAG_enumeration_type
	.long	75                              # DW_AT_type
	.byte	7                               # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	59                              # DW_AT_decl_line
	.byte	6                               # Abbrev [6] 0x44:0x3 DW_TAG_enumerator
	.byte	6                               # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	7                               # Abbrev [7] 0x48:0x2 DW_TAG_structure_type
	.byte	254                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0x4b:0x4 DW_TAG_base_type
	.byte	5                               # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	8                               # Abbrev [8] 0x4f:0x63 DW_TAG_namespace
	.byte	8                               # DW_AT_name
	.byte	5                               # Abbrev [5] 0x51:0x13 DW_TAG_enumeration_type
	.long	75                              # DW_AT_type
	.byte	12                              # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.byte	6                               # Abbrev [6] 0x5a:0x3 DW_TAG_enumerator
	.byte	9                               # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	6                               # Abbrev [6] 0x5d:0x3 DW_TAG_enumerator
	.byte	10                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	6                               # Abbrev [6] 0x60:0x3 DW_TAG_enumerator
	.byte	11                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0x64:0x13 DW_TAG_enumeration_type
	.long	54                              # DW_AT_type
                                        # DW_AT_enum_class
	.byte	13                              # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.byte	10                              # Abbrev [10] 0x6d:0x3 DW_TAG_enumerator
	.byte	9                               # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	10                              # Abbrev [10] 0x70:0x3 DW_TAG_enumerator
	.byte	10                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	10                              # Abbrev [10] 0x73:0x3 DW_TAG_enumerator
	.byte	11                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x77:0xe DW_TAG_enumeration_type
	.long	178                             # DW_AT_type
	.byte	16                              # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	30                              # DW_AT_decl_line
	.byte	6                               # Abbrev [6] 0x80:0x4 DW_TAG_enumerator
	.byte	15                              # DW_AT_name
	.ascii	"\377\001"                      # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	11                              # Abbrev [11] 0x85:0x12 DW_TAG_enumeration_type
	.long	75                              # DW_AT_type
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	29                              # DW_AT_decl_line
	.byte	6                               # Abbrev [6] 0x8d:0x3 DW_TAG_enumerator
	.byte	17                              # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	6                               # Abbrev [6] 0x90:0x3 DW_TAG_enumerator
	.byte	18                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	6                               # Abbrev [6] 0x93:0x3 DW_TAG_enumerator
	.byte	19                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x97:0x13 DW_TAG_subprogram
	.byte	101                             # DW_AT_low_pc
	.long	.Lfunc_end100-.Lfunc_begin100   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	308                             # DW_AT_linkage_name
	.short	309                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	13                              # Abbrev [13] 0xa5:0x4 DW_TAG_GNU_template_template_param
	.byte	20                              # DW_AT_name
	.short	307                             # DW_AT_GNU_template_name
	.byte	0                               # End Of Children Mark
	.byte	7                               # Abbrev [7] 0xaa:0x2 DW_TAG_structure_type
	.byte	158                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	8                               # Abbrev [8] 0xac:0x5 DW_TAG_namespace
	.byte	165                             # DW_AT_name
	.byte	7                               # Abbrev [7] 0xae:0x2 DW_TAG_structure_type
	.byte	158                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0xb2:0x4 DW_TAG_base_type
	.byte	14                              # DW_AT_name
	.byte	8                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	14                              # Abbrev [14] 0xb6:0x14 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	23                              # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	48                              # DW_AT_decl_line
	.byte	15                              # Abbrev [15] 0xbc:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	16                              # Abbrev [16] 0xc2:0x7 DW_TAG_template_value_parameter
	.long	202                             # DW_AT_type
	.byte	22                              # DW_AT_name
                                        # DW_AT_default_value
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0xca:0x4 DW_TAG_base_type
	.byte	21                              # DW_AT_name
	.byte	2                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	14                              # Abbrev [14] 0xce:0x13 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	24                              # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	165                             # DW_AT_decl_line
	.byte	17                              # Abbrev [17] 0xd4:0xc DW_TAG_subprogram
	.byte	130                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	167                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	18                              # Abbrev [18] 0xd8:0x2 DW_TAG_template_type_parameter
	.byte	20                              # DW_AT_name
                                        # DW_AT_default_value
	.byte	19                              # Abbrev [19] 0xda:0x5 DW_TAG_formal_parameter
	.long	5915                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	8                               # Abbrev [8] 0xe1:0xd0 DW_TAG_namespace
	.byte	25                              # DW_AT_name
	.byte	20                              # Abbrev [20] 0xe3:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	47                              # DW_AT_decl_line
	.long	433                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0xea:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	48                              # DW_AT_decl_line
	.long	453                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0xf1:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	49                              # DW_AT_decl_line
	.long	473                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0xf8:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	50                              # DW_AT_decl_line
	.long	489                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0xff:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	52                              # DW_AT_decl_line
	.long	509                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x106:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	53                              # DW_AT_decl_line
	.long	517                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x10d:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	54                              # DW_AT_decl_line
	.long	525                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x114:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	55                              # DW_AT_decl_line
	.long	533                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x11b:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	57                              # DW_AT_decl_line
	.long	541                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x122:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.long	557                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x129:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	59                              # DW_AT_decl_line
	.long	573                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x130:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	60                              # DW_AT_decl_line
	.long	589                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x137:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	62                              # DW_AT_decl_line
	.long	605                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x13e:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	63                              # DW_AT_decl_line
	.long	621                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x145:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	65                              # DW_AT_decl_line
	.long	629                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x14c:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	66                              # DW_AT_decl_line
	.long	645                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x153:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	67                              # DW_AT_decl_line
	.long	665                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x15a:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	68                              # DW_AT_decl_line
	.long	681                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x161:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	70                              # DW_AT_decl_line
	.long	701                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x168:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	71                              # DW_AT_decl_line
	.long	709                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x16f:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	72                              # DW_AT_decl_line
	.long	717                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x176:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
	.long	725                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x17d:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	75                              # DW_AT_decl_line
	.long	733                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x184:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	76                              # DW_AT_decl_line
	.long	749                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x18b:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	77                              # DW_AT_decl_line
	.long	765                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x192:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	78                              # DW_AT_decl_line
	.long	781                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x199:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	80                              # DW_AT_decl_line
	.long	797                             # DW_AT_import
	.byte	20                              # Abbrev [20] 0x1a0:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	81                              # DW_AT_decl_line
	.long	813                             # DW_AT_import
	.byte	21                              # Abbrev [21] 0x1a7:0x9 DW_TAG_typedef
	.long	697                             # DW_AT_type
	.byte	121                             # DW_AT_name
	.byte	6                               # DW_AT_decl_file
	.short	260                             # DW_AT_decl_line
	.byte	0                               # End Of Children Mark
	.byte	22                              # Abbrev [22] 0x1b1:0x8 DW_TAG_typedef
	.long	441                             # DW_AT_type
	.byte	28                              # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	24                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x1b9:0x8 DW_TAG_typedef
	.long	449                             # DW_AT_type
	.byte	27                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x1c1:0x4 DW_TAG_base_type
	.byte	26                              # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	22                              # Abbrev [22] 0x1c5:0x8 DW_TAG_typedef
	.long	461                             # DW_AT_type
	.byte	31                              # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	25                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x1cd:0x8 DW_TAG_typedef
	.long	469                             # DW_AT_type
	.byte	30                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	39                              # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x1d5:0x4 DW_TAG_base_type
	.byte	29                              # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	2                               # DW_AT_byte_size
	.byte	22                              # Abbrev [22] 0x1d9:0x8 DW_TAG_typedef
	.long	481                             # DW_AT_type
	.byte	33                              # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x1e1:0x8 DW_TAG_typedef
	.long	54                              # DW_AT_type
	.byte	32                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x1e9:0x8 DW_TAG_typedef
	.long	497                             # DW_AT_type
	.byte	36                              # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x1f1:0x8 DW_TAG_typedef
	.long	505                             # DW_AT_type
	.byte	35                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	44                              # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x1f9:0x4 DW_TAG_base_type
	.byte	34                              # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	22                              # Abbrev [22] 0x1fd:0x8 DW_TAG_typedef
	.long	449                             # DW_AT_type
	.byte	37                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x205:0x8 DW_TAG_typedef
	.long	505                             # DW_AT_type
	.byte	38                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	60                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x20d:0x8 DW_TAG_typedef
	.long	505                             # DW_AT_type
	.byte	39                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	61                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x215:0x8 DW_TAG_typedef
	.long	505                             # DW_AT_type
	.byte	40                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	62                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x21d:0x8 DW_TAG_typedef
	.long	549                             # DW_AT_type
	.byte	42                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x225:0x8 DW_TAG_typedef
	.long	441                             # DW_AT_type
	.byte	41                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	52                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x22d:0x8 DW_TAG_typedef
	.long	565                             # DW_AT_type
	.byte	44                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	44                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x235:0x8 DW_TAG_typedef
	.long	461                             # DW_AT_type
	.byte	43                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	54                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x23d:0x8 DW_TAG_typedef
	.long	581                             # DW_AT_type
	.byte	46                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	45                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x245:0x8 DW_TAG_typedef
	.long	481                             # DW_AT_type
	.byte	45                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	56                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x24d:0x8 DW_TAG_typedef
	.long	597                             # DW_AT_type
	.byte	48                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	46                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x255:0x8 DW_TAG_typedef
	.long	497                             # DW_AT_type
	.byte	47                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x25d:0x8 DW_TAG_typedef
	.long	613                             # DW_AT_type
	.byte	50                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	101                             # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x265:0x8 DW_TAG_typedef
	.long	505                             # DW_AT_type
	.byte	49                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	72                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x26d:0x8 DW_TAG_typedef
	.long	505                             # DW_AT_type
	.byte	51                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	87                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x275:0x8 DW_TAG_typedef
	.long	637                             # DW_AT_type
	.byte	53                              # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	24                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x27d:0x8 DW_TAG_typedef
	.long	178                             # DW_AT_type
	.byte	52                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	38                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x285:0x8 DW_TAG_typedef
	.long	653                             # DW_AT_type
	.byte	56                              # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	25                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x28d:0x8 DW_TAG_typedef
	.long	661                             # DW_AT_type
	.byte	55                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	40                              # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x295:0x4 DW_TAG_base_type
	.byte	54                              # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	2                               # DW_AT_byte_size
	.byte	22                              # Abbrev [22] 0x299:0x8 DW_TAG_typedef
	.long	673                             # DW_AT_type
	.byte	58                              # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x2a1:0x8 DW_TAG_typedef
	.long	75                              # DW_AT_type
	.byte	57                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	42                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x2a9:0x8 DW_TAG_typedef
	.long	689                             # DW_AT_type
	.byte	61                              # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x2b1:0x8 DW_TAG_typedef
	.long	697                             # DW_AT_type
	.byte	60                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	45                              # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x2b9:0x4 DW_TAG_base_type
	.byte	59                              # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	22                              # Abbrev [22] 0x2bd:0x8 DW_TAG_typedef
	.long	178                             # DW_AT_type
	.byte	62                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	71                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x2c5:0x8 DW_TAG_typedef
	.long	697                             # DW_AT_type
	.byte	63                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x2cd:0x8 DW_TAG_typedef
	.long	697                             # DW_AT_type
	.byte	64                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	74                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x2d5:0x8 DW_TAG_typedef
	.long	697                             # DW_AT_type
	.byte	65                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	75                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x2dd:0x8 DW_TAG_typedef
	.long	741                             # DW_AT_type
	.byte	67                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	49                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x2e5:0x8 DW_TAG_typedef
	.long	637                             # DW_AT_type
	.byte	66                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	53                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x2ed:0x8 DW_TAG_typedef
	.long	757                             # DW_AT_type
	.byte	69                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	50                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x2f5:0x8 DW_TAG_typedef
	.long	653                             # DW_AT_type
	.byte	68                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	55                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x2fd:0x8 DW_TAG_typedef
	.long	773                             # DW_AT_type
	.byte	71                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	51                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x305:0x8 DW_TAG_typedef
	.long	673                             # DW_AT_type
	.byte	70                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	57                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x30d:0x8 DW_TAG_typedef
	.long	789                             # DW_AT_type
	.byte	73                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	52                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x315:0x8 DW_TAG_typedef
	.long	689                             # DW_AT_type
	.byte	72                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	59                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x31d:0x8 DW_TAG_typedef
	.long	805                             # DW_AT_type
	.byte	75                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	102                             # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x325:0x8 DW_TAG_typedef
	.long	697                             # DW_AT_type
	.byte	74                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x32d:0x8 DW_TAG_typedef
	.long	697                             # DW_AT_type
	.byte	76                              # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	90                              # DW_AT_decl_line
	.byte	23                              # Abbrev [23] 0x335:0x17 DW_TAG_subprogram
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	134                             # DW_AT_linkage_name
	.byte	135                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	142                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x341:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	0                               # DW_AT_decl_file
	.byte	142                             # DW_AT_decl_line
	.long	7147                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	25                              # Abbrev [25] 0x34c:0x8a DW_TAG_subprogram
	.byte	2                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	136                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	182                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
                                        # DW_AT_external
	.byte	26                              # Abbrev [26] 0x35b:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	194                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	183                             # DW_AT_decl_line
	.long	971                             # DW_AT_type
	.byte	27                              # Abbrev [27] 0x366:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	384                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	184                             # DW_AT_decl_line
	.long	966                             # DW_AT_type
	.byte	27                              # Abbrev [27] 0x372:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	104
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	216                             # DW_AT_decl_line
	.long	7787                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x37e:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	96
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	231                             # DW_AT_decl_line
	.long	7262                            # DW_AT_type
	.byte	28                              # Abbrev [28] 0x38a:0xd DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	88
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.short	256                             # DW_AT_decl_line
	.long	3851                            # DW_AT_type
	.byte	28                              # Abbrev [28] 0x397:0xd DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	80
	.short	390                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.short	291                             # DW_AT_decl_line
	.long	7803                            # DW_AT_type
	.byte	28                              # Abbrev [28] 0x3a4:0xd DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	72
	.short	392                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.short	300                             # DW_AT_decl_line
	.long	206                             # DW_AT_type
	.byte	28                              # Abbrev [28] 0x3b1:0xd DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	64
	.short	393                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.short	321                             # DW_AT_decl_line
	.long	7812                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x3be:0x8 DW_TAG_imported_module
	.byte	0                               # DW_AT_decl_file
	.short	288                             # DW_AT_decl_line
	.long	79                              # DW_AT_import
	.byte	30                              # Abbrev [30] 0x3c6:0x5 DW_TAG_class_type
	.byte	5                               # DW_AT_calling_convention
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	184                             # DW_AT_decl_line
	.byte	31                              # Abbrev [31] 0x3cb:0x5 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	183                             # DW_AT_decl_line
	.byte	32                              # Abbrev [32] 0x3d0:0x3 DW_TAG_structure_type
	.short	295                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	7                               # Abbrev [7] 0x3d3:0x2 DW_TAG_structure_type
	.byte	133                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x3d6:0x2d DW_TAG_subprogram
	.byte	3                               # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	137                             # DW_AT_linkage_name
	.byte	138                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x3e2:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	7162                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x3ee:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	7853                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x3fa:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x3fc:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x403:0x2d DW_TAG_subprogram
	.byte	4                               # DW_AT_low_pc
	.long	.Lfunc_end3-.Lfunc_begin3       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	139                             # DW_AT_linkage_name
	.byte	140                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x40f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	4516                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x41b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	7870                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x427:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x429:0x5 DW_TAG_template_type_parameter
	.long	4531                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x430:0x2d DW_TAG_subprogram
	.byte	5                               # DW_AT_low_pc
	.long	.Lfunc_end4-.Lfunc_begin4       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	141                             # DW_AT_linkage_name
	.byte	142                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x43c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	7887                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x448:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	7903                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x454:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x456:0x5 DW_TAG_template_type_parameter
	.long	202                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x45d:0x2d DW_TAG_subprogram
	.byte	6                               # DW_AT_low_pc
	.long	.Lfunc_end5-.Lfunc_begin5       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	144                             # DW_AT_linkage_name
	.byte	145                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x469:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	7920                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x475:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	7936                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x481:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x483:0x5 DW_TAG_template_type_parameter
	.long	7143                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x48a:0x2d DW_TAG_subprogram
	.byte	7                               # DW_AT_low_pc
	.long	.Lfunc_end6-.Lfunc_begin6       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	146                             # DW_AT_linkage_name
	.byte	147                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x496:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	7953                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x4a2:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	7969                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x4ae:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x4b0:0x5 DW_TAG_template_type_parameter
	.long	505                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x4b7:0x2d DW_TAG_subprogram
	.byte	8                               # DW_AT_low_pc
	.long	.Lfunc_end7-.Lfunc_begin7       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	148                             # DW_AT_linkage_name
	.byte	149                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x4c3:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	7986                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x4cf:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8002                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x4db:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x4dd:0x5 DW_TAG_template_type_parameter
	.long	469                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x4e4:0x2d DW_TAG_subprogram
	.byte	9                               # DW_AT_low_pc
	.long	.Lfunc_end8-.Lfunc_begin8       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	150                             # DW_AT_linkage_name
	.byte	151                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x4f0:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8019                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x4fc:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8035                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x508:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x50a:0x5 DW_TAG_template_type_parameter
	.long	75                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x511:0x2d DW_TAG_subprogram
	.byte	10                              # DW_AT_low_pc
	.long	.Lfunc_end9-.Lfunc_begin9       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	153                             # DW_AT_linkage_name
	.byte	154                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x51d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8052                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x529:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8068                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x535:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x537:0x5 DW_TAG_template_type_parameter
	.long	7147                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x53e:0x2d DW_TAG_subprogram
	.byte	11                              # DW_AT_low_pc
	.long	.Lfunc_end10-.Lfunc_begin10     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	156                             # DW_AT_linkage_name
	.byte	157                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x54a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8085                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x556:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8101                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x562:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x564:0x5 DW_TAG_template_type_parameter
	.long	7151                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x56b:0x2d DW_TAG_subprogram
	.byte	12                              # DW_AT_low_pc
	.long	.Lfunc_end11-.Lfunc_begin11     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	159                             # DW_AT_linkage_name
	.byte	160                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x577:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8118                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x583:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8134                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x58f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x591:0x5 DW_TAG_template_type_parameter
	.long	7155                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x598:0x2d DW_TAG_subprogram
	.byte	13                              # DW_AT_low_pc
	.long	.Lfunc_end12-.Lfunc_begin12     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	161                             # DW_AT_linkage_name
	.byte	162                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x5a4:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8151                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x5b0:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8167                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x5bc:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x5be:0x5 DW_TAG_template_type_parameter
	.long	170                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x5c5:0x2d DW_TAG_subprogram
	.byte	14                              # DW_AT_low_pc
	.long	.Lfunc_end13-.Lfunc_begin13     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	163                             # DW_AT_linkage_name
	.byte	164                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x5d1:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8184                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x5dd:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8200                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x5e9:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x5eb:0x5 DW_TAG_template_type_parameter
	.long	7157                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x5f2:0x2d DW_TAG_subprogram
	.byte	15                              # DW_AT_low_pc
	.long	.Lfunc_end14-.Lfunc_begin14     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	166                             # DW_AT_linkage_name
	.byte	167                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x5fe:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8217                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x60a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8233                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x616:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x618:0x5 DW_TAG_template_type_parameter
	.long	174                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x61f:0x2d DW_TAG_subprogram
	.byte	16                              # DW_AT_low_pc
	.long	.Lfunc_end15-.Lfunc_begin15     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	169                             # DW_AT_linkage_name
	.byte	170                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x62b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8250                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x637:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8266                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x643:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x645:0x5 DW_TAG_template_type_parameter
	.long	7162                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x64c:0x32 DW_TAG_subprogram
	.byte	17                              # DW_AT_low_pc
	.long	.Lfunc_end16-.Lfunc_begin16     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	171                             # DW_AT_linkage_name
	.byte	172                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x658:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8283                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x664:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8304                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x670:0xd DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x672:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	34                              # Abbrev [34] 0x677:0x5 DW_TAG_template_type_parameter
	.long	4531                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x67e:0x2d DW_TAG_subprogram
	.byte	18                              # DW_AT_low_pc
	.long	.Lfunc_end17-.Lfunc_begin17     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	173                             # DW_AT_linkage_name
	.byte	174                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x68a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	7374                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x696:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8326                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x6a2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x6a4:0x5 DW_TAG_template_type_parameter
	.long	7177                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x6ab:0x2d DW_TAG_subprogram
	.byte	19                              # DW_AT_low_pc
	.long	.Lfunc_end18-.Lfunc_begin18     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	175                             # DW_AT_linkage_name
	.byte	176                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x6b7:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8343                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x6c3:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8359                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x6cf:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x6d1:0x5 DW_TAG_template_type_parameter
	.long	7182                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x6d8:0x2d DW_TAG_subprogram
	.byte	20                              # DW_AT_low_pc
	.long	.Lfunc_end19-.Lfunc_begin19     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	177                             # DW_AT_linkage_name
	.byte	178                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x6e4:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8376                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x6f0:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8392                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x6fc:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x6fe:0x5 DW_TAG_template_type_parameter
	.long	7187                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x705:0x2d DW_TAG_subprogram
	.byte	21                              # DW_AT_low_pc
	.long	.Lfunc_end20-.Lfunc_begin20     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	179                             # DW_AT_linkage_name
	.byte	180                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x711:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8409                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x71d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8425                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x729:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x72b:0x5 DW_TAG_template_type_parameter
	.long	7192                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x732:0x2d DW_TAG_subprogram
	.byte	22                              # DW_AT_low_pc
	.long	.Lfunc_end21-.Lfunc_begin21     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	182                             # DW_AT_linkage_name
	.byte	183                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x73e:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8442                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x74a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8458                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x756:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x758:0x5 DW_TAG_template_type_parameter
	.long	7197                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x75f:0x29 DW_TAG_subprogram
	.byte	23                              # DW_AT_low_pc
	.long	.Lfunc_end22-.Lfunc_begin22     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	184                             # DW_AT_linkage_name
	.byte	185                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x76b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8475                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x777:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8487                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x783:0x4 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	35                              # Abbrev [35] 0x785:0x1 DW_TAG_template_type_parameter
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x788:0x2d DW_TAG_subprogram
	.byte	24                              # DW_AT_low_pc
	.long	.Lfunc_end23-.Lfunc_begin23     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	188                             # DW_AT_linkage_name
	.byte	189                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x794:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8500                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x7a0:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8516                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x7ac:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x7ae:0x5 DW_TAG_template_type_parameter
	.long	7219                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x7b5:0x2d DW_TAG_subprogram
	.byte	25                              # DW_AT_low_pc
	.long	.Lfunc_end24-.Lfunc_begin24     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	190                             # DW_AT_linkage_name
	.byte	191                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x7c1:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8533                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x7cd:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8549                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x7d9:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x7db:0x5 DW_TAG_template_type_parameter
	.long	697                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x7e2:0x1b DW_TAG_subprogram
	.byte	26                              # DW_AT_low_pc
	.long	.Lfunc_end25-.Lfunc_begin25     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	192                             # DW_AT_linkage_name
	.byte	193                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	38                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	36                              # Abbrev [36] 0x7ee:0x7 DW_TAG_template_value_parameter
	.long	202                             # DW_AT_type
	.byte	22                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	37                              # Abbrev [37] 0x7f5:0x7 DW_TAG_template_value_parameter
	.long	54                              # DW_AT_type
	.byte	3                               # DW_AT_name
	.byte	3                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x7fd:0x22 DW_TAG_subprogram
	.byte	27                              # DW_AT_low_pc
	.long	.Lfunc_end26-.Lfunc_begin26     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	195                             # DW_AT_linkage_name
	.byte	196                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x809:0x6 DW_TAG_template_type_parameter
	.long	81                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0x80f:0xf DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	38                              # Abbrev [38] 0x811:0x6 DW_TAG_template_value_parameter
	.long	81                              # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	38                              # Abbrev [38] 0x817:0x6 DW_TAG_template_value_parameter
	.long	81                              # DW_AT_type
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x81f:0x22 DW_TAG_subprogram
	.byte	28                              # DW_AT_low_pc
	.long	.Lfunc_end27-.Lfunc_begin27     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	197                             # DW_AT_linkage_name
	.byte	198                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x82b:0x6 DW_TAG_template_type_parameter
	.long	100                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0x831:0xf DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	39                              # Abbrev [39] 0x833:0x6 DW_TAG_template_value_parameter
	.long	100                             # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	39                              # Abbrev [39] 0x839:0x6 DW_TAG_template_value_parameter
	.long	100                             # DW_AT_type
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x841:0x1d DW_TAG_subprogram
	.byte	29                              # DW_AT_low_pc
	.long	.Lfunc_end28-.Lfunc_begin28     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	199                             # DW_AT_linkage_name
	.byte	200                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x84d:0x6 DW_TAG_template_type_parameter
	.long	119                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0x853:0xa DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	38                              # Abbrev [38] 0x855:0x7 DW_TAG_template_value_parameter
	.long	119                             # DW_AT_type
	.ascii	"\377\001"                      # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	40                              # Abbrev [40] 0x85e:0x22 DW_TAG_subprogram
	.byte	30                              # DW_AT_low_pc
	.long	.Lfunc_end29-.Lfunc_begin29     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	201                             # DW_AT_linkage_name
	.byte	202                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
	.byte	15                              # Abbrev [15] 0x86a:0x6 DW_TAG_template_type_parameter
	.long	133                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0x870:0xf DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	38                              # Abbrev [38] 0x872:0x6 DW_TAG_template_value_parameter
	.long	133                             # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	38                              # Abbrev [38] 0x878:0x6 DW_TAG_template_value_parameter
	.long	133                             # DW_AT_type
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	40                              # Abbrev [40] 0x880:0x1c DW_TAG_subprogram
	.byte	31                              # DW_AT_low_pc
	.long	.Lfunc_end30-.Lfunc_begin30     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	203                             # DW_AT_linkage_name
	.byte	204                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
	.byte	15                              # Abbrev [15] 0x88c:0x6 DW_TAG_template_type_parameter
	.long	59                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0x892:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	38                              # Abbrev [38] 0x894:0x6 DW_TAG_template_value_parameter
	.long	59                              # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x89c:0x1f DW_TAG_subprogram
	.byte	32                              # DW_AT_low_pc
	.long	.Lfunc_end31-.Lfunc_begin31     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	205                             # DW_AT_linkage_name
	.byte	206                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x8a8:0x6 DW_TAG_template_type_parameter
	.long	7177                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0x8ae:0xc DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	41                              # Abbrev [41] 0x8b0:0x9 DW_TAG_template_value_parameter
	.long	7177                            # DW_AT_type
	.byte	3                               # DW_AT_location
	.byte	161
	.byte	0
	.byte	159
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x8bb:0x1c DW_TAG_subprogram
	.byte	33                              # DW_AT_low_pc
	.long	.Lfunc_end32-.Lfunc_begin32     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	207                             # DW_AT_linkage_name
	.byte	208                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x8c7:0x6 DW_TAG_template_type_parameter
	.long	7177                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0x8cd:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	38                              # Abbrev [38] 0x8cf:0x6 DW_TAG_template_value_parameter
	.long	7177                            # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x8d7:0x1c DW_TAG_subprogram
	.byte	34                              # DW_AT_low_pc
	.long	.Lfunc_end33-.Lfunc_begin33     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	209                             # DW_AT_linkage_name
	.byte	210                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x8e3:0x6 DW_TAG_template_type_parameter
	.long	697                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0x8e9:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	38                              # Abbrev [38] 0x8eb:0x6 DW_TAG_template_value_parameter
	.long	697                             # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x8f3:0x1c DW_TAG_subprogram
	.byte	35                              # DW_AT_low_pc
	.long	.Lfunc_end34-.Lfunc_begin34     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	211                             # DW_AT_linkage_name
	.byte	212                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x8ff:0x6 DW_TAG_template_type_parameter
	.long	7147                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0x905:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	38                              # Abbrev [38] 0x907:0x6 DW_TAG_template_value_parameter
	.long	7147                            # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x90f:0x1c DW_TAG_subprogram
	.byte	36                              # DW_AT_low_pc
	.long	.Lfunc_end35-.Lfunc_begin35     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	213                             # DW_AT_linkage_name
	.byte	214                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x91b:0x6 DW_TAG_template_type_parameter
	.long	505                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0x921:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	39                              # Abbrev [39] 0x923:0x6 DW_TAG_template_value_parameter
	.long	505                             # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x92b:0x1c DW_TAG_subprogram
	.byte	37                              # DW_AT_low_pc
	.long	.Lfunc_end36-.Lfunc_begin36     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	215                             # DW_AT_linkage_name
	.byte	216                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x937:0x6 DW_TAG_template_type_parameter
	.long	75                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0x93d:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	38                              # Abbrev [38] 0x93f:0x6 DW_TAG_template_value_parameter
	.long	75                              # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x947:0x1c DW_TAG_subprogram
	.byte	38                              # DW_AT_low_pc
	.long	.Lfunc_end37-.Lfunc_begin37     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	217                             # DW_AT_linkage_name
	.byte	218                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x953:0x6 DW_TAG_template_type_parameter
	.long	469                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0x959:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	39                              # Abbrev [39] 0x95b:0x6 DW_TAG_template_value_parameter
	.long	469                             # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x963:0x1c DW_TAG_subprogram
	.byte	39                              # DW_AT_low_pc
	.long	.Lfunc_end38-.Lfunc_begin38     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	219                             # DW_AT_linkage_name
	.byte	220                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x96f:0x6 DW_TAG_template_type_parameter
	.long	178                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0x975:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	38                              # Abbrev [38] 0x977:0x6 DW_TAG_template_value_parameter
	.long	178                             # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x97f:0x1c DW_TAG_subprogram
	.byte	40                              # DW_AT_low_pc
	.long	.Lfunc_end39-.Lfunc_begin39     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	221                             # DW_AT_linkage_name
	.byte	222                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x98b:0x6 DW_TAG_template_type_parameter
	.long	449                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0x991:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	39                              # Abbrev [39] 0x993:0x6 DW_TAG_template_value_parameter
	.long	449                             # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x99b:0x22 DW_TAG_subprogram
	.byte	41                              # DW_AT_low_pc
	.long	.Lfunc_end40-.Lfunc_begin40     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	223                             # DW_AT_linkage_name
	.byte	224                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x9a7:0x6 DW_TAG_template_type_parameter
	.long	661                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0x9ad:0xf DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	38                              # Abbrev [38] 0x9af:0x6 DW_TAG_template_value_parameter
	.long	661                             # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	38                              # Abbrev [38] 0x9b5:0x6 DW_TAG_template_value_parameter
	.long	661                             # DW_AT_type
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x9bd:0x5a DW_TAG_subprogram
	.byte	42                              # DW_AT_low_pc
	.long	.Lfunc_end41-.Lfunc_begin41     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	226                             # DW_AT_linkage_name
	.byte	227                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x9c9:0x6 DW_TAG_template_type_parameter
	.long	7222                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0x9cf:0x47 DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	39                              # Abbrev [39] 0x9d1:0x6 DW_TAG_template_value_parameter
	.long	7222                            # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	39                              # Abbrev [39] 0x9d7:0x6 DW_TAG_template_value_parameter
	.long	7222                            # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	39                              # Abbrev [39] 0x9dd:0x6 DW_TAG_template_value_parameter
	.long	7222                            # DW_AT_type
	.byte	6                               # DW_AT_const_value
	.byte	39                              # Abbrev [39] 0x9e3:0x6 DW_TAG_template_value_parameter
	.long	7222                            # DW_AT_type
	.byte	7                               # DW_AT_const_value
	.byte	39                              # Abbrev [39] 0x9e9:0x6 DW_TAG_template_value_parameter
	.long	7222                            # DW_AT_type
	.byte	13                              # DW_AT_const_value
	.byte	39                              # Abbrev [39] 0x9ef:0x6 DW_TAG_template_value_parameter
	.long	7222                            # DW_AT_type
	.byte	14                              # DW_AT_const_value
	.byte	39                              # Abbrev [39] 0x9f5:0x6 DW_TAG_template_value_parameter
	.long	7222                            # DW_AT_type
	.byte	31                              # DW_AT_const_value
	.byte	39                              # Abbrev [39] 0x9fb:0x6 DW_TAG_template_value_parameter
	.long	7222                            # DW_AT_type
	.byte	32                              # DW_AT_const_value
	.byte	39                              # Abbrev [39] 0xa01:0x6 DW_TAG_template_value_parameter
	.long	7222                            # DW_AT_type
	.byte	33                              # DW_AT_const_value
	.byte	39                              # Abbrev [39] 0xa07:0x7 DW_TAG_template_value_parameter
	.long	7222                            # DW_AT_type
	.asciz	"\377"                          # DW_AT_const_value
	.byte	39                              # Abbrev [39] 0xa0e:0x7 DW_TAG_template_value_parameter
	.long	7222                            # DW_AT_type
	.ascii	"\200\177"                      # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0xa17:0x2c DW_TAG_subprogram
	.byte	43                              # DW_AT_low_pc
	.long	.Lfunc_end42-.Lfunc_begin42     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	229                             # DW_AT_linkage_name
	.byte	230                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xa23:0x6 DW_TAG_template_type_parameter
	.long	7226                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0xa29:0x19 DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	42                              # Abbrev [42] 0xa2b:0x16 DW_TAG_template_value_parameter
	.long	7226                            # DW_AT_type
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
	.byte	23                              # Abbrev [23] 0xa43:0x19 DW_TAG_subprogram
	.byte	44                              # DW_AT_low_pc
	.long	.Lfunc_end43-.Lfunc_begin43     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	231                             # DW_AT_linkage_name
	.byte	232                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	44                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xa4f:0x6 DW_TAG_template_type_parameter
	.long	75                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	43                              # Abbrev [43] 0xa55:0x6 DW_TAG_template_value_parameter
	.long	75                              # DW_AT_type
                                        # DW_AT_default_value
	.byte	3                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0xa5c:0x2d DW_TAG_subprogram
	.byte	45                              # DW_AT_low_pc
	.long	.Lfunc_end44-.Lfunc_begin44     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	233                             # DW_AT_linkage_name
	.byte	234                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xa68:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8566                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xa74:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8582                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xa80:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xa82:0x5 DW_TAG_template_type_parameter
	.long	182                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0xa89:0x2d DW_TAG_subprogram
	.byte	46                              # DW_AT_low_pc
	.long	.Lfunc_end45-.Lfunc_begin45     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	236                             # DW_AT_linkage_name
	.byte	237                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xa95:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8599                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xaa1:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8615                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xaad:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xaaf:0x5 DW_TAG_template_type_parameter
	.long	7230                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	40                              # Abbrev [40] 0xab6:0x2d DW_TAG_subprogram
	.byte	47                              # DW_AT_low_pc
	.long	.Lfunc_end46-.Lfunc_begin46     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	238                             # DW_AT_linkage_name
	.byte	239                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
	.byte	27                              # Abbrev [27] 0xac2:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	7468                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xace:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8632                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xada:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xadc:0x5 DW_TAG_template_type_parameter
	.long	966                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	40                              # Abbrev [40] 0xae3:0x2d DW_TAG_subprogram
	.byte	48                              # DW_AT_low_pc
	.long	.Lfunc_end47-.Lfunc_begin47     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	242                             # DW_AT_linkage_name
	.byte	243                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
	.byte	27                              # Abbrev [27] 0xaef:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8649                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xafb:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8665                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xb07:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xb09:0x5 DW_TAG_template_type_parameter
	.long	7246                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0xb10:0x2d DW_TAG_subprogram
	.byte	49                              # DW_AT_low_pc
	.long	.Lfunc_end48-.Lfunc_begin48     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	244                             # DW_AT_linkage_name
	.byte	245                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xb1c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8682                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xb28:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8698                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xb34:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xb36:0x5 DW_TAG_template_type_parameter
	.long	7282                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0xb3d:0x2d DW_TAG_subprogram
	.byte	50                              # DW_AT_low_pc
	.long	.Lfunc_end49-.Lfunc_begin49     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	246                             # DW_AT_linkage_name
	.byte	247                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xb49:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8715                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xb55:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8731                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xb61:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xb63:0x5 DW_TAG_template_type_parameter
	.long	7293                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0xb6a:0x2d DW_TAG_subprogram
	.byte	51                              # DW_AT_low_pc
	.long	.Lfunc_end50-.Lfunc_begin50     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	248                             # DW_AT_linkage_name
	.byte	249                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xb76:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8748                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xb82:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8764                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xb8e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xb90:0x5 DW_TAG_template_type_parameter
	.long	7296                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0xb97:0x2d DW_TAG_subprogram
	.byte	52                              # DW_AT_low_pc
	.long	.Lfunc_end51-.Lfunc_begin51     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	250                             # DW_AT_linkage_name
	.byte	251                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xba3:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8781                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xbaf:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8797                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xbbb:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xbbd:0x5 DW_TAG_template_type_parameter
	.long	7304                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0xbc4:0x2d DW_TAG_subprogram
	.byte	53                              # DW_AT_low_pc
	.long	.Lfunc_end52-.Lfunc_begin52     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	252                             # DW_AT_linkage_name
	.byte	253                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xbd0:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8814                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xbdc:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8830                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xbe8:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xbea:0x5 DW_TAG_template_type_parameter
	.long	7309                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	44                              # Abbrev [44] 0xbf1:0x2e DW_TAG_subprogram
	.byte	54                              # DW_AT_low_pc
	.long	.Lfunc_end53-.Lfunc_begin53     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	255                             # DW_AT_linkage_name
	.short	256                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
	.byte	27                              # Abbrev [27] 0xbfe:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8847                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xc0a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8863                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xc16:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xc18:0x5 DW_TAG_template_type_parameter
	.long	72                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xc1f:0x2f DW_TAG_subprogram
	.byte	55                              # DW_AT_low_pc
	.long	.Lfunc_end54-.Lfunc_begin54     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	258                             # DW_AT_linkage_name
	.short	259                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xc2d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8880                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xc39:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8896                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xc45:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xc47:0x5 DW_TAG_template_type_parameter
	.long	7319                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xc4e:0x34 DW_TAG_subprogram
	.byte	56                              # DW_AT_low_pc
	.long	.Lfunc_end55-.Lfunc_begin55     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	260                             # DW_AT_linkage_name
	.short	261                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xc5c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8913                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xc68:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8934                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xc74:0xd DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xc76:0x5 DW_TAG_template_type_parameter
	.long	7322                            # DW_AT_type
	.byte	34                              # Abbrev [34] 0xc7b:0x5 DW_TAG_template_type_parameter
	.long	7322                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xc82:0x34 DW_TAG_subprogram
	.byte	57                              # DW_AT_low_pc
	.long	.Lfunc_end56-.Lfunc_begin56     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	262                             # DW_AT_linkage_name
	.short	263                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xc90:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8956                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xc9c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	8977                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xca8:0xd DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xcaa:0x5 DW_TAG_template_type_parameter
	.long	7322                            # DW_AT_type
	.byte	34                              # Abbrev [34] 0xcaf:0x5 DW_TAG_template_type_parameter
	.long	7327                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xcb6:0x2f DW_TAG_subprogram
	.byte	58                              # DW_AT_low_pc
	.long	.Lfunc_end57-.Lfunc_begin57     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	264                             # DW_AT_linkage_name
	.short	265                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xcc4:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	8999                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xcd0:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9015                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xcdc:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xcde:0x5 DW_TAG_template_type_parameter
	.long	7332                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xce5:0x2f DW_TAG_subprogram
	.byte	59                              # DW_AT_low_pc
	.long	.Lfunc_end58-.Lfunc_begin58     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	266                             # DW_AT_linkage_name
	.short	267                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xcf3:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9032                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xcff:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9048                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xd0b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xd0d:0x5 DW_TAG_template_type_parameter
	.long	7337                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xd14:0x2f DW_TAG_subprogram
	.byte	60                              # DW_AT_low_pc
	.long	.Lfunc_end59-.Lfunc_begin59     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	268                             # DW_AT_linkage_name
	.short	269                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xd22:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9065                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xd2e:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9081                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xd3a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xd3c:0x5 DW_TAG_template_type_parameter
	.long	7353                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xd43:0x2f DW_TAG_subprogram
	.byte	61                              # DW_AT_low_pc
	.long	.Lfunc_end60-.Lfunc_begin60     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	270                             # DW_AT_linkage_name
	.short	271                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xd51:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9098                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xd5d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9114                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xd69:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xd6b:0x5 DW_TAG_template_type_parameter
	.long	7354                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	45                              # Abbrev [45] 0xd72:0x2f DW_TAG_subprogram
	.byte	62                              # DW_AT_low_pc
	.long	.Lfunc_end61-.Lfunc_begin61     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	272                             # DW_AT_linkage_name
	.short	273                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
	.byte	27                              # Abbrev [27] 0xd80:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9131                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xd8c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9147                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xd98:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xd9a:0x5 DW_TAG_template_type_parameter
	.long	7359                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	45                              # Abbrev [45] 0xda1:0x2f DW_TAG_subprogram
	.byte	63                              # DW_AT_low_pc
	.long	.Lfunc_end62-.Lfunc_begin62     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	274                             # DW_AT_linkage_name
	.short	275                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
	.byte	27                              # Abbrev [27] 0xdaf:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9164                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xdbb:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9180                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xdc7:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xdc9:0x5 DW_TAG_template_type_parameter
	.long	971                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	45                              # Abbrev [45] 0xdd0:0x2f DW_TAG_subprogram
	.byte	64                              # DW_AT_low_pc
	.long	.Lfunc_end63-.Lfunc_begin63     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	276                             # DW_AT_linkage_name
	.short	277                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
	.byte	27                              # Abbrev [27] 0xdde:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9197                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xdea:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9213                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xdf6:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xdf8:0x5 DW_TAG_template_type_parameter
	.long	7364                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xdff:0x1f DW_TAG_subprogram
	.byte	65                              # DW_AT_low_pc
	.long	.Lfunc_end64-.Lfunc_begin64     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	280                             # DW_AT_linkage_name
	.short	281                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	62                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	46                              # Abbrev [46] 0xe0d:0x9 DW_TAG_GNU_template_parameter_pack
	.short	278                             # DW_AT_name
	.byte	34                              # Abbrev [34] 0xe10:0x5 DW_TAG_template_type_parameter
	.long	7162                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0xe16:0x7 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.short	279                             # DW_AT_name
                                        # DW_AT_default_value
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xe1e:0x19 DW_TAG_subprogram
	.byte	66                              # DW_AT_low_pc
	.long	.Lfunc_end65-.Lfunc_begin65     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	282                             # DW_AT_linkage_name
	.short	283                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	62                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	48                              # Abbrev [48] 0xe2c:0x3 DW_TAG_GNU_template_parameter_pack
	.short	278                             # DW_AT_name
	.byte	47                              # Abbrev [47] 0xe2f:0x7 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.short	279                             # DW_AT_name
                                        # DW_AT_default_value
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xe37:0x19 DW_TAG_subprogram
	.byte	67                              # DW_AT_low_pc
	.long	.Lfunc_end66-.Lfunc_begin66     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	284                             # DW_AT_linkage_name
	.short	285                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	64                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	49                              # Abbrev [49] 0xe45:0x7 DW_TAG_template_type_parameter
	.long	7162                            # DW_AT_type
	.short	278                             # DW_AT_name
	.byte	48                              # Abbrev [48] 0xe4c:0x3 DW_TAG_GNU_template_parameter_pack
	.short	279                             # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xe50:0x29 DW_TAG_subprogram
	.byte	68                              # DW_AT_low_pc
	.long	.Lfunc_end67-.Lfunc_begin67     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	286                             # DW_AT_linkage_name
	.short	287                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xe5e:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	7752                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xe6a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9230                            # DW_AT_type
	.byte	50                              # Abbrev [50] 0xe76:0x2 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xe79:0x34 DW_TAG_subprogram
	.byte	69                              # DW_AT_low_pc
	.long	.Lfunc_end68-.Lfunc_begin68     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	288                             # DW_AT_linkage_name
	.short	289                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xe87:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9241                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xe93:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9262                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xe9f:0xd DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xea1:0x5 DW_TAG_template_type_parameter
	.long	7347                            # DW_AT_type
	.byte	34                              # Abbrev [34] 0xea6:0x5 DW_TAG_template_type_parameter
	.long	7347                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xead:0x2f DW_TAG_subprogram
	.byte	70                              # DW_AT_low_pc
	.long	.Lfunc_end69-.Lfunc_begin69     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	291                             # DW_AT_linkage_name
	.short	292                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xebb:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9284                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xec7:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9300                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xed3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xed5:0x5 DW_TAG_template_type_parameter
	.long	7369                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xedc:0x2f DW_TAG_subprogram
	.byte	71                              # DW_AT_low_pc
	.long	.Lfunc_end70-.Lfunc_begin70     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	293                             # DW_AT_linkage_name
	.short	294                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0xeea:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9317                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0xef6:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9333                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0xf02:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0xf04:0x5 DW_TAG_template_type_parameter
	.long	7390                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	14                              # Abbrev [14] 0xf0b:0x20e DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	77                              # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	65                              # DW_AT_decl_line
	.byte	51                              # Abbrev [51] 0xf11:0x16 DW_TAG_subprogram
	.byte	78                              # DW_AT_linkage_name
	.byte	79                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	67                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xf16:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xf1c:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	52                              # Abbrev [52] 0xf21:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0xf27:0x16 DW_TAG_subprogram
	.byte	80                              # DW_AT_linkage_name
	.byte	81                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	70                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xf2c:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xf32:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	52                              # Abbrev [52] 0xf37:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0xf3d:0x16 DW_TAG_subprogram
	.byte	82                              # DW_AT_linkage_name
	.byte	83                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xf42:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xf48:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	52                              # Abbrev [52] 0xf4d:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	53                              # Abbrev [53] 0xf53:0x15 DW_TAG_subprogram
	.byte	84                              # DW_AT_linkage_name
	.byte	85                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	76                              # DW_AT_decl_line
	.long	4511                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	54                              # Abbrev [54] 0xf5c:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
                                        # DW_AT_default_value
	.byte	19                              # Abbrev [19] 0xf62:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0xf68:0x16 DW_TAG_subprogram
	.byte	89                              # DW_AT_linkage_name
	.byte	90                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	80                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xf6d:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xf73:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	52                              # Abbrev [52] 0xf78:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0xf7e:0x16 DW_TAG_subprogram
	.byte	91                              # DW_AT_linkage_name
	.byte	92                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	83                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xf83:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xf89:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	52                              # Abbrev [52] 0xf8e:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0xf94:0x16 DW_TAG_subprogram
	.byte	93                              # DW_AT_linkage_name
	.byte	94                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	86                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xf99:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xf9f:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	52                              # Abbrev [52] 0xfa4:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0xfaa:0x16 DW_TAG_subprogram
	.byte	95                              # DW_AT_linkage_name
	.byte	96                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	89                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xfaf:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xfb5:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	52                              # Abbrev [52] 0xfba:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0xfc0:0x16 DW_TAG_subprogram
	.byte	97                              # DW_AT_linkage_name
	.byte	98                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	92                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xfc5:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xfcb:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	52                              # Abbrev [52] 0xfd0:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0xfd6:0x16 DW_TAG_subprogram
	.byte	99                              # DW_AT_linkage_name
	.byte	100                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	95                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xfdb:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xfe1:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	52                              # Abbrev [52] 0xfe6:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0xfec:0x16 DW_TAG_subprogram
	.byte	101                             # DW_AT_linkage_name
	.byte	102                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	98                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xff1:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xff7:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	52                              # Abbrev [52] 0xffc:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0x1002:0x11 DW_TAG_subprogram
	.byte	103                             # DW_AT_linkage_name
	.byte	104                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	101                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x1007:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0x100d:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0x1013:0x11 DW_TAG_subprogram
	.byte	105                             # DW_AT_linkage_name
	.byte	106                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	104                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x1018:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0x101e:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0x1024:0x16 DW_TAG_subprogram
	.byte	107                             # DW_AT_linkage_name
	.byte	108                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	107                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x1029:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0x102f:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	52                              # Abbrev [52] 0x1034:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0x103a:0x16 DW_TAG_subprogram
	.byte	109                             # DW_AT_linkage_name
	.byte	110                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	110                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x103f:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0x1045:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	52                              # Abbrev [52] 0x104a:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0x1050:0x16 DW_TAG_subprogram
	.byte	111                             # DW_AT_linkage_name
	.byte	112                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	113                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x1055:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0x105b:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	52                              # Abbrev [52] 0x1060:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0x1066:0x11 DW_TAG_subprogram
	.byte	113                             # DW_AT_linkage_name
	.byte	114                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	116                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x106b:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0x1071:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0x1077:0x16 DW_TAG_subprogram
	.byte	115                             # DW_AT_linkage_name
	.byte	116                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	119                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x107c:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0x1082:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	52                              # Abbrev [52] 0x1087:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0x108d:0x16 DW_TAG_subprogram
	.byte	117                             # DW_AT_linkage_name
	.byte	118                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	122                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x1092:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0x1098:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	52                              # Abbrev [52] 0x109d:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	53                              # Abbrev [53] 0x10a3:0x1a DW_TAG_subprogram
	.byte	119                             # DW_AT_linkage_name
	.byte	120                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	125                             # DW_AT_decl_line
	.long	5183                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x10ac:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	52                              # Abbrev [52] 0x10b2:0x5 DW_TAG_formal_parameter
	.long	423                             # DW_AT_type
	.byte	52                              # Abbrev [52] 0x10b7:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	53                              # Abbrev [53] 0x10bd:0x1a DW_TAG_subprogram
	.byte	122                             # DW_AT_linkage_name
	.byte	123                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	132                             # DW_AT_decl_line
	.long	5183                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x10c6:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	52                              # Abbrev [52] 0x10cc:0x5 DW_TAG_formal_parameter
	.long	423                             # DW_AT_type
	.byte	52                              # Abbrev [52] 0x10d1:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0x10d7:0x16 DW_TAG_subprogram
	.byte	124                             # DW_AT_linkage_name
	.byte	125                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	129                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x10dc:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	52                              # Abbrev [52] 0x10e2:0x5 DW_TAG_formal_parameter
	.long	5183                            # DW_AT_type
	.byte	52                              # Abbrev [52] 0x10e7:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	51                              # Abbrev [51] 0x10ed:0x16 DW_TAG_subprogram
	.byte	126                             # DW_AT_linkage_name
	.byte	127                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	136                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x10f2:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	52                              # Abbrev [52] 0x10f8:0x5 DW_TAG_formal_parameter
	.long	5183                            # DW_AT_type
	.byte	52                              # Abbrev [52] 0x10fd:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	53                              # Abbrev [53] 0x1103:0x15 DW_TAG_subprogram
	.byte	128                             # DW_AT_linkage_name
	.byte	129                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	139                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x110c:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0x1112:0x5 DW_TAG_formal_parameter
	.long	4377                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1119:0x5 DW_TAG_pointer_type
	.long	3851                            # DW_AT_type
	.byte	56                              # Abbrev [56] 0x111e:0x2b DW_TAG_subprogram
	.byte	72                              # DW_AT_low_pc
	.long	.Lfunc_end71-.Lfunc_begin71     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4398                            # DW_AT_object_pointer
	.long	3857                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x112e:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	24                              # Abbrev [24] 0x1138:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	67                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x1142:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x1149:0x2b DW_TAG_subprogram
	.byte	73                              # DW_AT_low_pc
	.long	.Lfunc_end72-.Lfunc_begin72     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4441                            # DW_AT_object_pointer
	.long	3879                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x1159:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	24                              # Abbrev [24] 0x1163:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	70                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x116d:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x1174:0x2b DW_TAG_subprogram
	.byte	74                              # DW_AT_low_pc
	.long	.Lfunc_end73-.Lfunc_begin73     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4484                            # DW_AT_object_pointer
	.long	3901                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x1184:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	24                              # Abbrev [24] 0x118e:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x1198:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x119f:0x5 DW_TAG_pointer_type
	.long	4516                            # DW_AT_type
	.byte	14                              # Abbrev [14] 0x11a4:0xf DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	88                              # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x11aa:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x11ac:0x5 DW_TAG_template_type_parameter
	.long	4531                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0x11b3:0x4 DW_TAG_base_type
	.byte	87                              # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	56                              # Abbrev [56] 0x11b7:0x21 DW_TAG_subprogram
	.byte	75                              # DW_AT_low_pc
	.long	.Lfunc_end74-.Lfunc_begin74     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4551                            # DW_AT_object_pointer
	.long	3923                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x11c7:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	54                              # Abbrev [54] 0x11d1:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
                                        # DW_AT_default_value
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x11d8:0x2b DW_TAG_subprogram
	.byte	76                              # DW_AT_low_pc
	.long	.Lfunc_end75-.Lfunc_begin75     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4584                            # DW_AT_object_pointer
	.long	3944                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x11e8:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	24                              # Abbrev [24] 0x11f2:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	80                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x11fc:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x1203:0x2b DW_TAG_subprogram
	.byte	77                              # DW_AT_low_pc
	.long	.Lfunc_end76-.Lfunc_begin76     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4627                            # DW_AT_object_pointer
	.long	3966                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x1213:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	24                              # Abbrev [24] 0x121d:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	83                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x1227:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x122e:0x2b DW_TAG_subprogram
	.byte	78                              # DW_AT_low_pc
	.long	.Lfunc_end77-.Lfunc_begin77     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4670                            # DW_AT_object_pointer
	.long	3988                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x123e:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	24                              # Abbrev [24] 0x1248:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	86                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x1252:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x1259:0x2b DW_TAG_subprogram
	.byte	79                              # DW_AT_low_pc
	.long	.Lfunc_end78-.Lfunc_begin78     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4713                            # DW_AT_object_pointer
	.long	4010                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x1269:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	24                              # Abbrev [24] 0x1273:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	89                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x127d:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x1284:0x2b DW_TAG_subprogram
	.byte	80                              # DW_AT_low_pc
	.long	.Lfunc_end79-.Lfunc_begin79     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4756                            # DW_AT_object_pointer
	.long	4032                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x1294:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	24                              # Abbrev [24] 0x129e:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	92                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x12a8:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x12af:0x2b DW_TAG_subprogram
	.byte	81                              # DW_AT_low_pc
	.long	.Lfunc_end80-.Lfunc_begin80     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4799                            # DW_AT_object_pointer
	.long	4054                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x12bf:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	24                              # Abbrev [24] 0x12c9:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	95                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x12d3:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x12da:0x2b DW_TAG_subprogram
	.byte	82                              # DW_AT_low_pc
	.long	.Lfunc_end81-.Lfunc_begin81     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4842                            # DW_AT_object_pointer
	.long	4076                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x12ea:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	24                              # Abbrev [24] 0x12f4:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	98                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x12fe:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x1305:0x21 DW_TAG_subprogram
	.byte	83                              # DW_AT_low_pc
	.long	.Lfunc_end82-.Lfunc_begin82     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4885                            # DW_AT_object_pointer
	.long	4098                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x1315:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	15                              # Abbrev [15] 0x131f:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x1326:0x21 DW_TAG_subprogram
	.byte	84                              # DW_AT_low_pc
	.long	.Lfunc_end83-.Lfunc_begin83     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4918                            # DW_AT_object_pointer
	.long	4115                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x1336:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	15                              # Abbrev [15] 0x1340:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x1347:0x2b DW_TAG_subprogram
	.byte	85                              # DW_AT_low_pc
	.long	.Lfunc_end84-.Lfunc_begin84     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4951                            # DW_AT_object_pointer
	.long	4132                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x1357:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	24                              # Abbrev [24] 0x1361:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	107                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x136b:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x1372:0x2b DW_TAG_subprogram
	.byte	86                              # DW_AT_low_pc
	.long	.Lfunc_end85-.Lfunc_begin85     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4994                            # DW_AT_object_pointer
	.long	4154                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x1382:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	24                              # Abbrev [24] 0x138c:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	110                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x1396:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x139d:0x2b DW_TAG_subprogram
	.byte	87                              # DW_AT_low_pc
	.long	.Lfunc_end86-.Lfunc_begin86     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	5037                            # DW_AT_object_pointer
	.long	4176                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x13ad:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	24                              # Abbrev [24] 0x13b7:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	113                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x13c1:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x13c8:0x21 DW_TAG_subprogram
	.byte	88                              # DW_AT_low_pc
	.long	.Lfunc_end87-.Lfunc_begin87     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	5080                            # DW_AT_object_pointer
	.long	4198                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x13d8:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	15                              # Abbrev [15] 0x13e2:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x13e9:0x2b DW_TAG_subprogram
	.byte	89                              # DW_AT_low_pc
	.long	.Lfunc_end88-.Lfunc_begin88     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	5113                            # DW_AT_object_pointer
	.long	4215                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x13f9:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	24                              # Abbrev [24] 0x1403:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	119                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x140d:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x1414:0x2b DW_TAG_subprogram
	.byte	90                              # DW_AT_low_pc
	.long	.Lfunc_end89-.Lfunc_begin89     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	5156                            # DW_AT_object_pointer
	.long	4237                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x1424:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9350                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	24                              # Abbrev [24] 0x142e:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	122                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x1438:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	58                              # Abbrev [58] 0x143f:0x1 DW_TAG_pointer_type
	.byte	59                              # Abbrev [59] 0x1440:0x13 DW_TAG_subprogram
	.byte	91                              # DW_AT_low_pc
	.long	.Lfunc_end90-.Lfunc_begin90     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4259                            # DW_AT_specification
	.byte	15                              # Abbrev [15] 0x144c:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	59                              # Abbrev [59] 0x1453:0x13 DW_TAG_subprogram
	.byte	92                              # DW_AT_low_pc
	.long	.Lfunc_end91-.Lfunc_begin91     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4285                            # DW_AT_specification
	.byte	15                              # Abbrev [15] 0x145f:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	59                              # Abbrev [59] 0x1466:0x27 DW_TAG_subprogram
	.byte	93                              # DW_AT_low_pc
	.long	.Lfunc_end92-.Lfunc_begin92     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4311                            # DW_AT_specification
	.byte	24                              # Abbrev [24] 0x1472:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	0                               # DW_AT_decl_file
	.byte	129                             # DW_AT_decl_line
	.long	5183                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x147c:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	129                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x1486:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	59                              # Abbrev [59] 0x148d:0x27 DW_TAG_subprogram
	.byte	94                              # DW_AT_low_pc
	.long	.Lfunc_end93-.Lfunc_begin93     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4333                            # DW_AT_specification
	.byte	24                              # Abbrev [24] 0x1499:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	0                               # DW_AT_decl_file
	.byte	136                             # DW_AT_decl_line
	.long	5183                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x14a3:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	136                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x14ad:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	59                              # Abbrev [59] 0x14b4:0x13 DW_TAG_subprogram
	.byte	95                              # DW_AT_low_pc
	.long	.Lfunc_end94-.Lfunc_begin94     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4355                            # DW_AT_specification
	.byte	15                              # Abbrev [15] 0x14c0:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	45                              # Abbrev [45] 0x14c7:0x2f DW_TAG_subprogram
	.byte	96                              # DW_AT_low_pc
	.long	.Lfunc_end95-.Lfunc_begin95     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	296                             # DW_AT_linkage_name
	.short	297                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
	.byte	27                              # Abbrev [27] 0x14d5:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9355                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x14e1:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9371                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x14ed:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x14ef:0x5 DW_TAG_template_type_parameter
	.long	976                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x14f6:0x2f DW_TAG_subprogram
	.byte	97                              # DW_AT_low_pc
	.long	.Lfunc_end96-.Lfunc_begin96     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	298                             # DW_AT_linkage_name
	.short	299                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x1504:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9388                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1510:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9404                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x151c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x151e:0x5 DW_TAG_template_type_parameter
	.long	7401                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1525:0x2f DW_TAG_subprogram
	.byte	98                              # DW_AT_low_pc
	.long	.Lfunc_end97-.Lfunc_begin97     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	300                             # DW_AT_linkage_name
	.short	301                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x1533:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9421                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x153f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9437                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x154b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x154d:0x5 DW_TAG_template_type_parameter
	.long	7406                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1554:0x13 DW_TAG_subprogram
	.byte	99                              # DW_AT_low_pc
	.long	.Lfunc_end98-.Lfunc_begin98     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	303                             # DW_AT_linkage_name
	.short	304                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	143                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	13                              # Abbrev [13] 0x1562:0x4 DW_TAG_GNU_template_template_param
	.byte	20                              # DW_AT_name
	.short	302                             # DW_AT_GNU_template_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1567:0x1a DW_TAG_subprogram
	.byte	100                             # DW_AT_low_pc
	.long	.Lfunc_end99-.Lfunc_begin99     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	305                             # DW_AT_linkage_name
	.short	306                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	144                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	13                              # Abbrev [13] 0x1575:0x4 DW_TAG_GNU_template_template_param
	.byte	20                              # DW_AT_name
	.short	302                             # DW_AT_GNU_template_name
	.byte	49                              # Abbrev [49] 0x1579:0x7 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.short	279                             # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1581:0x34 DW_TAG_subprogram
	.byte	102                             # DW_AT_low_pc
	.long	.Lfunc_end101-.Lfunc_begin101   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	310                             # DW_AT_linkage_name
	.short	311                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x158f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9454                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x159b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9475                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x15a7:0xd DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x15a9:0x5 DW_TAG_template_type_parameter
	.long	7177                            # DW_AT_type
	.byte	34                              # Abbrev [34] 0x15ae:0x5 DW_TAG_template_type_parameter
	.long	7411                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x15b5:0x2f DW_TAG_subprogram
	.byte	103                             # DW_AT_low_pc
	.long	.Lfunc_end102-.Lfunc_begin102   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	313                             # DW_AT_linkage_name
	.short	314                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x15c3:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9497                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x15cf:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9513                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x15db:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x15dd:0x5 DW_TAG_template_type_parameter
	.long	7416                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x15e4:0x13 DW_TAG_subprogram
	.byte	104                             # DW_AT_low_pc
	.long	.Lfunc_end103-.Lfunc_begin103   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	316                             # DW_AT_linkage_name
	.short	317                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	143                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	13                              # Abbrev [13] 0x15f2:0x4 DW_TAG_GNU_template_template_param
	.byte	20                              # DW_AT_name
	.short	315                             # DW_AT_GNU_template_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x15f7:0x2f DW_TAG_subprogram
	.byte	105                             # DW_AT_low_pc
	.long	.Lfunc_end104-.Lfunc_begin104   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	318                             # DW_AT_linkage_name
	.short	319                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x1605:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9530                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1611:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9546                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x161d:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x161f:0x5 DW_TAG_template_type_parameter
	.long	7430                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1626:0x39 DW_TAG_subprogram
	.byte	106                             # DW_AT_low_pc
	.long	.Lfunc_end105-.Lfunc_begin105   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	320                             # DW_AT_linkage_name
	.short	321                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x1634:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9563                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1640:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9589                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x164c:0x12 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x164e:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	34                              # Abbrev [34] 0x1653:0x5 DW_TAG_template_type_parameter
	.long	505                             # DW_AT_type
	.byte	34                              # Abbrev [34] 0x1658:0x5 DW_TAG_template_type_parameter
	.long	7435                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x165f:0x2f DW_TAG_subprogram
	.byte	107                             # DW_AT_low_pc
	.long	.Lfunc_end106-.Lfunc_begin106   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	322                             # DW_AT_linkage_name
	.short	323                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x166d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9616                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1679:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9632                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1685:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1687:0x5 DW_TAG_template_type_parameter
	.long	7440                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x168e:0x2f DW_TAG_subprogram
	.byte	108                             # DW_AT_low_pc
	.long	.Lfunc_end107-.Lfunc_begin107   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	324                             # DW_AT_linkage_name
	.short	325                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x169c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9649                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x16a8:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9665                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x16b4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x16b6:0x5 DW_TAG_template_type_parameter
	.long	7452                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x16bd:0x2f DW_TAG_subprogram
	.byte	109                             # DW_AT_low_pc
	.long	.Lfunc_end108-.Lfunc_begin108   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	326                             # DW_AT_linkage_name
	.short	327                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x16cb:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9682                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x16d7:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9698                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x16e3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x16e5:0x5 DW_TAG_template_type_parameter
	.long	7462                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	45                              # Abbrev [45] 0x16ec:0x2f DW_TAG_subprogram
	.byte	110                             # DW_AT_low_pc
	.long	.Lfunc_end109-.Lfunc_begin109   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	329                             # DW_AT_linkage_name
	.short	330                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
	.byte	27                              # Abbrev [27] 0x16fa:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9715                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1706:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9731                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1712:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1714:0x5 DW_TAG_template_type_parameter
	.long	7468                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x171b:0x5 DW_TAG_pointer_type
	.long	206                             # DW_AT_type
	.byte	60                              # Abbrev [60] 0x1720:0x1f DW_TAG_subprogram
	.byte	111                             # DW_AT_low_pc
	.long	.Lfunc_end110-.Lfunc_begin110   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	5938                            # DW_AT_object_pointer
	.short	331                             # DW_AT_linkage_name
	.long	212                             # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x1732:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	9748                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	18                              # Abbrev [18] 0x173c:0x2 DW_TAG_template_type_parameter
	.byte	20                              # DW_AT_name
                                        # DW_AT_default_value
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x173f:0x2f DW_TAG_subprogram
	.byte	112                             # DW_AT_low_pc
	.long	.Lfunc_end111-.Lfunc_begin111   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	332                             # DW_AT_linkage_name
	.short	333                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x174d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9753                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1759:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9769                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1765:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1767:0x5 DW_TAG_template_type_parameter
	.long	7484                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x176e:0x2f DW_TAG_subprogram
	.byte	113                             # DW_AT_low_pc
	.long	.Lfunc_end112-.Lfunc_begin112   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	334                             # DW_AT_linkage_name
	.short	335                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x177c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9786                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1788:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9802                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1794:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1796:0x5 DW_TAG_template_type_parameter
	.long	7510                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x179d:0x2f DW_TAG_subprogram
	.byte	114                             # DW_AT_low_pc
	.long	.Lfunc_end113-.Lfunc_begin113   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	336                             # DW_AT_linkage_name
	.short	337                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x17ab:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9819                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x17b7:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9835                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x17c3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x17c5:0x5 DW_TAG_template_type_parameter
	.long	7536                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	61                              # Abbrev [61] 0x17cc:0x19 DW_TAG_subprogram
	.byte	115                             # DW_AT_low_pc
	.long	.Lfunc_end114-.Lfunc_begin114   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	338                             # DW_AT_linkage_name
	.short	339                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	162                             # DW_AT_decl_line
	.long	7354                            # DW_AT_type
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x17de:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x17e5:0x2f DW_TAG_subprogram
	.byte	116                             # DW_AT_low_pc
	.long	.Lfunc_end115-.Lfunc_begin115   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	340                             # DW_AT_linkage_name
	.short	341                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x17f3:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9852                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x17ff:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9868                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x180b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x180d:0x5 DW_TAG_template_type_parameter
	.long	7562                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1814:0x2f DW_TAG_subprogram
	.byte	117                             # DW_AT_low_pc
	.long	.Lfunc_end116-.Lfunc_begin116   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	342                             # DW_AT_linkage_name
	.short	343                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x1822:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9885                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x182e:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9901                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x183a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x183c:0x5 DW_TAG_template_type_parameter
	.long	7567                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1843:0x2f DW_TAG_subprogram
	.byte	118                             # DW_AT_low_pc
	.long	.Lfunc_end117-.Lfunc_begin117   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	344                             # DW_AT_linkage_name
	.short	345                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x1851:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9918                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x185d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9934                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1869:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x186b:0x5 DW_TAG_template_type_parameter
	.long	7589                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1872:0x2f DW_TAG_subprogram
	.byte	119                             # DW_AT_low_pc
	.long	.Lfunc_end118-.Lfunc_begin118   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	346                             # DW_AT_linkage_name
	.short	347                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x1880:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9951                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x188c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	9967                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1898:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x189a:0x5 DW_TAG_template_type_parameter
	.long	7595                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x18a1:0x2f DW_TAG_subprogram
	.byte	120                             # DW_AT_low_pc
	.long	.Lfunc_end119-.Lfunc_begin119   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	348                             # DW_AT_linkage_name
	.short	349                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x18af:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	9984                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x18bb:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	10000                           # DW_AT_type
	.byte	33                              # Abbrev [33] 0x18c7:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x18c9:0x5 DW_TAG_template_type_parameter
	.long	7601                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x18d0:0x2f DW_TAG_subprogram
	.byte	121                             # DW_AT_low_pc
	.long	.Lfunc_end120-.Lfunc_begin120   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	350                             # DW_AT_linkage_name
	.short	351                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x18de:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	10017                           # DW_AT_type
	.byte	27                              # Abbrev [27] 0x18ea:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	10033                           # DW_AT_type
	.byte	33                              # Abbrev [33] 0x18f6:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x18f8:0x5 DW_TAG_template_type_parameter
	.long	7611                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x18ff:0x2f DW_TAG_subprogram
	.byte	122                             # DW_AT_low_pc
	.long	.Lfunc_end121-.Lfunc_begin121   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	352                             # DW_AT_linkage_name
	.short	353                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x190d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	10050                           # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1919:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	10066                           # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1925:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1927:0x5 DW_TAG_template_type_parameter
	.long	7628                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x192e:0x2f DW_TAG_subprogram
	.byte	123                             # DW_AT_low_pc
	.long	.Lfunc_end122-.Lfunc_begin122   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	354                             # DW_AT_linkage_name
	.short	355                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x193c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	10083                           # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1948:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	10099                           # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1954:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1956:0x5 DW_TAG_template_type_parameter
	.long	7633                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x195d:0x2f DW_TAG_subprogram
	.byte	124                             # DW_AT_low_pc
	.long	.Lfunc_end123-.Lfunc_begin123   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	356                             # DW_AT_linkage_name
	.short	357                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x196b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	10116                           # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1977:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	10132                           # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1983:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1985:0x5 DW_TAG_template_type_parameter
	.long	7664                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x198c:0x2f DW_TAG_subprogram
	.byte	125                             # DW_AT_low_pc
	.long	.Lfunc_end124-.Lfunc_begin124   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	358                             # DW_AT_linkage_name
	.short	359                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x199a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	10149                           # DW_AT_type
	.byte	27                              # Abbrev [27] 0x19a6:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	10165                           # DW_AT_type
	.byte	33                              # Abbrev [33] 0x19b2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x19b4:0x5 DW_TAG_template_type_parameter
	.long	7687                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x19bb:0x2f DW_TAG_subprogram
	.byte	126                             # DW_AT_low_pc
	.long	.Lfunc_end125-.Lfunc_begin125   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	360                             # DW_AT_linkage_name
	.short	361                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x19c9:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	10182                           # DW_AT_type
	.byte	27                              # Abbrev [27] 0x19d5:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	10198                           # DW_AT_type
	.byte	33                              # Abbrev [33] 0x19e1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x19e3:0x5 DW_TAG_template_type_parameter
	.long	7354                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	45                              # Abbrev [45] 0x19ea:0x2f DW_TAG_subprogram
	.byte	127                             # DW_AT_low_pc
	.long	.Lfunc_end126-.Lfunc_begin126   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	362                             # DW_AT_linkage_name
	.short	363                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
	.byte	27                              # Abbrev [27] 0x19f8:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	10215                           # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1a04:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	10231                           # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1a10:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1a12:0x5 DW_TAG_template_type_parameter
	.long	7699                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	45                              # Abbrev [45] 0x1a19:0x30 DW_TAG_subprogram
	.ascii	"\200\001"                      # DW_AT_low_pc
	.long	.Lfunc_end127-.Lfunc_begin127   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	364                             # DW_AT_linkage_name
	.short	365                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
	.byte	27                              # Abbrev [27] 0x1a28:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	10248                           # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1a34:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	10264                           # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1a40:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1a42:0x5 DW_TAG_template_type_parameter
	.long	7706                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	45                              # Abbrev [45] 0x1a49:0x30 DW_TAG_subprogram
	.ascii	"\201\001"                      # DW_AT_low_pc
	.long	.Lfunc_end128-.Lfunc_begin128   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	366                             # DW_AT_linkage_name
	.short	367                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
	.byte	27                              # Abbrev [27] 0x1a58:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	10281                           # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1a64:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	10297                           # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1a70:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1a72:0x5 DW_TAG_template_type_parameter
	.long	7718                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1a79:0x16 DW_TAG_subprogram
	.ascii	"\202\001"                      # DW_AT_low_pc
	.long	.Lfunc_end129-.Lfunc_begin129   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	368                             # DW_AT_linkage_name
	.short	369                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	171                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x1a88:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1a8f:0x30 DW_TAG_subprogram
	.ascii	"\203\001"                      # DW_AT_low_pc
	.long	.Lfunc_end130-.Lfunc_begin130   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	371                             # DW_AT_linkage_name
	.short	372                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x1a9e:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	10314                           # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1aaa:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	10330                           # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1ab6:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1ab8:0x5 DW_TAG_template_type_parameter
	.long	7725                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1abf:0x30 DW_TAG_subprogram
	.ascii	"\204\001"                      # DW_AT_low_pc
	.long	.Lfunc_end131-.Lfunc_begin131   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	374                             # DW_AT_linkage_name
	.short	375                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x1ace:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	10347                           # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1ada:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	10363                           # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1ae6:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1ae8:0x5 DW_TAG_template_type_parameter
	.long	7730                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1aef:0x30 DW_TAG_subprogram
	.ascii	"\205\001"                      # DW_AT_low_pc
	.long	.Lfunc_end132-.Lfunc_begin132   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	377                             # DW_AT_linkage_name
	.short	378                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x1afe:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	10380                           # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1b0a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	10396                           # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1b16:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1b18:0x5 DW_TAG_template_type_parameter
	.long	7740                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1b1f:0x30 DW_TAG_subprogram
	.ascii	"\206\001"                      # DW_AT_low_pc
	.long	.Lfunc_end133-.Lfunc_begin133   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	379                             # DW_AT_linkage_name
	.short	380                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x1b2e:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	10413                           # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1b3a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	10429                           # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1b46:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1b48:0x5 DW_TAG_template_type_parameter
	.long	7762                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	14                              # Abbrev [14] 0x1b4f:0x12 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	133                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	153                             # DW_AT_decl_line
	.byte	51                              # Abbrev [51] 0x1b55:0xb DW_TAG_subprogram
	.byte	131                             # DW_AT_linkage_name
	.byte	132                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	154                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	19                              # Abbrev [19] 0x1b5a:0x5 DW_TAG_formal_parameter
	.long	7009                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1b61:0x5 DW_TAG_pointer_type
	.long	6991                            # DW_AT_type
	.byte	62                              # Abbrev [62] 0x1b66:0x21 DW_TAG_subprogram
	.ascii	"\207\001"                      # DW_AT_low_pc
	.long	.Lfunc_end134-.Lfunc_begin134   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	7033                            # DW_AT_object_pointer
	.short	327                             # DW_AT_decl_line
	.long	6997                            # DW_AT_specification
	.byte	57                              # Abbrev [57] 0x1b79:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	485                             # DW_AT_name
	.long	10446                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	32                              # Abbrev [32] 0x1b83:0x3 DW_TAG_structure_type
	.short	295                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	45                              # Abbrev [45] 0x1b87:0x30 DW_TAG_subprogram
	.ascii	"\210\001"                      # DW_AT_low_pc
	.long	.Lfunc_end135-.Lfunc_begin135   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	381                             # DW_AT_linkage_name
	.short	297                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
	.byte	27                              # Abbrev [27] 0x1b96:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	10451                           # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1ba2:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	10467                           # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1bae:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1bb0:0x5 DW_TAG_template_type_parameter
	.long	7043                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1bb7:0x30 DW_TAG_subprogram
	.ascii	"\211\001"                      # DW_AT_low_pc
	.long	.Lfunc_end136-.Lfunc_begin136   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	382                             # DW_AT_linkage_name
	.short	383                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	33                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	27                              # Abbrev [27] 0x1bc6:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	388                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.long	10484                           # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1bd2:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.long	10500                           # DW_AT_type
	.byte	33                              # Abbrev [33] 0x1bde:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1be0:0x5 DW_TAG_template_type_parameter
	.long	7771                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0x1be7:0x4 DW_TAG_base_type
	.byte	143                             # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x1beb:0x4 DW_TAG_base_type
	.byte	152                             # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x1bef:0x4 DW_TAG_base_type
	.byte	155                             # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	7                               # Abbrev [7] 0x1bf3:0x2 DW_TAG_structure_type
	.byte	158                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	55                              # Abbrev [55] 0x1bf5:0x5 DW_TAG_pointer_type
	.long	170                             # DW_AT_type
	.byte	14                              # Abbrev [14] 0x1bfa:0xf DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	168                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x1c00:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1c02:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1c09:0x5 DW_TAG_pointer_type
	.long	54                              # DW_AT_type
	.byte	63                              # Abbrev [63] 0x1c0e:0x5 DW_TAG_reference_type
	.long	54                              # DW_AT_type
	.byte	64                              # Abbrev [64] 0x1c13:0x5 DW_TAG_rvalue_reference_type
	.long	54                              # DW_AT_type
	.byte	65                              # Abbrev [65] 0x1c18:0x5 DW_TAG_const_type
	.long	54                              # DW_AT_type
	.byte	66                              # Abbrev [66] 0x1c1d:0xc DW_TAG_array_type
	.long	54                              # DW_AT_type
	.byte	67                              # Abbrev [67] 0x1c22:0x6 DW_TAG_subrange_type
	.long	7209                            # DW_AT_type
	.byte	3                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	68                              # Abbrev [68] 0x1c29:0x4 DW_TAG_base_type
	.byte	181                             # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	7                               # DW_AT_encoding
	.byte	14                              # Abbrev [14] 0x1c2d:0x9 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	186                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	50                              # DW_AT_decl_line
	.byte	7                               # Abbrev [7] 0x1c33:0x2 DW_TAG_structure_type
	.byte	187                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0x1c36:0x4 DW_TAG_base_type
	.byte	225                             # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x1c3a:0x4 DW_TAG_base_type
	.byte	228                             # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	16                              # DW_AT_byte_size
	.byte	69                              # Abbrev [69] 0x1c3e:0x10 DW_TAG_structure_type
	.byte	235                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	15                              # Abbrev [15] 0x1c40:0x6 DW_TAG_template_type_parameter
	.long	182                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	16                              # Abbrev [16] 0x1c46:0x7 DW_TAG_template_value_parameter
	.long	202                             # DW_AT_type
	.byte	22                              # DW_AT_name
                                        # DW_AT_default_value
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	69                              # Abbrev [69] 0x1c4e:0x10 DW_TAG_structure_type
	.byte	241                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	15                              # Abbrev [15] 0x1c50:0x6 DW_TAG_template_type_parameter
	.long	7262                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	16                              # Abbrev [16] 0x1c56:0x7 DW_TAG_template_value_parameter
	.long	202                             # DW_AT_type
	.byte	22                              # DW_AT_name
                                        # DW_AT_default_value
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	14                              # Abbrev [14] 0x1c5e:0x14 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	240                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	47                              # DW_AT_decl_line
	.byte	15                              # Abbrev [15] 0x1c64:0x6 DW_TAG_template_type_parameter
	.long	966                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	16                              # Abbrev [16] 0x1c6a:0x7 DW_TAG_template_value_parameter
	.long	202                             # DW_AT_type
	.byte	22                              # DW_AT_name
                                        # DW_AT_default_value
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	70                              # Abbrev [70] 0x1c72:0xb DW_TAG_subroutine_type
	.long	54                              # DW_AT_type
	.byte	52                              # Abbrev [52] 0x1c77:0x5 DW_TAG_formal_parameter
	.long	4531                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	71                              # Abbrev [71] 0x1c7d:0x3 DW_TAG_subroutine_type
	.byte	72                              # Abbrev [72] 0x1c7e:0x1 DW_TAG_unspecified_parameters
	.byte	0                               # End Of Children Mark
	.byte	71                              # Abbrev [71] 0x1c80:0x8 DW_TAG_subroutine_type
	.byte	52                              # Abbrev [52] 0x1c81:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	72                              # Abbrev [72] 0x1c86:0x1 DW_TAG_unspecified_parameters
	.byte	0                               # End Of Children Mark
	.byte	63                              # Abbrev [63] 0x1c88:0x5 DW_TAG_reference_type
	.long	7192                            # DW_AT_type
	.byte	63                              # Abbrev [63] 0x1c8d:0x5 DW_TAG_reference_type
	.long	7314                            # DW_AT_type
	.byte	55                              # Abbrev [55] 0x1c92:0x5 DW_TAG_pointer_type
	.long	7192                            # DW_AT_type
	.byte	73                              # Abbrev [73] 0x1c97:0x3 DW_TAG_unspecified_type
	.short	257                             # DW_AT_name
	.byte	55                              # Abbrev [55] 0x1c9a:0x5 DW_TAG_pointer_type
	.long	505                             # DW_AT_type
	.byte	55                              # Abbrev [55] 0x1c9f:0x5 DW_TAG_pointer_type
	.long	7155                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x1ca4:0x5 DW_TAG_const_type
	.long	5183                            # DW_AT_type
	.byte	55                              # Abbrev [55] 0x1ca9:0x5 DW_TAG_pointer_type
	.long	7342                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x1cae:0x5 DW_TAG_const_type
	.long	7347                            # DW_AT_type
	.byte	55                              # Abbrev [55] 0x1cb3:0x5 DW_TAG_pointer_type
	.long	7352                            # DW_AT_type
	.byte	74                              # Abbrev [74] 0x1cb8:0x1 DW_TAG_const_type
	.byte	75                              # Abbrev [75] 0x1cb9:0x1 DW_TAG_subroutine_type
	.byte	55                              # Abbrev [55] 0x1cba:0x5 DW_TAG_pointer_type
	.long	7353                            # DW_AT_type
	.byte	55                              # Abbrev [55] 0x1cbf:0x5 DW_TAG_pointer_type
	.long	966                             # DW_AT_type
	.byte	55                              # Abbrev [55] 0x1cc4:0x5 DW_TAG_pointer_type
	.long	971                             # DW_AT_type
	.byte	55                              # Abbrev [55] 0x1cc9:0x5 DW_TAG_pointer_type
	.long	7374                            # DW_AT_type
	.byte	76                              # Abbrev [76] 0x1cce:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	290                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x1cd5:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1cd7:0x5 DW_TAG_template_type_parameter
	.long	7177                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	66                              # Abbrev [66] 0x1cde:0xb DW_TAG_array_type
	.long	7177                            # DW_AT_type
	.byte	77                              # Abbrev [77] 0x1ce3:0x5 DW_TAG_subrange_type
	.long	7209                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	63                              # Abbrev [63] 0x1ce9:0x5 DW_TAG_reference_type
	.long	7197                            # DW_AT_type
	.byte	55                              # Abbrev [55] 0x1cee:0x5 DW_TAG_pointer_type
	.long	7197                            # DW_AT_type
	.byte	55                              # Abbrev [55] 0x1cf3:0x5 DW_TAG_pointer_type
	.long	7319                            # DW_AT_type
	.byte	76                              # Abbrev [76] 0x1cf8:0xe DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	312                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	151                             # DW_AT_decl_line
	.byte	15                              # Abbrev [15] 0x1cff:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	78                              # Abbrev [78] 0x1d06:0x5 DW_TAG_atomic_type
	.long	54                              # DW_AT_type
	.byte	79                              # Abbrev [79] 0x1d0b:0x5 DW_TAG_volatile_type
	.long	7222                            # DW_AT_type
	.byte	80                              # Abbrev [80] 0x1d10:0xc DW_TAG_array_type
                                        # DW_AT_GNU_vector
	.long	54                              # DW_AT_type
	.byte	67                              # Abbrev [67] 0x1d15:0x6 DW_TAG_subrange_type
	.long	7209                            # DW_AT_type
	.byte	2                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	65                              # Abbrev [65] 0x1d1c:0x5 DW_TAG_const_type
	.long	7457                            # DW_AT_type
	.byte	79                              # Abbrev [79] 0x1d21:0x5 DW_TAG_volatile_type
	.long	7177                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x1d26:0x5 DW_TAG_const_type
	.long	7467                            # DW_AT_type
	.byte	81                              # Abbrev [81] 0x1d2b:0x1 DW_TAG_volatile_type
	.byte	76                              # Abbrev [76] 0x1d2c:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	328                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x1d33:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1d35:0x5 DW_TAG_template_type_parameter
	.long	966                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	82                              # Abbrev [82] 0x1d3c:0x9 DW_TAG_ptr_to_member_type
	.long	7493                            # DW_AT_type
	.long	7155                            # DW_AT_containing_type
	.byte	71                              # Abbrev [71] 0x1d45:0x7 DW_TAG_subroutine_type
	.byte	19                              # Abbrev [19] 0x1d46:0x5 DW_TAG_formal_parameter
	.long	7500                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1d4c:0x5 DW_TAG_pointer_type
	.long	7505                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x1d51:0x5 DW_TAG_const_type
	.long	7155                            # DW_AT_type
	.byte	82                              # Abbrev [82] 0x1d56:0x9 DW_TAG_ptr_to_member_type
	.long	7519                            # DW_AT_type
	.long	7155                            # DW_AT_containing_type
	.byte	83                              # Abbrev [83] 0x1d5f:0x7 DW_TAG_subroutine_type
                                        # DW_AT_reference
	.byte	19                              # Abbrev [19] 0x1d60:0x5 DW_TAG_formal_parameter
	.long	7526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1d66:0x5 DW_TAG_pointer_type
	.long	7531                            # DW_AT_type
	.byte	79                              # Abbrev [79] 0x1d6b:0x5 DW_TAG_volatile_type
	.long	7155                            # DW_AT_type
	.byte	82                              # Abbrev [82] 0x1d70:0x9 DW_TAG_ptr_to_member_type
	.long	7545                            # DW_AT_type
	.long	7155                            # DW_AT_containing_type
	.byte	84                              # Abbrev [84] 0x1d79:0x7 DW_TAG_subroutine_type
                                        # DW_AT_rvalue_reference
	.byte	19                              # Abbrev [19] 0x1d7a:0x5 DW_TAG_formal_parameter
	.long	7552                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1d80:0x5 DW_TAG_pointer_type
	.long	7557                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x1d85:0x5 DW_TAG_const_type
	.long	7531                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x1d8a:0x5 DW_TAG_const_type
	.long	7354                            # DW_AT_type
	.byte	63                              # Abbrev [63] 0x1d8f:0x5 DW_TAG_reference_type
	.long	7572                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x1d94:0x5 DW_TAG_const_type
	.long	7577                            # DW_AT_type
	.byte	66                              # Abbrev [66] 0x1d99:0xc DW_TAG_array_type
	.long	7222                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x1d9e:0x6 DW_TAG_subrange_type
	.long	7209                            # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	65                              # Abbrev [65] 0x1da5:0x5 DW_TAG_const_type
	.long	7594                            # DW_AT_type
	.byte	85                              # Abbrev [85] 0x1daa:0x1 DW_TAG_subroutine_type
                                        # DW_AT_reference
	.byte	79                              # Abbrev [79] 0x1dab:0x5 DW_TAG_volatile_type
	.long	7600                            # DW_AT_type
	.byte	86                              # Abbrev [86] 0x1db0:0x1 DW_TAG_subroutine_type
                                        # DW_AT_rvalue_reference
	.byte	65                              # Abbrev [65] 0x1db1:0x5 DW_TAG_const_type
	.long	7606                            # DW_AT_type
	.byte	79                              # Abbrev [79] 0x1db6:0x5 DW_TAG_volatile_type
	.long	7353                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x1dbb:0x5 DW_TAG_const_type
	.long	7616                            # DW_AT_type
	.byte	66                              # Abbrev [66] 0x1dc0:0xc DW_TAG_array_type
	.long	7177                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x1dc5:0x6 DW_TAG_subrange_type
	.long	7209                            # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	63                              # Abbrev [63] 0x1dcc:0x5 DW_TAG_reference_type
	.long	7611                            # DW_AT_type
	.byte	63                              # Abbrev [63] 0x1dd1:0x5 DW_TAG_reference_type
	.long	7638                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x1dd6:0x5 DW_TAG_const_type
	.long	7643                            # DW_AT_type
	.byte	82                              # Abbrev [82] 0x1ddb:0x9 DW_TAG_ptr_to_member_type
	.long	7652                            # DW_AT_type
	.long	7155                            # DW_AT_containing_type
	.byte	71                              # Abbrev [71] 0x1de4:0x7 DW_TAG_subroutine_type
	.byte	19                              # Abbrev [19] 0x1de5:0x5 DW_TAG_formal_parameter
	.long	7659                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1deb:0x5 DW_TAG_pointer_type
	.long	7155                            # DW_AT_type
	.byte	70                              # Abbrev [70] 0x1df0:0xb DW_TAG_subroutine_type
	.long	7675                            # DW_AT_type
	.byte	52                              # Abbrev [52] 0x1df5:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1dfb:0x5 DW_TAG_pointer_type
	.long	7680                            # DW_AT_type
	.byte	71                              # Abbrev [71] 0x1e00:0x7 DW_TAG_subroutine_type
	.byte	52                              # Abbrev [52] 0x1e01:0x5 DW_TAG_formal_parameter
	.long	4531                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	66                              # Abbrev [66] 0x1e07:0xc DW_TAG_array_type
	.long	7162                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x1e0c:0x6 DW_TAG_subrange_type
	.long	7209                            # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	71                              # Abbrev [71] 0x1e13:0x7 DW_TAG_subroutine_type
	.byte	52                              # Abbrev [52] 0x1e14:0x5 DW_TAG_formal_parameter
	.long	971                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	71                              # Abbrev [71] 0x1e1a:0xc DW_TAG_subroutine_type
	.byte	52                              # Abbrev [52] 0x1e1b:0x5 DW_TAG_formal_parameter
	.long	979                             # DW_AT_type
	.byte	52                              # Abbrev [52] 0x1e20:0x5 DW_TAG_formal_parameter
	.long	971                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	71                              # Abbrev [71] 0x1e26:0x7 DW_TAG_subroutine_type
	.byte	52                              # Abbrev [52] 0x1e27:0x5 DW_TAG_formal_parameter
	.long	979                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	87                              # Abbrev [87] 0x1e2d:0x5 DW_TAG_base_type
	.short	370                             # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	65                              # Abbrev [65] 0x1e32:0x5 DW_TAG_const_type
	.long	7735                            # DW_AT_type
	.byte	87                              # Abbrev [87] 0x1e37:0x5 DW_TAG_base_type
	.short	373                             # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	71                              # Abbrev [71] 0x1e3c:0xc DW_TAG_subroutine_type
	.byte	52                              # Abbrev [52] 0x1e3d:0x5 DW_TAG_formal_parameter
	.long	7752                            # DW_AT_type
	.byte	52                              # Abbrev [52] 0x1e42:0x5 DW_TAG_formal_parameter
	.long	7752                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x1e48:0xa DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	376                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	50                              # Abbrev [50] 0x1e4f:0x2 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	82                              # Abbrev [82] 0x1e52:0x9 DW_TAG_ptr_to_member_type
	.long	54                              # DW_AT_type
	.long	7752                            # DW_AT_containing_type
	.byte	82                              # Abbrev [82] 0x1e5b:0x9 DW_TAG_ptr_to_member_type
	.long	7780                            # DW_AT_type
	.long	6991                            # DW_AT_containing_type
	.byte	71                              # Abbrev [71] 0x1e64:0x7 DW_TAG_subroutine_type
	.byte	19                              # Abbrev [19] 0x1e65:0x5 DW_TAG_formal_parameter
	.long	7009                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x1e6b:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	387                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	56                              # DW_AT_decl_line
	.byte	88                              # Abbrev [88] 0x1e72:0x8 DW_TAG_template_value_parameter
	.long	75                              # DW_AT_type
	.short	386                             # DW_AT_name
	.byte	3                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	89                              # Abbrev [89] 0x1e7b:0x9 DW_TAG_typedef
	.long	7416                            # DW_AT_type
	.short	391                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	147                             # DW_AT_decl_line
	.byte	76                              # Abbrev [76] 0x1e84:0x12 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	395                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	178                             # DW_AT_decl_line
	.byte	90                              # Abbrev [90] 0x1e8b:0xa DW_TAG_member
	.short	388                             # DW_AT_name
	.long	7830                            # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	179                             # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x1e96:0x17 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	394                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	175                             # DW_AT_decl_line
	.byte	15                              # Abbrev [15] 0x1e9d:0x6 DW_TAG_template_type_parameter
	.long	59                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	33                              # Abbrev [33] 0x1ea3:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	194                             # DW_AT_name
	.byte	38                              # Abbrev [38] 0x1ea5:0x6 DW_TAG_template_value_parameter
	.long	59                              # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1ead:0x5 DW_TAG_pointer_type
	.long	7858                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x1eb2:0xc DW_TAG_structure_type
	.short	396                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x1eb5:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1eb7:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1ebe:0x5 DW_TAG_pointer_type
	.long	7875                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x1ec3:0xc DW_TAG_structure_type
	.short	397                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x1ec6:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1ec8:0x5 DW_TAG_template_type_parameter
	.long	4531                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x1ecf:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	398                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x1ed6:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1ed8:0x5 DW_TAG_template_type_parameter
	.long	202                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1edf:0x5 DW_TAG_pointer_type
	.long	7908                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x1ee4:0xc DW_TAG_structure_type
	.short	399                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x1ee7:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1ee9:0x5 DW_TAG_template_type_parameter
	.long	202                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x1ef0:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	400                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x1ef7:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1ef9:0x5 DW_TAG_template_type_parameter
	.long	7143                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1f00:0x5 DW_TAG_pointer_type
	.long	7941                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x1f05:0xc DW_TAG_structure_type
	.short	401                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x1f08:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1f0a:0x5 DW_TAG_template_type_parameter
	.long	7143                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x1f11:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	402                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x1f18:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1f1a:0x5 DW_TAG_template_type_parameter
	.long	505                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1f21:0x5 DW_TAG_pointer_type
	.long	7974                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x1f26:0xc DW_TAG_structure_type
	.short	403                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x1f29:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1f2b:0x5 DW_TAG_template_type_parameter
	.long	505                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x1f32:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	404                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x1f39:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1f3b:0x5 DW_TAG_template_type_parameter
	.long	469                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1f42:0x5 DW_TAG_pointer_type
	.long	8007                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x1f47:0xc DW_TAG_structure_type
	.short	405                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x1f4a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1f4c:0x5 DW_TAG_template_type_parameter
	.long	469                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x1f53:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	406                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x1f5a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1f5c:0x5 DW_TAG_template_type_parameter
	.long	75                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1f63:0x5 DW_TAG_pointer_type
	.long	8040                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x1f68:0xc DW_TAG_structure_type
	.short	407                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x1f6b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1f6d:0x5 DW_TAG_template_type_parameter
	.long	75                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x1f74:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	408                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x1f7b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1f7d:0x5 DW_TAG_template_type_parameter
	.long	7147                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1f84:0x5 DW_TAG_pointer_type
	.long	8073                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x1f89:0xc DW_TAG_structure_type
	.short	409                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x1f8c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1f8e:0x5 DW_TAG_template_type_parameter
	.long	7147                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x1f95:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	410                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x1f9c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1f9e:0x5 DW_TAG_template_type_parameter
	.long	7151                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1fa5:0x5 DW_TAG_pointer_type
	.long	8106                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x1faa:0xc DW_TAG_structure_type
	.short	411                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x1fad:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1faf:0x5 DW_TAG_template_type_parameter
	.long	7151                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x1fb6:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	412                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x1fbd:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1fbf:0x5 DW_TAG_template_type_parameter
	.long	7155                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1fc6:0x5 DW_TAG_pointer_type
	.long	8139                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x1fcb:0xc DW_TAG_structure_type
	.short	413                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x1fce:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1fd0:0x5 DW_TAG_template_type_parameter
	.long	7155                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x1fd7:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	414                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x1fde:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1fe0:0x5 DW_TAG_template_type_parameter
	.long	170                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x1fe7:0x5 DW_TAG_pointer_type
	.long	8172                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x1fec:0xc DW_TAG_structure_type
	.short	415                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x1fef:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x1ff1:0x5 DW_TAG_template_type_parameter
	.long	170                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x1ff8:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	416                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x1fff:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2001:0x5 DW_TAG_template_type_parameter
	.long	7157                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2008:0x5 DW_TAG_pointer_type
	.long	8205                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x200d:0xc DW_TAG_structure_type
	.short	417                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2010:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2012:0x5 DW_TAG_template_type_parameter
	.long	7157                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2019:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	418                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2020:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2022:0x5 DW_TAG_template_type_parameter
	.long	174                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2029:0x5 DW_TAG_pointer_type
	.long	8238                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x202e:0xc DW_TAG_structure_type
	.short	419                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2031:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2033:0x5 DW_TAG_template_type_parameter
	.long	174                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x203a:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	420                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2041:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2043:0x5 DW_TAG_template_type_parameter
	.long	7162                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x204a:0x5 DW_TAG_pointer_type
	.long	8271                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x204f:0xc DW_TAG_structure_type
	.short	421                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2052:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2054:0x5 DW_TAG_template_type_parameter
	.long	7162                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x205b:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	422                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2062:0xd DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2064:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	34                              # Abbrev [34] 0x2069:0x5 DW_TAG_template_type_parameter
	.long	4531                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2070:0x5 DW_TAG_pointer_type
	.long	8309                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2075:0x11 DW_TAG_structure_type
	.short	423                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2078:0xd DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x207a:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	34                              # Abbrev [34] 0x207f:0x5 DW_TAG_template_type_parameter
	.long	4531                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2086:0x5 DW_TAG_pointer_type
	.long	8331                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x208b:0xc DW_TAG_structure_type
	.short	424                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x208e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2090:0x5 DW_TAG_template_type_parameter
	.long	7177                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2097:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	425                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x209e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x20a0:0x5 DW_TAG_template_type_parameter
	.long	7182                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x20a7:0x5 DW_TAG_pointer_type
	.long	8364                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x20ac:0xc DW_TAG_structure_type
	.short	426                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x20af:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x20b1:0x5 DW_TAG_template_type_parameter
	.long	7182                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x20b8:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	427                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x20bf:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x20c1:0x5 DW_TAG_template_type_parameter
	.long	7187                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x20c8:0x5 DW_TAG_pointer_type
	.long	8397                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x20cd:0xc DW_TAG_structure_type
	.short	428                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x20d0:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x20d2:0x5 DW_TAG_template_type_parameter
	.long	7187                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x20d9:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	429                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x20e0:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x20e2:0x5 DW_TAG_template_type_parameter
	.long	7192                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x20e9:0x5 DW_TAG_pointer_type
	.long	8430                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x20ee:0xc DW_TAG_structure_type
	.short	430                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x20f1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x20f3:0x5 DW_TAG_template_type_parameter
	.long	7192                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x20fa:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	431                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2101:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2103:0x5 DW_TAG_template_type_parameter
	.long	7197                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x210a:0x5 DW_TAG_pointer_type
	.long	8463                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x210f:0xc DW_TAG_structure_type
	.short	432                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2112:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2114:0x5 DW_TAG_template_type_parameter
	.long	7197                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x211b:0xc DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	433                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2122:0x4 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	35                              # Abbrev [35] 0x2124:0x1 DW_TAG_template_type_parameter
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2127:0x5 DW_TAG_pointer_type
	.long	8492                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x212c:0x8 DW_TAG_structure_type
	.short	434                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x212f:0x4 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	35                              # Abbrev [35] 0x2131:0x1 DW_TAG_template_type_parameter
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2134:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	435                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x213b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x213d:0x5 DW_TAG_template_type_parameter
	.long	7219                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2144:0x5 DW_TAG_pointer_type
	.long	8521                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2149:0xc DW_TAG_structure_type
	.short	436                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x214c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x214e:0x5 DW_TAG_template_type_parameter
	.long	7219                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2155:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	437                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x215c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x215e:0x5 DW_TAG_template_type_parameter
	.long	697                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2165:0x5 DW_TAG_pointer_type
	.long	8554                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x216a:0xc DW_TAG_structure_type
	.short	438                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x216d:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x216f:0x5 DW_TAG_template_type_parameter
	.long	697                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2176:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	439                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x217d:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x217f:0x5 DW_TAG_template_type_parameter
	.long	182                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2186:0x5 DW_TAG_pointer_type
	.long	8587                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x218b:0xc DW_TAG_structure_type
	.short	440                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x218e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2190:0x5 DW_TAG_template_type_parameter
	.long	182                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2197:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	441                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x219e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x21a0:0x5 DW_TAG_template_type_parameter
	.long	7230                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x21a7:0x5 DW_TAG_pointer_type
	.long	8620                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x21ac:0xc DW_TAG_structure_type
	.short	442                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x21af:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x21b1:0x5 DW_TAG_template_type_parameter
	.long	7230                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x21b8:0x5 DW_TAG_pointer_type
	.long	8637                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x21bd:0xc DW_TAG_structure_type
	.short	443                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x21c0:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x21c2:0x5 DW_TAG_template_type_parameter
	.long	966                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x21c9:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	444                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x21d0:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x21d2:0x5 DW_TAG_template_type_parameter
	.long	7246                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x21d9:0x5 DW_TAG_pointer_type
	.long	8670                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x21de:0xc DW_TAG_structure_type
	.short	445                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x21e1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x21e3:0x5 DW_TAG_template_type_parameter
	.long	7246                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x21ea:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	446                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x21f1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x21f3:0x5 DW_TAG_template_type_parameter
	.long	7282                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x21fa:0x5 DW_TAG_pointer_type
	.long	8703                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x21ff:0xc DW_TAG_structure_type
	.short	447                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2202:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2204:0x5 DW_TAG_template_type_parameter
	.long	7282                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x220b:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	448                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2212:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2214:0x5 DW_TAG_template_type_parameter
	.long	7293                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x221b:0x5 DW_TAG_pointer_type
	.long	8736                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2220:0xc DW_TAG_structure_type
	.short	449                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2223:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2225:0x5 DW_TAG_template_type_parameter
	.long	7293                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x222c:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	450                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2233:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2235:0x5 DW_TAG_template_type_parameter
	.long	7296                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x223c:0x5 DW_TAG_pointer_type
	.long	8769                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2241:0xc DW_TAG_structure_type
	.short	451                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2244:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2246:0x5 DW_TAG_template_type_parameter
	.long	7296                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x224d:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	452                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2254:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2256:0x5 DW_TAG_template_type_parameter
	.long	7304                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x225d:0x5 DW_TAG_pointer_type
	.long	8802                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2262:0xc DW_TAG_structure_type
	.short	453                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2265:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2267:0x5 DW_TAG_template_type_parameter
	.long	7304                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x226e:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	454                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2275:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2277:0x5 DW_TAG_template_type_parameter
	.long	7309                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x227e:0x5 DW_TAG_pointer_type
	.long	8835                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2283:0xc DW_TAG_structure_type
	.short	455                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2286:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2288:0x5 DW_TAG_template_type_parameter
	.long	7309                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x228f:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	456                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2296:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2298:0x5 DW_TAG_template_type_parameter
	.long	72                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x229f:0x5 DW_TAG_pointer_type
	.long	8868                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x22a4:0xc DW_TAG_structure_type
	.short	457                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x22a7:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x22a9:0x5 DW_TAG_template_type_parameter
	.long	72                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x22b0:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	458                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x22b7:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x22b9:0x5 DW_TAG_template_type_parameter
	.long	7319                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x22c0:0x5 DW_TAG_pointer_type
	.long	8901                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x22c5:0xc DW_TAG_structure_type
	.short	459                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x22c8:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x22ca:0x5 DW_TAG_template_type_parameter
	.long	7319                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x22d1:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	460                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x22d8:0xd DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x22da:0x5 DW_TAG_template_type_parameter
	.long	7322                            # DW_AT_type
	.byte	34                              # Abbrev [34] 0x22df:0x5 DW_TAG_template_type_parameter
	.long	7322                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x22e6:0x5 DW_TAG_pointer_type
	.long	8939                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x22eb:0x11 DW_TAG_structure_type
	.short	461                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x22ee:0xd DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x22f0:0x5 DW_TAG_template_type_parameter
	.long	7322                            # DW_AT_type
	.byte	34                              # Abbrev [34] 0x22f5:0x5 DW_TAG_template_type_parameter
	.long	7322                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x22fc:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	462                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2303:0xd DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2305:0x5 DW_TAG_template_type_parameter
	.long	7322                            # DW_AT_type
	.byte	34                              # Abbrev [34] 0x230a:0x5 DW_TAG_template_type_parameter
	.long	7327                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2311:0x5 DW_TAG_pointer_type
	.long	8982                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2316:0x11 DW_TAG_structure_type
	.short	463                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2319:0xd DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x231b:0x5 DW_TAG_template_type_parameter
	.long	7322                            # DW_AT_type
	.byte	34                              # Abbrev [34] 0x2320:0x5 DW_TAG_template_type_parameter
	.long	7327                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2327:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	464                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x232e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2330:0x5 DW_TAG_template_type_parameter
	.long	7332                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2337:0x5 DW_TAG_pointer_type
	.long	9020                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x233c:0xc DW_TAG_structure_type
	.short	465                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x233f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2341:0x5 DW_TAG_template_type_parameter
	.long	7332                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2348:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	466                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x234f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2351:0x5 DW_TAG_template_type_parameter
	.long	7337                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2358:0x5 DW_TAG_pointer_type
	.long	9053                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x235d:0xc DW_TAG_structure_type
	.short	467                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2360:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2362:0x5 DW_TAG_template_type_parameter
	.long	7337                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2369:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	468                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2370:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2372:0x5 DW_TAG_template_type_parameter
	.long	7353                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2379:0x5 DW_TAG_pointer_type
	.long	9086                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x237e:0xc DW_TAG_structure_type
	.short	469                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2381:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2383:0x5 DW_TAG_template_type_parameter
	.long	7353                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x238a:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	470                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2391:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2393:0x5 DW_TAG_template_type_parameter
	.long	7354                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x239a:0x5 DW_TAG_pointer_type
	.long	9119                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x239f:0xc DW_TAG_structure_type
	.short	471                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x23a2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x23a4:0x5 DW_TAG_template_type_parameter
	.long	7354                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x23ab:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	472                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x23b2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x23b4:0x5 DW_TAG_template_type_parameter
	.long	7359                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x23bb:0x5 DW_TAG_pointer_type
	.long	9152                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x23c0:0xc DW_TAG_structure_type
	.short	473                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x23c3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x23c5:0x5 DW_TAG_template_type_parameter
	.long	7359                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x23cc:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	474                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x23d3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x23d5:0x5 DW_TAG_template_type_parameter
	.long	971                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x23dc:0x5 DW_TAG_pointer_type
	.long	9185                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x23e1:0xc DW_TAG_structure_type
	.short	475                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x23e4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x23e6:0x5 DW_TAG_template_type_parameter
	.long	971                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x23ed:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	476                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x23f4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x23f6:0x5 DW_TAG_template_type_parameter
	.long	7364                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x23fd:0x5 DW_TAG_pointer_type
	.long	9218                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2402:0xc DW_TAG_structure_type
	.short	477                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2405:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2407:0x5 DW_TAG_template_type_parameter
	.long	7364                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x240e:0x5 DW_TAG_pointer_type
	.long	9235                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2413:0x6 DW_TAG_structure_type
	.short	478                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	50                              # Abbrev [50] 0x2416:0x2 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2419:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	479                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2420:0xd DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2422:0x5 DW_TAG_template_type_parameter
	.long	7347                            # DW_AT_type
	.byte	34                              # Abbrev [34] 0x2427:0x5 DW_TAG_template_type_parameter
	.long	7347                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x242e:0x5 DW_TAG_pointer_type
	.long	9267                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2433:0x11 DW_TAG_structure_type
	.short	480                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2436:0xd DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2438:0x5 DW_TAG_template_type_parameter
	.long	7347                            # DW_AT_type
	.byte	34                              # Abbrev [34] 0x243d:0x5 DW_TAG_template_type_parameter
	.long	7347                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2444:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	481                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x244b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x244d:0x5 DW_TAG_template_type_parameter
	.long	7369                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2454:0x5 DW_TAG_pointer_type
	.long	9305                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2459:0xc DW_TAG_structure_type
	.short	482                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x245c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x245e:0x5 DW_TAG_template_type_parameter
	.long	7369                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2465:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	483                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x246c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x246e:0x5 DW_TAG_template_type_parameter
	.long	7390                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2475:0x5 DW_TAG_pointer_type
	.long	9338                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x247a:0xc DW_TAG_structure_type
	.short	484                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x247d:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x247f:0x5 DW_TAG_template_type_parameter
	.long	7390                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2486:0x5 DW_TAG_pointer_type
	.long	3851                            # DW_AT_type
	.byte	76                              # Abbrev [76] 0x248b:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	486                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2492:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2494:0x5 DW_TAG_template_type_parameter
	.long	976                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x249b:0x5 DW_TAG_pointer_type
	.long	9376                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x24a0:0xc DW_TAG_structure_type
	.short	487                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x24a3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x24a5:0x5 DW_TAG_template_type_parameter
	.long	976                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x24ac:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	488                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x24b3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x24b5:0x5 DW_TAG_template_type_parameter
	.long	7401                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x24bc:0x5 DW_TAG_pointer_type
	.long	9409                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x24c1:0xc DW_TAG_structure_type
	.short	489                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x24c4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x24c6:0x5 DW_TAG_template_type_parameter
	.long	7401                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x24cd:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	490                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x24d4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x24d6:0x5 DW_TAG_template_type_parameter
	.long	7406                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x24dd:0x5 DW_TAG_pointer_type
	.long	9442                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x24e2:0xc DW_TAG_structure_type
	.short	491                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x24e5:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x24e7:0x5 DW_TAG_template_type_parameter
	.long	7406                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x24ee:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	492                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x24f5:0xd DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x24f7:0x5 DW_TAG_template_type_parameter
	.long	7177                            # DW_AT_type
	.byte	34                              # Abbrev [34] 0x24fc:0x5 DW_TAG_template_type_parameter
	.long	7411                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2503:0x5 DW_TAG_pointer_type
	.long	9480                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2508:0x11 DW_TAG_structure_type
	.short	493                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x250b:0xd DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x250d:0x5 DW_TAG_template_type_parameter
	.long	7177                            # DW_AT_type
	.byte	34                              # Abbrev [34] 0x2512:0x5 DW_TAG_template_type_parameter
	.long	7411                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2519:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	494                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2520:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2522:0x5 DW_TAG_template_type_parameter
	.long	7416                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2529:0x5 DW_TAG_pointer_type
	.long	9518                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x252e:0xc DW_TAG_structure_type
	.short	495                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2531:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2533:0x5 DW_TAG_template_type_parameter
	.long	7416                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x253a:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	496                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2541:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2543:0x5 DW_TAG_template_type_parameter
	.long	7430                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x254a:0x5 DW_TAG_pointer_type
	.long	9551                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x254f:0xc DW_TAG_structure_type
	.short	497                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2552:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2554:0x5 DW_TAG_template_type_parameter
	.long	7430                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x255b:0x1a DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	498                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2562:0x12 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2564:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	34                              # Abbrev [34] 0x2569:0x5 DW_TAG_template_type_parameter
	.long	505                             # DW_AT_type
	.byte	34                              # Abbrev [34] 0x256e:0x5 DW_TAG_template_type_parameter
	.long	7435                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2575:0x5 DW_TAG_pointer_type
	.long	9594                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x257a:0x16 DW_TAG_structure_type
	.short	499                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x257d:0x12 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x257f:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	34                              # Abbrev [34] 0x2584:0x5 DW_TAG_template_type_parameter
	.long	505                             # DW_AT_type
	.byte	34                              # Abbrev [34] 0x2589:0x5 DW_TAG_template_type_parameter
	.long	7435                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2590:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	500                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2597:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2599:0x5 DW_TAG_template_type_parameter
	.long	7440                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x25a0:0x5 DW_TAG_pointer_type
	.long	9637                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x25a5:0xc DW_TAG_structure_type
	.short	501                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x25a8:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x25aa:0x5 DW_TAG_template_type_parameter
	.long	7440                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x25b1:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	502                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x25b8:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x25ba:0x5 DW_TAG_template_type_parameter
	.long	7452                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x25c1:0x5 DW_TAG_pointer_type
	.long	9670                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x25c6:0xc DW_TAG_structure_type
	.short	503                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x25c9:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x25cb:0x5 DW_TAG_template_type_parameter
	.long	7452                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x25d2:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	504                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x25d9:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x25db:0x5 DW_TAG_template_type_parameter
	.long	7462                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x25e2:0x5 DW_TAG_pointer_type
	.long	9703                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x25e7:0xc DW_TAG_structure_type
	.short	505                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x25ea:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x25ec:0x5 DW_TAG_template_type_parameter
	.long	7462                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x25f3:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	506                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x25fa:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x25fc:0x5 DW_TAG_template_type_parameter
	.long	7468                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2603:0x5 DW_TAG_pointer_type
	.long	9736                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2608:0xc DW_TAG_structure_type
	.short	507                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x260b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x260d:0x5 DW_TAG_template_type_parameter
	.long	7468                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2614:0x5 DW_TAG_pointer_type
	.long	206                             # DW_AT_type
	.byte	76                              # Abbrev [76] 0x2619:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	508                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2620:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2622:0x5 DW_TAG_template_type_parameter
	.long	7484                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2629:0x5 DW_TAG_pointer_type
	.long	9774                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x262e:0xc DW_TAG_structure_type
	.short	509                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2631:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2633:0x5 DW_TAG_template_type_parameter
	.long	7484                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x263a:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	510                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2641:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2643:0x5 DW_TAG_template_type_parameter
	.long	7510                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x264a:0x5 DW_TAG_pointer_type
	.long	9807                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x264f:0xc DW_TAG_structure_type
	.short	511                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2652:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2654:0x5 DW_TAG_template_type_parameter
	.long	7510                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x265b:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	512                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2662:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2664:0x5 DW_TAG_template_type_parameter
	.long	7536                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x266b:0x5 DW_TAG_pointer_type
	.long	9840                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2670:0xc DW_TAG_structure_type
	.short	513                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2673:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2675:0x5 DW_TAG_template_type_parameter
	.long	7536                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x267c:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	514                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2683:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2685:0x5 DW_TAG_template_type_parameter
	.long	7562                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x268c:0x5 DW_TAG_pointer_type
	.long	9873                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2691:0xc DW_TAG_structure_type
	.short	515                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2694:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2696:0x5 DW_TAG_template_type_parameter
	.long	7562                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x269d:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	516                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x26a4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x26a6:0x5 DW_TAG_template_type_parameter
	.long	7567                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x26ad:0x5 DW_TAG_pointer_type
	.long	9906                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x26b2:0xc DW_TAG_structure_type
	.short	517                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x26b5:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x26b7:0x5 DW_TAG_template_type_parameter
	.long	7567                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x26be:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	518                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x26c5:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x26c7:0x5 DW_TAG_template_type_parameter
	.long	7589                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x26ce:0x5 DW_TAG_pointer_type
	.long	9939                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x26d3:0xc DW_TAG_structure_type
	.short	519                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x26d6:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x26d8:0x5 DW_TAG_template_type_parameter
	.long	7589                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x26df:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	520                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x26e6:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x26e8:0x5 DW_TAG_template_type_parameter
	.long	7595                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x26ef:0x5 DW_TAG_pointer_type
	.long	9972                            # DW_AT_type
	.byte	91                              # Abbrev [91] 0x26f4:0xc DW_TAG_structure_type
	.short	521                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x26f7:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x26f9:0x5 DW_TAG_template_type_parameter
	.long	7595                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2700:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	522                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2707:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2709:0x5 DW_TAG_template_type_parameter
	.long	7601                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2710:0x5 DW_TAG_pointer_type
	.long	10005                           # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2715:0xc DW_TAG_structure_type
	.short	523                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2718:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x271a:0x5 DW_TAG_template_type_parameter
	.long	7601                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2721:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	524                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2728:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x272a:0x5 DW_TAG_template_type_parameter
	.long	7611                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2731:0x5 DW_TAG_pointer_type
	.long	10038                           # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2736:0xc DW_TAG_structure_type
	.short	525                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2739:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x273b:0x5 DW_TAG_template_type_parameter
	.long	7611                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2742:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	526                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2749:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x274b:0x5 DW_TAG_template_type_parameter
	.long	7628                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2752:0x5 DW_TAG_pointer_type
	.long	10071                           # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2757:0xc DW_TAG_structure_type
	.short	527                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x275a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x275c:0x5 DW_TAG_template_type_parameter
	.long	7628                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2763:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	528                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x276a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x276c:0x5 DW_TAG_template_type_parameter
	.long	7633                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2773:0x5 DW_TAG_pointer_type
	.long	10104                           # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2778:0xc DW_TAG_structure_type
	.short	529                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x277b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x277d:0x5 DW_TAG_template_type_parameter
	.long	7633                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2784:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	530                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x278b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x278d:0x5 DW_TAG_template_type_parameter
	.long	7664                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2794:0x5 DW_TAG_pointer_type
	.long	10137                           # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2799:0xc DW_TAG_structure_type
	.short	531                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x279c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x279e:0x5 DW_TAG_template_type_parameter
	.long	7664                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x27a5:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	532                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x27ac:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x27ae:0x5 DW_TAG_template_type_parameter
	.long	7687                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x27b5:0x5 DW_TAG_pointer_type
	.long	10170                           # DW_AT_type
	.byte	91                              # Abbrev [91] 0x27ba:0xc DW_TAG_structure_type
	.short	533                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x27bd:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x27bf:0x5 DW_TAG_template_type_parameter
	.long	7687                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x27c6:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	534                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x27cd:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x27cf:0x5 DW_TAG_template_type_parameter
	.long	7354                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x27d6:0x5 DW_TAG_pointer_type
	.long	10203                           # DW_AT_type
	.byte	91                              # Abbrev [91] 0x27db:0xc DW_TAG_structure_type
	.short	535                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x27de:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x27e0:0x5 DW_TAG_template_type_parameter
	.long	7354                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x27e7:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	536                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x27ee:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x27f0:0x5 DW_TAG_template_type_parameter
	.long	7699                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x27f7:0x5 DW_TAG_pointer_type
	.long	10236                           # DW_AT_type
	.byte	91                              # Abbrev [91] 0x27fc:0xc DW_TAG_structure_type
	.short	537                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x27ff:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2801:0x5 DW_TAG_template_type_parameter
	.long	7699                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2808:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	538                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x280f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2811:0x5 DW_TAG_template_type_parameter
	.long	7706                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2818:0x5 DW_TAG_pointer_type
	.long	10269                           # DW_AT_type
	.byte	91                              # Abbrev [91] 0x281d:0xc DW_TAG_structure_type
	.short	539                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2820:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2822:0x5 DW_TAG_template_type_parameter
	.long	7706                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x2829:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	540                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2830:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2832:0x5 DW_TAG_template_type_parameter
	.long	7718                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2839:0x5 DW_TAG_pointer_type
	.long	10302                           # DW_AT_type
	.byte	91                              # Abbrev [91] 0x283e:0xc DW_TAG_structure_type
	.short	541                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2841:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2843:0x5 DW_TAG_template_type_parameter
	.long	7718                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x284a:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	542                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2851:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2853:0x5 DW_TAG_template_type_parameter
	.long	7725                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x285a:0x5 DW_TAG_pointer_type
	.long	10335                           # DW_AT_type
	.byte	91                              # Abbrev [91] 0x285f:0xc DW_TAG_structure_type
	.short	543                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2862:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2864:0x5 DW_TAG_template_type_parameter
	.long	7725                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x286b:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	544                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2872:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2874:0x5 DW_TAG_template_type_parameter
	.long	7730                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x287b:0x5 DW_TAG_pointer_type
	.long	10368                           # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2880:0xc DW_TAG_structure_type
	.short	545                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2883:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2885:0x5 DW_TAG_template_type_parameter
	.long	7730                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x288c:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	546                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x2893:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x2895:0x5 DW_TAG_template_type_parameter
	.long	7740                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x289c:0x5 DW_TAG_pointer_type
	.long	10401                           # DW_AT_type
	.byte	91                              # Abbrev [91] 0x28a1:0xc DW_TAG_structure_type
	.short	547                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x28a4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x28a6:0x5 DW_TAG_template_type_parameter
	.long	7740                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x28ad:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	548                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x28b4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x28b6:0x5 DW_TAG_template_type_parameter
	.long	7762                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x28bd:0x5 DW_TAG_pointer_type
	.long	10434                           # DW_AT_type
	.byte	91                              # Abbrev [91] 0x28c2:0xc DW_TAG_structure_type
	.short	549                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x28c5:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x28c7:0x5 DW_TAG_template_type_parameter
	.long	7762                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x28ce:0x5 DW_TAG_pointer_type
	.long	6991                            # DW_AT_type
	.byte	76                              # Abbrev [76] 0x28d3:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	486                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x28da:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x28dc:0x5 DW_TAG_template_type_parameter
	.long	7043                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x28e3:0x5 DW_TAG_pointer_type
	.long	10472                           # DW_AT_type
	.byte	91                              # Abbrev [91] 0x28e8:0xc DW_TAG_structure_type
	.short	487                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x28eb:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x28ed:0x5 DW_TAG_template_type_parameter
	.long	7043                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x28f4:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	550                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	33                              # Abbrev [33] 0x28fb:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x28fd:0x5 DW_TAG_template_type_parameter
	.long	7771                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	55                              # Abbrev [55] 0x2904:0x5 DW_TAG_pointer_type
	.long	10505                           # DW_AT_type
	.byte	91                              # Abbrev [91] 0x2909:0xc DW_TAG_structure_type
	.short	551                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x290c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	86                              # DW_AT_name
	.byte	34                              # Abbrev [34] 0x290e:0x5 DW_TAG_template_type_parameter
	.long	7771                            # DW_AT_type
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
	.uleb128 .Lfunc_begin134-.Lfunc_begin0  #   starting offset
	.uleb128 .Lfunc_end135-.Lfunc_begin0    #   ending offset
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
	.ascii	"\211\001"                      #   start index
	.uleb128 .Lfunc_end136-.Lfunc_begin136  #   length
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	2212                            # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 15.0.0 (git@github.com:llvm/llvm-project.git 9980a3f8318cbfe2305b4ffc58a7256da1a3891d)" # string offset=0
.Linfo_string1:
	.asciz	"cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp" # string offset=101
.Linfo_string2:
	.asciz	"/usr/local/google/home/blaikie/dev/llvm/src" # string offset=188
.Linfo_string3:
	.asciz	"i"                             # string offset=232
.Linfo_string4:
	.asciz	"int"                           # string offset=234
.Linfo_string5:
	.asciz	"unsigned int"                  # string offset=238
.Linfo_string6:
	.asciz	"LocalEnum1"                    # string offset=251
.Linfo_string7:
	.asciz	"LocalEnum"                     # string offset=262
.Linfo_string8:
	.asciz	"ns"                            # string offset=272
.Linfo_string9:
	.asciz	"Enumerator1"                   # string offset=275
.Linfo_string10:
	.asciz	"Enumerator2"                   # string offset=287
.Linfo_string11:
	.asciz	"Enumerator3"                   # string offset=299
.Linfo_string12:
	.asciz	"Enumeration"                   # string offset=311
.Linfo_string13:
	.asciz	"EnumerationClass"              # string offset=323
.Linfo_string14:
	.asciz	"unsigned char"                 # string offset=340
.Linfo_string15:
	.asciz	"kNeg"                          # string offset=354
.Linfo_string16:
	.asciz	"EnumerationSmall"              # string offset=359
.Linfo_string17:
	.asciz	"AnonEnum1"                     # string offset=376
.Linfo_string18:
	.asciz	"AnonEnum2"                     # string offset=386
.Linfo_string19:
	.asciz	"AnonEnum3"                     # string offset=396
.Linfo_string20:
	.asciz	"T"                             # string offset=406
.Linfo_string21:
	.asciz	"bool"                          # string offset=408
.Linfo_string22:
	.asciz	"b"                             # string offset=413
.Linfo_string23:
	.asciz	"_STNt3|<int, false>"           # string offset=415
.Linfo_string24:
	.asciz	"t10"                           # string offset=435
.Linfo_string25:
	.asciz	"std"                           # string offset=439
.Linfo_string26:
	.asciz	"signed char"                   # string offset=443
.Linfo_string27:
	.asciz	"__int8_t"                      # string offset=455
.Linfo_string28:
	.asciz	"int8_t"                        # string offset=464
.Linfo_string29:
	.asciz	"short"                         # string offset=471
.Linfo_string30:
	.asciz	"__int16_t"                     # string offset=477
.Linfo_string31:
	.asciz	"int16_t"                       # string offset=487
.Linfo_string32:
	.asciz	"__int32_t"                     # string offset=495
.Linfo_string33:
	.asciz	"int32_t"                       # string offset=505
.Linfo_string34:
	.asciz	"long"                          # string offset=513
.Linfo_string35:
	.asciz	"__int64_t"                     # string offset=518
.Linfo_string36:
	.asciz	"int64_t"                       # string offset=528
.Linfo_string37:
	.asciz	"int_fast8_t"                   # string offset=536
.Linfo_string38:
	.asciz	"int_fast16_t"                  # string offset=548
.Linfo_string39:
	.asciz	"int_fast32_t"                  # string offset=561
.Linfo_string40:
	.asciz	"int_fast64_t"                  # string offset=574
.Linfo_string41:
	.asciz	"__int_least8_t"                # string offset=587
.Linfo_string42:
	.asciz	"int_least8_t"                  # string offset=602
.Linfo_string43:
	.asciz	"__int_least16_t"               # string offset=615
.Linfo_string44:
	.asciz	"int_least16_t"                 # string offset=631
.Linfo_string45:
	.asciz	"__int_least32_t"               # string offset=645
.Linfo_string46:
	.asciz	"int_least32_t"                 # string offset=661
.Linfo_string47:
	.asciz	"__int_least64_t"               # string offset=675
.Linfo_string48:
	.asciz	"int_least64_t"                 # string offset=691
.Linfo_string49:
	.asciz	"__intmax_t"                    # string offset=705
.Linfo_string50:
	.asciz	"intmax_t"                      # string offset=716
.Linfo_string51:
	.asciz	"intptr_t"                      # string offset=725
.Linfo_string52:
	.asciz	"__uint8_t"                     # string offset=734
.Linfo_string53:
	.asciz	"uint8_t"                       # string offset=744
.Linfo_string54:
	.asciz	"unsigned short"                # string offset=752
.Linfo_string55:
	.asciz	"__uint16_t"                    # string offset=767
.Linfo_string56:
	.asciz	"uint16_t"                      # string offset=778
.Linfo_string57:
	.asciz	"__uint32_t"                    # string offset=787
.Linfo_string58:
	.asciz	"uint32_t"                      # string offset=798
.Linfo_string59:
	.asciz	"unsigned long"                 # string offset=807
.Linfo_string60:
	.asciz	"__uint64_t"                    # string offset=821
.Linfo_string61:
	.asciz	"uint64_t"                      # string offset=832
.Linfo_string62:
	.asciz	"uint_fast8_t"                  # string offset=841
.Linfo_string63:
	.asciz	"uint_fast16_t"                 # string offset=854
.Linfo_string64:
	.asciz	"uint_fast32_t"                 # string offset=868
.Linfo_string65:
	.asciz	"uint_fast64_t"                 # string offset=882
.Linfo_string66:
	.asciz	"__uint_least8_t"               # string offset=896
.Linfo_string67:
	.asciz	"uint_least8_t"                 # string offset=912
.Linfo_string68:
	.asciz	"__uint_least16_t"              # string offset=926
.Linfo_string69:
	.asciz	"uint_least16_t"                # string offset=943
.Linfo_string70:
	.asciz	"__uint_least32_t"              # string offset=958
.Linfo_string71:
	.asciz	"uint_least32_t"                # string offset=975
.Linfo_string72:
	.asciz	"__uint_least64_t"              # string offset=990
.Linfo_string73:
	.asciz	"uint_least64_t"                # string offset=1007
.Linfo_string74:
	.asciz	"__uintmax_t"                   # string offset=1022
.Linfo_string75:
	.asciz	"uintmax_t"                     # string offset=1034
.Linfo_string76:
	.asciz	"uintptr_t"                     # string offset=1044
.Linfo_string77:
	.asciz	"t6"                            # string offset=1054
.Linfo_string78:
	.asciz	"_ZN2t6lsIiEEvi"                # string offset=1057
.Linfo_string79:
	.asciz	"operator<<<int>"               # string offset=1072
.Linfo_string80:
	.asciz	"_ZN2t6ltIiEEvi"                # string offset=1088
.Linfo_string81:
	.asciz	"operator<<int>"                # string offset=1103
.Linfo_string82:
	.asciz	"_ZN2t6leIiEEvi"                # string offset=1118
.Linfo_string83:
	.asciz	"operator<=<int>"               # string offset=1133
.Linfo_string84:
	.asciz	"_ZN2t6cvP2t1IJfEEIiEEv"        # string offset=1149
.Linfo_string85:
	.asciz	"operator t1<float> *<int>"     # string offset=1172
.Linfo_string86:
	.asciz	"Ts"                            # string offset=1198
.Linfo_string87:
	.asciz	"float"                         # string offset=1201
.Linfo_string88:
	.asciz	"_STNt1|<float>"                # string offset=1207
.Linfo_string89:
	.asciz	"_ZN2t6miIiEEvi"                # string offset=1222
.Linfo_string90:
	.asciz	"operator-<int>"                # string offset=1237
.Linfo_string91:
	.asciz	"_ZN2t6mlIiEEvi"                # string offset=1252
.Linfo_string92:
	.asciz	"operator*<int>"                # string offset=1267
.Linfo_string93:
	.asciz	"_ZN2t6dvIiEEvi"                # string offset=1282
.Linfo_string94:
	.asciz	"operator/<int>"                # string offset=1297
.Linfo_string95:
	.asciz	"_ZN2t6rmIiEEvi"                # string offset=1312
.Linfo_string96:
	.asciz	"operator%<int>"                # string offset=1327
.Linfo_string97:
	.asciz	"_ZN2t6eoIiEEvi"                # string offset=1342
.Linfo_string98:
	.asciz	"operator^<int>"                # string offset=1357
.Linfo_string99:
	.asciz	"_ZN2t6anIiEEvi"                # string offset=1372
.Linfo_string100:
	.asciz	"operator&<int>"                # string offset=1387
.Linfo_string101:
	.asciz	"_ZN2t6orIiEEvi"                # string offset=1402
.Linfo_string102:
	.asciz	"operator|<int>"                # string offset=1417
.Linfo_string103:
	.asciz	"_ZN2t6coIiEEvv"                # string offset=1432
.Linfo_string104:
	.asciz	"operator~<int>"                # string offset=1447
.Linfo_string105:
	.asciz	"_ZN2t6ntIiEEvv"                # string offset=1462
.Linfo_string106:
	.asciz	"operator!<int>"                # string offset=1477
.Linfo_string107:
	.asciz	"_ZN2t6aSIiEEvi"                # string offset=1492
.Linfo_string108:
	.asciz	"operator=<int>"                # string offset=1507
.Linfo_string109:
	.asciz	"_ZN2t6gtIiEEvi"                # string offset=1522
.Linfo_string110:
	.asciz	"operator><int>"                # string offset=1537
.Linfo_string111:
	.asciz	"_ZN2t6cmIiEEvi"                # string offset=1552
.Linfo_string112:
	.asciz	"operator,<int>"                # string offset=1567
.Linfo_string113:
	.asciz	"_ZN2t6clIiEEvv"                # string offset=1582
.Linfo_string114:
	.asciz	"operator()<int>"               # string offset=1597
.Linfo_string115:
	.asciz	"_ZN2t6ixIiEEvi"                # string offset=1613
.Linfo_string116:
	.asciz	"operator[]<int>"               # string offset=1628
.Linfo_string117:
	.asciz	"_ZN2t6ssIiEEvi"                # string offset=1644
.Linfo_string118:
	.asciz	"operator<=><int>"              # string offset=1659
.Linfo_string119:
	.asciz	"_ZN2t6nwIiEEPvmT_"             # string offset=1676
.Linfo_string120:
	.asciz	"operator new<int>"             # string offset=1694
.Linfo_string121:
	.asciz	"size_t"                        # string offset=1712
.Linfo_string122:
	.asciz	"_ZN2t6naIiEEPvmT_"             # string offset=1719
.Linfo_string123:
	.asciz	"operator new[]<int>"           # string offset=1737
.Linfo_string124:
	.asciz	"_ZN2t6dlIiEEvPvT_"             # string offset=1757
.Linfo_string125:
	.asciz	"operator delete<int>"          # string offset=1775
.Linfo_string126:
	.asciz	"_ZN2t6daIiEEvPvT_"             # string offset=1796
.Linfo_string127:
	.asciz	"operator delete[]<int>"        # string offset=1814
.Linfo_string128:
	.asciz	"_ZN2t6awIiEEiv"                # string offset=1837
.Linfo_string129:
	.asciz	"operator co_await<int>"        # string offset=1852
.Linfo_string130:
	.asciz	"_STNt10|<void>"                # string offset=1875
.Linfo_string131:
	.asciz	"_ZN2t83memEv"                  # string offset=1890
.Linfo_string132:
	.asciz	"mem"                           # string offset=1903
.Linfo_string133:
	.asciz	"t8"                            # string offset=1907
.Linfo_string134:
	.asciz	"_Zli5_suffy"                   # string offset=1910
.Linfo_string135:
	.asciz	"operator\"\"_suff"             # string offset=1922
.Linfo_string136:
	.asciz	"main"                          # string offset=1938
.Linfo_string137:
	.asciz	"_Z2f1IJiEEvv"                  # string offset=1943
.Linfo_string138:
	.asciz	"_STNf1|<int>"                  # string offset=1956
.Linfo_string139:
	.asciz	"_Z2f1IJfEEvv"                  # string offset=1969
.Linfo_string140:
	.asciz	"_STNf1|<float>"                # string offset=1982
.Linfo_string141:
	.asciz	"_Z2f1IJbEEvv"                  # string offset=1997
.Linfo_string142:
	.asciz	"_STNf1|<bool>"                 # string offset=2010
.Linfo_string143:
	.asciz	"double"                        # string offset=2024
.Linfo_string144:
	.asciz	"_Z2f1IJdEEvv"                  # string offset=2031
.Linfo_string145:
	.asciz	"_STNf1|<double>"               # string offset=2044
.Linfo_string146:
	.asciz	"_Z2f1IJlEEvv"                  # string offset=2060
.Linfo_string147:
	.asciz	"_STNf1|<long>"                 # string offset=2073
.Linfo_string148:
	.asciz	"_Z2f1IJsEEvv"                  # string offset=2087
.Linfo_string149:
	.asciz	"_STNf1|<short>"                # string offset=2100
.Linfo_string150:
	.asciz	"_Z2f1IJjEEvv"                  # string offset=2115
.Linfo_string151:
	.asciz	"_STNf1|<unsigned int>"         # string offset=2128
.Linfo_string152:
	.asciz	"unsigned long long"            # string offset=2150
.Linfo_string153:
	.asciz	"_Z2f1IJyEEvv"                  # string offset=2169
.Linfo_string154:
	.asciz	"_STNf1|<unsigned long long>"   # string offset=2182
.Linfo_string155:
	.asciz	"long long"                     # string offset=2210
.Linfo_string156:
	.asciz	"_Z2f1IJxEEvv"                  # string offset=2220
.Linfo_string157:
	.asciz	"_STNf1|<long long>"            # string offset=2233
.Linfo_string158:
	.asciz	"udt"                           # string offset=2252
.Linfo_string159:
	.asciz	"_Z2f1IJ3udtEEvv"               # string offset=2256
.Linfo_string160:
	.asciz	"_STNf1|<udt>"                  # string offset=2272
.Linfo_string161:
	.asciz	"_Z2f1IJN2ns3udtEEEvv"          # string offset=2285
.Linfo_string162:
	.asciz	"_STNf1|<ns::udt>"              # string offset=2306
.Linfo_string163:
	.asciz	"_Z2f1IJPN2ns3udtEEEvv"         # string offset=2323
.Linfo_string164:
	.asciz	"_STNf1|<ns::udt *>"            # string offset=2345
.Linfo_string165:
	.asciz	"inner"                         # string offset=2364
.Linfo_string166:
	.asciz	"_Z2f1IJN2ns5inner3udtEEEvv"    # string offset=2370
.Linfo_string167:
	.asciz	"_STNf1|<ns::inner::udt>"       # string offset=2397
.Linfo_string168:
	.asciz	"_STNt1|<int>"                  # string offset=2421
.Linfo_string169:
	.asciz	"_Z2f1IJ2t1IJiEEEEvv"           # string offset=2434
.Linfo_string170:
	.asciz	"_STNf1|<t1<int> >"             # string offset=2454
.Linfo_string171:
	.asciz	"_Z2f1IJifEEvv"                 # string offset=2472
.Linfo_string172:
	.asciz	"_STNf1|<int, float>"           # string offset=2486
.Linfo_string173:
	.asciz	"_Z2f1IJPiEEvv"                 # string offset=2506
.Linfo_string174:
	.asciz	"_STNf1|<int *>"                # string offset=2520
.Linfo_string175:
	.asciz	"_Z2f1IJRiEEvv"                 # string offset=2535
.Linfo_string176:
	.asciz	"_STNf1|<int &>"                # string offset=2549
.Linfo_string177:
	.asciz	"_Z2f1IJOiEEvv"                 # string offset=2564
.Linfo_string178:
	.asciz	"_STNf1|<int &&>"               # string offset=2578
.Linfo_string179:
	.asciz	"_Z2f1IJKiEEvv"                 # string offset=2594
.Linfo_string180:
	.asciz	"_STNf1|<const int>"            # string offset=2608
.Linfo_string181:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=2627
.Linfo_string182:
	.asciz	"_Z2f1IJA3_iEEvv"               # string offset=2647
.Linfo_string183:
	.asciz	"_STNf1|<int[3]>"               # string offset=2663
.Linfo_string184:
	.asciz	"_Z2f1IJvEEvv"                  # string offset=2679
.Linfo_string185:
	.asciz	"_STNf1|<void>"                 # string offset=2692
.Linfo_string186:
	.asciz	"outer_class"                   # string offset=2706
.Linfo_string187:
	.asciz	"inner_class"                   # string offset=2718
.Linfo_string188:
	.asciz	"_Z2f1IJN11outer_class11inner_classEEEvv" # string offset=2730
.Linfo_string189:
	.asciz	"_STNf1|<outer_class::inner_class>" # string offset=2770
.Linfo_string190:
	.asciz	"_Z2f1IJmEEvv"                  # string offset=2804
.Linfo_string191:
	.asciz	"_STNf1|<unsigned long>"        # string offset=2817
.Linfo_string192:
	.asciz	"_Z2f2ILb1ELi3EEvv"             # string offset=2840
.Linfo_string193:
	.asciz	"_STNf2|<true, 3>"              # string offset=2858
.Linfo_string194:
	.asciz	"A"                             # string offset=2875
.Linfo_string195:
	.asciz	"_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv" # string offset=2877
.Linfo_string196:
	.asciz	"_STNf3|<ns::Enumeration, ns::Enumerator2, (ns::Enumeration)2>" # string offset=2919
.Linfo_string197:
	.asciz	"_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv" # string offset=2981
.Linfo_string198:
	.asciz	"_STNf3|<ns::EnumerationClass, ns::EnumerationClass::Enumerator2, (ns::EnumerationClass)2>" # string offset=3028
.Linfo_string199:
	.asciz	"_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv" # string offset=3118
.Linfo_string200:
	.asciz	"_STNf3|<ns::EnumerationSmall, ns::kNeg>" # string offset=3161
.Linfo_string201:
	.asciz	"_Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv" # string offset=3201
.Linfo_string202:
	.asciz	"f3<ns::(unnamed enum at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1), ns::AnonEnum2, (ns::(unnamed enum at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1))2>" # string offset=3234
.Linfo_string203:
	.asciz	"_Z2f3IN12_GLOBAL__N_19LocalEnumEJLS1_0EEEvv" # string offset=3485
.Linfo_string204:
	.asciz	"f3<(anonymous namespace)::LocalEnum, (anonymous namespace)::LocalEnum1>" # string offset=3529
.Linfo_string205:
	.asciz	"_Z2f3IPiJXadL_Z1iEEEEvv"       # string offset=3601
.Linfo_string206:
	.asciz	"f3<int *, &i>"                 # string offset=3625
.Linfo_string207:
	.asciz	"_Z2f3IPiJLS0_0EEEvv"           # string offset=3639
.Linfo_string208:
	.asciz	"f3<int *, nullptr>"            # string offset=3659
.Linfo_string209:
	.asciz	"_Z2f3ImJLm1EEEvv"              # string offset=3678
.Linfo_string210:
	.asciz	"_STNf3|<unsigned long, 1UL>"   # string offset=3695
.Linfo_string211:
	.asciz	"_Z2f3IyJLy1EEEvv"              # string offset=3723
.Linfo_string212:
	.asciz	"_STNf3|<unsigned long long, 1ULL>" # string offset=3740
.Linfo_string213:
	.asciz	"_Z2f3IlJLl1EEEvv"              # string offset=3774
.Linfo_string214:
	.asciz	"_STNf3|<long, 1L>"             # string offset=3791
.Linfo_string215:
	.asciz	"_Z2f3IjJLj1EEEvv"              # string offset=3809
.Linfo_string216:
	.asciz	"_STNf3|<unsigned int, 1U>"     # string offset=3826
.Linfo_string217:
	.asciz	"_Z2f3IsJLs1EEEvv"              # string offset=3852
.Linfo_string218:
	.asciz	"_STNf3|<short, (short)1>"      # string offset=3869
.Linfo_string219:
	.asciz	"_Z2f3IhJLh0EEEvv"              # string offset=3894
.Linfo_string220:
	.asciz	"_STNf3|<unsigned char, (unsigned char)'\\x00'>" # string offset=3911
.Linfo_string221:
	.asciz	"_Z2f3IaJLa0EEEvv"              # string offset=3957
.Linfo_string222:
	.asciz	"_STNf3|<signed char, (signed char)'\\x00'>" # string offset=3974
.Linfo_string223:
	.asciz	"_Z2f3ItJLt1ELt2EEEvv"          # string offset=4016
.Linfo_string224:
	.asciz	"_STNf3|<unsigned short, (unsigned short)1, (unsigned short)2>" # string offset=4037
.Linfo_string225:
	.asciz	"char"                          # string offset=4099
.Linfo_string226:
	.asciz	"_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv" # string offset=4104
.Linfo_string227:
	.asciz	"_STNf3|<char, '\\x00', '\\x01', '\\x06', '\\a', '\\r', '\\x0e', '\\x1f', ' ', '!', '\\x7f', '\\x80'>" # string offset=4171
.Linfo_string228:
	.asciz	"__int128"                      # string offset=4263
.Linfo_string229:
	.asciz	"_Z2f3InJLn18446744073709551614EEEvv" # string offset=4272
.Linfo_string230:
	.asciz	"f3<__int128, (__int128)18446744073709551614>" # string offset=4308
.Linfo_string231:
	.asciz	"_Z2f4IjLj3EEvv"                # string offset=4353
.Linfo_string232:
	.asciz	"_STNf4|<unsigned int, 3U>"     # string offset=4368
.Linfo_string233:
	.asciz	"_Z2f1IJ2t3IiLb0EEEEvv"         # string offset=4394
.Linfo_string234:
	.asciz	"_STNf1|<t3<int, false> >"      # string offset=4416
.Linfo_string235:
	.asciz	"_STNt3|<t3<int, false>, false>" # string offset=4441
.Linfo_string236:
	.asciz	"_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv" # string offset=4472
.Linfo_string237:
	.asciz	"_STNf1|<t3<t3<int, false>, false> >" # string offset=4503
.Linfo_string238:
	.asciz	"_Z2f1IJZ4mainE3$_1EEvv"        # string offset=4539
.Linfo_string239:
	.asciz	"f1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:184:12)>" # string offset=4562
.Linfo_string240:
	.asciz	"t3<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:184:12), false>" # string offset=4672
.Linfo_string241:
	.asciz	"t3<t3<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:184:12), false>, false>" # string offset=4789
.Linfo_string242:
	.asciz	"_Z2f1IJ2t3IS0_IZ4mainE3$_1Lb0EELb0EEEEvv" # string offset=4917
.Linfo_string243:
	.asciz	"f1<t3<t3<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:184:12), false>, false> >" # string offset=4958
.Linfo_string244:
	.asciz	"_Z2f1IJFifEEEvv"               # string offset=5091
.Linfo_string245:
	.asciz	"_STNf1|<int (float)>"          # string offset=5107
.Linfo_string246:
	.asciz	"_Z2f1IJFvzEEEvv"               # string offset=5128
.Linfo_string247:
	.asciz	"_STNf1|<void (...)>"           # string offset=5144
.Linfo_string248:
	.asciz	"_Z2f1IJFvizEEEvv"              # string offset=5164
.Linfo_string249:
	.asciz	"_STNf1|<void (int, ...)>"      # string offset=5181
.Linfo_string250:
	.asciz	"_Z2f1IJRKiEEvv"                # string offset=5206
.Linfo_string251:
	.asciz	"_STNf1|<const int &>"          # string offset=5221
.Linfo_string252:
	.asciz	"_Z2f1IJRPKiEEvv"               # string offset=5242
.Linfo_string253:
	.asciz	"_STNf1|<const int *&>"         # string offset=5258
.Linfo_string254:
	.asciz	"t5"                            # string offset=5280
.Linfo_string255:
	.asciz	"_Z2f1IJN12_GLOBAL__N_12t5EEEvv" # string offset=5283
.Linfo_string256:
	.asciz	"_STNf1|<(anonymous namespace)::t5>" # string offset=5314
.Linfo_string257:
	.asciz	"decltype(nullptr)"             # string offset=5349
.Linfo_string258:
	.asciz	"_Z2f1IJDnEEvv"                 # string offset=5367
.Linfo_string259:
	.asciz	"_STNf1|<std::nullptr_t>"       # string offset=5381
.Linfo_string260:
	.asciz	"_Z2f1IJPlS0_EEvv"              # string offset=5405
.Linfo_string261:
	.asciz	"_STNf1|<long *, long *>"       # string offset=5422
.Linfo_string262:
	.asciz	"_Z2f1IJPlP3udtEEvv"            # string offset=5446
.Linfo_string263:
	.asciz	"_STNf1|<long *, udt *>"        # string offset=5465
.Linfo_string264:
	.asciz	"_Z2f1IJKPvEEvv"                # string offset=5488
.Linfo_string265:
	.asciz	"_STNf1|<void *const>"          # string offset=5503
.Linfo_string266:
	.asciz	"_Z2f1IJPKPKvEEvv"              # string offset=5524
.Linfo_string267:
	.asciz	"_STNf1|<const void *const *>"  # string offset=5541
.Linfo_string268:
	.asciz	"_Z2f1IJFvvEEEvv"               # string offset=5570
.Linfo_string269:
	.asciz	"_STNf1|<void ()>"              # string offset=5586
.Linfo_string270:
	.asciz	"_Z2f1IJPFvvEEEvv"              # string offset=5603
.Linfo_string271:
	.asciz	"_STNf1|<void (*)()>"           # string offset=5620
.Linfo_string272:
	.asciz	"_Z2f1IJPZ4mainE3$_1EEvv"       # string offset=5640
.Linfo_string273:
	.asciz	"f1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:184:12) *>" # string offset=5664
.Linfo_string274:
	.asciz	"_Z2f1IJZ4mainE3$_2EEvv"        # string offset=5776
.Linfo_string275:
	.asciz	"f1<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:183:3)>" # string offset=5799
.Linfo_string276:
	.asciz	"_Z2f1IJPZ4mainE3$_2EEvv"       # string offset=5916
.Linfo_string277:
	.asciz	"f1<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:183:3) *>" # string offset=5940
.Linfo_string278:
	.asciz	"T1"                            # string offset=6059
.Linfo_string279:
	.asciz	"T2"                            # string offset=6062
.Linfo_string280:
	.asciz	"_Z2f5IJ2t1IJiEEEiEvv"          # string offset=6065
.Linfo_string281:
	.asciz	"_STNf5|<t1<int>, int>"         # string offset=6086
.Linfo_string282:
	.asciz	"_Z2f5IJEiEvv"                  # string offset=6108
.Linfo_string283:
	.asciz	"_STNf5|<int>"                  # string offset=6121
.Linfo_string284:
	.asciz	"_Z2f6I2t1IJiEEJEEvv"           # string offset=6134
.Linfo_string285:
	.asciz	"_STNf6|<t1<int> >"             # string offset=6154
.Linfo_string286:
	.asciz	"_Z2f1IJEEvv"                   # string offset=6172
.Linfo_string287:
	.asciz	"_STNf1|<>"                     # string offset=6184
.Linfo_string288:
	.asciz	"_Z2f1IJPKvS1_EEvv"             # string offset=6194
.Linfo_string289:
	.asciz	"_STNf1|<const void *, const void *>" # string offset=6212
.Linfo_string290:
	.asciz	"_STNt1|<int *>"                # string offset=6248
.Linfo_string291:
	.asciz	"_Z2f1IJP2t1IJPiEEEEvv"         # string offset=6263
.Linfo_string292:
	.asciz	"_STNf1|<t1<int *> *>"          # string offset=6285
.Linfo_string293:
	.asciz	"_Z2f1IJA_PiEEvv"               # string offset=6306
.Linfo_string294:
	.asciz	"_STNf1|<int *[]>"              # string offset=6322
.Linfo_string295:
	.asciz	"t7"                            # string offset=6339
.Linfo_string296:
	.asciz	"_Z2f1IJZ4mainE2t7EEvv"         # string offset=6342
.Linfo_string297:
	.asciz	"_STNf1|<t7>"                   # string offset=6364
.Linfo_string298:
	.asciz	"_Z2f1IJRA3_iEEvv"              # string offset=6376
.Linfo_string299:
	.asciz	"_STNf1|<int (&)[3]>"           # string offset=6393
.Linfo_string300:
	.asciz	"_Z2f1IJPA3_iEEvv"              # string offset=6413
.Linfo_string301:
	.asciz	"_STNf1|<int (*)[3]>"           # string offset=6430
.Linfo_string302:
	.asciz	"t1"                            # string offset=6450
.Linfo_string303:
	.asciz	"_Z2f7I2t1Evv"                  # string offset=6453
.Linfo_string304:
	.asciz	"_STNf7|<t1>"                   # string offset=6466
.Linfo_string305:
	.asciz	"_Z2f8I2t1iEvv"                 # string offset=6478
.Linfo_string306:
	.asciz	"_STNf8|<t1, int>"              # string offset=6492
.Linfo_string307:
	.asciz	"ns::inner::ttp"                # string offset=6509
.Linfo_string308:
	.asciz	"_ZN2ns8ttp_userINS_5inner3ttpEEEvv" # string offset=6524
.Linfo_string309:
	.asciz	"_STNttp_user|<ns::inner::ttp>" # string offset=6559
.Linfo_string310:
	.asciz	"_Z2f1IJPiPDnEEvv"              # string offset=6589
.Linfo_string311:
	.asciz	"_STNf1|<int *, std::nullptr_t *>" # string offset=6606
.Linfo_string312:
	.asciz	"_STNt7|<int>"                  # string offset=6639
.Linfo_string313:
	.asciz	"_Z2f1IJ2t7IiEEEvv"             # string offset=6652
.Linfo_string314:
	.asciz	"_STNf1|<t7<int> >"             # string offset=6670
.Linfo_string315:
	.asciz	"ns::inl::t9"                   # string offset=6688
.Linfo_string316:
	.asciz	"_Z2f7IN2ns3inl2t9EEvv"         # string offset=6700
.Linfo_string317:
	.asciz	"_STNf7|<ns::inl::t9>"          # string offset=6722
.Linfo_string318:
	.asciz	"_Z2f1IJU7_AtomiciEEvv"         # string offset=6743
.Linfo_string319:
	.asciz	"f1<_Atomic(int)>"              # string offset=6765
.Linfo_string320:
	.asciz	"_Z2f1IJilVcEEvv"               # string offset=6782
.Linfo_string321:
	.asciz	"_STNf1|<int, long, volatile char>" # string offset=6798
.Linfo_string322:
	.asciz	"_Z2f1IJDv2_iEEvv"              # string offset=6832
.Linfo_string323:
	.asciz	"f1<__attribute__((__vector_size__(2 * sizeof(int)))) int>" # string offset=6849
.Linfo_string324:
	.asciz	"_Z2f1IJVKPiEEvv"               # string offset=6907
.Linfo_string325:
	.asciz	"_STNf1|<int *const volatile>"  # string offset=6923
.Linfo_string326:
	.asciz	"_Z2f1IJVKvEEvv"                # string offset=6952
.Linfo_string327:
	.asciz	"_STNf1|<const volatile void>"  # string offset=6967
.Linfo_string328:
	.asciz	"t1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:184:12)>" # string offset=6996
.Linfo_string329:
	.asciz	"_Z2f1IJ2t1IJZ4mainE3$_1EEEEvv" # string offset=7106
.Linfo_string330:
	.asciz	"f1<t1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:184:12)> >" # string offset=7136
.Linfo_string331:
	.asciz	"_ZN3t10C2IvEEv"                # string offset=7251
.Linfo_string332:
	.asciz	"_Z2f1IJM3udtKFvvEEEvv"         # string offset=7266
.Linfo_string333:
	.asciz	"_STNf1|<void (udt::*)() const>" # string offset=7288
.Linfo_string334:
	.asciz	"_Z2f1IJM3udtVFvvREEEvv"        # string offset=7319
.Linfo_string335:
	.asciz	"_STNf1|<void (udt::*)() volatile &>" # string offset=7342
.Linfo_string336:
	.asciz	"_Z2f1IJM3udtVKFvvOEEEvv"       # string offset=7378
.Linfo_string337:
	.asciz	"_STNf1|<void (udt::*)() const volatile &&>" # string offset=7402
.Linfo_string338:
	.asciz	"_Z2f9IiEPFvvEv"                # string offset=7445
.Linfo_string339:
	.asciz	"_STNf9|<int>"                  # string offset=7460
.Linfo_string340:
	.asciz	"_Z2f1IJKPFvvEEEvv"             # string offset=7473
.Linfo_string341:
	.asciz	"_STNf1|<void (*const)()>"      # string offset=7491
.Linfo_string342:
	.asciz	"_Z2f1IJRA1_KcEEvv"             # string offset=7516
.Linfo_string343:
	.asciz	"_STNf1|<const char (&)[1]>"    # string offset=7534
.Linfo_string344:
	.asciz	"_Z2f1IJKFvvREEEvv"             # string offset=7561
.Linfo_string345:
	.asciz	"_STNf1|<void () const &>"      # string offset=7579
.Linfo_string346:
	.asciz	"_Z2f1IJVFvvOEEEvv"             # string offset=7604
.Linfo_string347:
	.asciz	"_STNf1|<void () volatile &&>"  # string offset=7622
.Linfo_string348:
	.asciz	"_Z2f1IJVKFvvEEEvv"             # string offset=7651
.Linfo_string349:
	.asciz	"_STNf1|<void () const volatile>" # string offset=7669
.Linfo_string350:
	.asciz	"_Z2f1IJA1_KPiEEvv"             # string offset=7701
.Linfo_string351:
	.asciz	"_STNf1|<int *const[1]>"        # string offset=7719
.Linfo_string352:
	.asciz	"_Z2f1IJRA1_KPiEEvv"            # string offset=7742
.Linfo_string353:
	.asciz	"_STNf1|<int *const (&)[1]>"    # string offset=7761
.Linfo_string354:
	.asciz	"_Z2f1IJRKM3udtFvvEEEvv"        # string offset=7788
.Linfo_string355:
	.asciz	"_STNf1|<void (udt::*const &)()>" # string offset=7811
.Linfo_string356:
	.asciz	"_Z2f1IJFPFvfEiEEEvv"           # string offset=7843
.Linfo_string357:
	.asciz	"_STNf1|<void (*(int))(float)>" # string offset=7863
.Linfo_string358:
	.asciz	"_Z2f1IJA1_2t1IJiEEEEvv"        # string offset=7893
.Linfo_string359:
	.asciz	"_STNf1|<t1<int>[1]>"           # string offset=7916
.Linfo_string360:
	.asciz	"_Z2f1IJPDoFvvEEEvv"            # string offset=7936
.Linfo_string361:
	.asciz	"f1<void (*)() noexcept>"       # string offset=7955
.Linfo_string362:
	.asciz	"_Z2f1IJFvZ4mainE3$_2EEEvv"     # string offset=7979
.Linfo_string363:
	.asciz	"f1<void ((unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:183:3))>" # string offset=8005
.Linfo_string364:
	.asciz	"_Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv" # string offset=8129
.Linfo_string365:
	.asciz	"f1<void (t8, (unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:183:3))>" # string offset=8165
.Linfo_string366:
	.asciz	"_Z2f1IJFvZ4mainE2t8EEEvv"      # string offset=8293
.Linfo_string367:
	.asciz	"_STNf1|<void (t8)>"            # string offset=8318
.Linfo_string368:
	.asciz	"_Z19operator_not_reallyIiEvv"  # string offset=8337
.Linfo_string369:
	.asciz	"_STNoperator_not_really|<int>" # string offset=8366
.Linfo_string370:
	.asciz	"_BitInt"                       # string offset=8396
.Linfo_string371:
	.asciz	"_Z2f1IJDB3_EEvv"               # string offset=8404
.Linfo_string372:
	.asciz	"f1<_BitInt(3)>"                # string offset=8420
.Linfo_string373:
	.asciz	"unsigned _BitInt"              # string offset=8435
.Linfo_string374:
	.asciz	"_Z2f1IJKDU5_EEvv"              # string offset=8452
.Linfo_string375:
	.asciz	"f1<const unsigned _BitInt(5)>" # string offset=8469
.Linfo_string376:
	.asciz	"_STNt1|<>"                     # string offset=8499
.Linfo_string377:
	.asciz	"_Z2f1IJFv2t1IJEES1_EEEvv"      # string offset=8509
.Linfo_string378:
	.asciz	"_STNf1|<void (t1<>, t1<>)>"    # string offset=8534
.Linfo_string379:
	.asciz	"_Z2f1IJM2t1IJEEiEEvv"          # string offset=8561
.Linfo_string380:
	.asciz	"_STNf1|<int t1<>::*>"          # string offset=8582
.Linfo_string381:
	.asciz	"_Z2f1IJZN2t83memEvE2t7EEvv"    # string offset=8603
.Linfo_string382:
	.asciz	"_Z2f1IJM2t8FvvEEEvv"           # string offset=8630
.Linfo_string383:
	.asciz	"_STNf1|<void (t8::*)()>"       # string offset=8650
.Linfo_string384:
	.asciz	"L"                             # string offset=8674
.Linfo_string385:
	.asciz	"v2"                            # string offset=8676
.Linfo_string386:
	.asciz	"N"                             # string offset=8679
.Linfo_string387:
	.asciz	"_STNt4|<3U>"                   # string offset=8681
.Linfo_string388:
	.asciz	"v1"                            # string offset=8693
.Linfo_string389:
	.asciz	"v6"                            # string offset=8696
.Linfo_string390:
	.asciz	"x"                             # string offset=8699
.Linfo_string391:
	.asciz	"t7i"                           # string offset=8701
.Linfo_string392:
	.asciz	"v3"                            # string offset=8705
.Linfo_string393:
	.asciz	"v4"                            # string offset=8708
.Linfo_string394:
	.asciz	"t11<(anonymous namespace)::LocalEnum, (anonymous namespace)::LocalEnum1>" # string offset=8711
.Linfo_string395:
	.asciz	"t12"                           # string offset=8784
.Linfo_string396:
	.asciz	"_STNt2|<int>"                  # string offset=8788
.Linfo_string397:
	.asciz	"_STNt2|<float>"                # string offset=8801
.Linfo_string398:
	.asciz	"_STNt1|<bool>"                 # string offset=8816
.Linfo_string399:
	.asciz	"_STNt2|<bool>"                 # string offset=8830
.Linfo_string400:
	.asciz	"_STNt1|<double>"               # string offset=8844
.Linfo_string401:
	.asciz	"_STNt2|<double>"               # string offset=8860
.Linfo_string402:
	.asciz	"_STNt1|<long>"                 # string offset=8876
.Linfo_string403:
	.asciz	"_STNt2|<long>"                 # string offset=8890
.Linfo_string404:
	.asciz	"_STNt1|<short>"                # string offset=8904
.Linfo_string405:
	.asciz	"_STNt2|<short>"                # string offset=8919
.Linfo_string406:
	.asciz	"_STNt1|<unsigned int>"         # string offset=8934
.Linfo_string407:
	.asciz	"_STNt2|<unsigned int>"         # string offset=8956
.Linfo_string408:
	.asciz	"_STNt1|<unsigned long long>"   # string offset=8978
.Linfo_string409:
	.asciz	"_STNt2|<unsigned long long>"   # string offset=9006
.Linfo_string410:
	.asciz	"_STNt1|<long long>"            # string offset=9034
.Linfo_string411:
	.asciz	"_STNt2|<long long>"            # string offset=9053
.Linfo_string412:
	.asciz	"_STNt1|<udt>"                  # string offset=9072
.Linfo_string413:
	.asciz	"_STNt2|<udt>"                  # string offset=9085
.Linfo_string414:
	.asciz	"_STNt1|<ns::udt>"              # string offset=9098
.Linfo_string415:
	.asciz	"_STNt2|<ns::udt>"              # string offset=9115
.Linfo_string416:
	.asciz	"_STNt1|<ns::udt *>"            # string offset=9132
.Linfo_string417:
	.asciz	"_STNt2|<ns::udt *>"            # string offset=9151
.Linfo_string418:
	.asciz	"_STNt1|<ns::inner::udt>"       # string offset=9170
.Linfo_string419:
	.asciz	"_STNt2|<ns::inner::udt>"       # string offset=9194
.Linfo_string420:
	.asciz	"_STNt1|<t1<int> >"             # string offset=9218
.Linfo_string421:
	.asciz	"_STNt2|<t1<int> >"             # string offset=9236
.Linfo_string422:
	.asciz	"_STNt1|<int, float>"           # string offset=9254
.Linfo_string423:
	.asciz	"_STNt2|<int, float>"           # string offset=9274
.Linfo_string424:
	.asciz	"_STNt2|<int *>"                # string offset=9294
.Linfo_string425:
	.asciz	"_STNt1|<int &>"                # string offset=9309
.Linfo_string426:
	.asciz	"_STNt2|<int &>"                # string offset=9324
.Linfo_string427:
	.asciz	"_STNt1|<int &&>"               # string offset=9339
.Linfo_string428:
	.asciz	"_STNt2|<int &&>"               # string offset=9355
.Linfo_string429:
	.asciz	"_STNt1|<const int>"            # string offset=9371
.Linfo_string430:
	.asciz	"_STNt2|<const int>"            # string offset=9390
.Linfo_string431:
	.asciz	"_STNt1|<int[3]>"               # string offset=9409
.Linfo_string432:
	.asciz	"_STNt2|<int[3]>"               # string offset=9425
.Linfo_string433:
	.asciz	"_STNt1|<void>"                 # string offset=9441
.Linfo_string434:
	.asciz	"_STNt2|<void>"                 # string offset=9455
.Linfo_string435:
	.asciz	"_STNt1|<outer_class::inner_class>" # string offset=9469
.Linfo_string436:
	.asciz	"_STNt2|<outer_class::inner_class>" # string offset=9503
.Linfo_string437:
	.asciz	"_STNt1|<unsigned long>"        # string offset=9537
.Linfo_string438:
	.asciz	"_STNt2|<unsigned long>"        # string offset=9560
.Linfo_string439:
	.asciz	"_STNt1|<t3<int, false> >"      # string offset=9583
.Linfo_string440:
	.asciz	"_STNt2|<t3<int, false> >"      # string offset=9608
.Linfo_string441:
	.asciz	"_STNt1|<t3<t3<int, false>, false> >" # string offset=9633
.Linfo_string442:
	.asciz	"_STNt2|<t3<t3<int, false>, false> >" # string offset=9669
.Linfo_string443:
	.asciz	"t2<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:184:12)>" # string offset=9705
.Linfo_string444:
	.asciz	"t1<t3<t3<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:184:12), false>, false> >" # string offset=9815
.Linfo_string445:
	.asciz	"t2<t3<t3<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:184:12), false>, false> >" # string offset=9948
.Linfo_string446:
	.asciz	"_STNt1|<int (float)>"          # string offset=10081
.Linfo_string447:
	.asciz	"_STNt2|<int (float)>"          # string offset=10102
.Linfo_string448:
	.asciz	"_STNt1|<void (...)>"           # string offset=10123
.Linfo_string449:
	.asciz	"_STNt2|<void (...)>"           # string offset=10143
.Linfo_string450:
	.asciz	"_STNt1|<void (int, ...)>"      # string offset=10163
.Linfo_string451:
	.asciz	"_STNt2|<void (int, ...)>"      # string offset=10188
.Linfo_string452:
	.asciz	"_STNt1|<const int &>"          # string offset=10213
.Linfo_string453:
	.asciz	"_STNt2|<const int &>"          # string offset=10234
.Linfo_string454:
	.asciz	"_STNt1|<const int *&>"         # string offset=10255
.Linfo_string455:
	.asciz	"_STNt2|<const int *&>"         # string offset=10277
.Linfo_string456:
	.asciz	"_STNt1|<(anonymous namespace)::t5>" # string offset=10299
.Linfo_string457:
	.asciz	"_STNt2|<(anonymous namespace)::t5>" # string offset=10334
.Linfo_string458:
	.asciz	"_STNt1|<std::nullptr_t>"       # string offset=10369
.Linfo_string459:
	.asciz	"_STNt2|<std::nullptr_t>"       # string offset=10393
.Linfo_string460:
	.asciz	"_STNt1|<long *, long *>"       # string offset=10417
.Linfo_string461:
	.asciz	"_STNt2|<long *, long *>"       # string offset=10441
.Linfo_string462:
	.asciz	"_STNt1|<long *, udt *>"        # string offset=10465
.Linfo_string463:
	.asciz	"_STNt2|<long *, udt *>"        # string offset=10488
.Linfo_string464:
	.asciz	"_STNt1|<void *const>"          # string offset=10511
.Linfo_string465:
	.asciz	"_STNt2|<void *const>"          # string offset=10532
.Linfo_string466:
	.asciz	"_STNt1|<const void *const *>"  # string offset=10553
.Linfo_string467:
	.asciz	"_STNt2|<const void *const *>"  # string offset=10582
.Linfo_string468:
	.asciz	"_STNt1|<void ()>"              # string offset=10611
.Linfo_string469:
	.asciz	"_STNt2|<void ()>"              # string offset=10628
.Linfo_string470:
	.asciz	"_STNt1|<void (*)()>"           # string offset=10645
.Linfo_string471:
	.asciz	"_STNt2|<void (*)()>"           # string offset=10665
.Linfo_string472:
	.asciz	"t1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:184:12) *>" # string offset=10685
.Linfo_string473:
	.asciz	"t2<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:184:12) *>" # string offset=10797
.Linfo_string474:
	.asciz	"t1<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:183:3)>" # string offset=10909
.Linfo_string475:
	.asciz	"t2<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:183:3)>" # string offset=11026
.Linfo_string476:
	.asciz	"t1<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:183:3) *>" # string offset=11143
.Linfo_string477:
	.asciz	"t2<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:183:3) *>" # string offset=11262
.Linfo_string478:
	.asciz	"_STNt2|<>"                     # string offset=11381
.Linfo_string479:
	.asciz	"_STNt1|<const void *, const void *>" # string offset=11391
.Linfo_string480:
	.asciz	"_STNt2|<const void *, const void *>" # string offset=11427
.Linfo_string481:
	.asciz	"_STNt1|<t1<int *> *>"          # string offset=11463
.Linfo_string482:
	.asciz	"_STNt2|<t1<int *> *>"          # string offset=11484
.Linfo_string483:
	.asciz	"_STNt1|<int *[]>"              # string offset=11505
.Linfo_string484:
	.asciz	"_STNt2|<int *[]>"              # string offset=11522
.Linfo_string485:
	.asciz	"this"                          # string offset=11539
.Linfo_string486:
	.asciz	"_STNt1|<t7>"                   # string offset=11544
.Linfo_string487:
	.asciz	"_STNt2|<t7>"                   # string offset=11556
.Linfo_string488:
	.asciz	"_STNt1|<int (&)[3]>"           # string offset=11568
.Linfo_string489:
	.asciz	"_STNt2|<int (&)[3]>"           # string offset=11588
.Linfo_string490:
	.asciz	"_STNt1|<int (*)[3]>"           # string offset=11608
.Linfo_string491:
	.asciz	"_STNt2|<int (*)[3]>"           # string offset=11628
.Linfo_string492:
	.asciz	"_STNt1|<int *, std::nullptr_t *>" # string offset=11648
.Linfo_string493:
	.asciz	"_STNt2|<int *, std::nullptr_t *>" # string offset=11681
.Linfo_string494:
	.asciz	"_STNt1|<t7<int> >"             # string offset=11714
.Linfo_string495:
	.asciz	"_STNt2|<t7<int> >"             # string offset=11732
.Linfo_string496:
	.asciz	"t1<_Atomic(int)>"              # string offset=11750
.Linfo_string497:
	.asciz	"t2<_Atomic(int)>"              # string offset=11767
.Linfo_string498:
	.asciz	"_STNt1|<int, long, volatile char>" # string offset=11784
.Linfo_string499:
	.asciz	"_STNt2|<int, long, volatile char>" # string offset=11818
.Linfo_string500:
	.asciz	"t1<__attribute__((__vector_size__(2 * sizeof(int)))) int>" # string offset=11852
.Linfo_string501:
	.asciz	"t2<__attribute__((__vector_size__(2 * sizeof(int)))) int>" # string offset=11910
.Linfo_string502:
	.asciz	"_STNt1|<int *const volatile>"  # string offset=11968
.Linfo_string503:
	.asciz	"_STNt2|<int *const volatile>"  # string offset=11997
.Linfo_string504:
	.asciz	"_STNt1|<const volatile void>"  # string offset=12026
.Linfo_string505:
	.asciz	"_STNt2|<const volatile void>"  # string offset=12055
.Linfo_string506:
	.asciz	"t1<t1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:184:12)> >" # string offset=12084
.Linfo_string507:
	.asciz	"t2<t1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:184:12)> >" # string offset=12199
.Linfo_string508:
	.asciz	"_STNt1|<void (udt::*)() const>" # string offset=12314
.Linfo_string509:
	.asciz	"_STNt2|<void (udt::*)() const>" # string offset=12345
.Linfo_string510:
	.asciz	"_STNt1|<void (udt::*)() volatile &>" # string offset=12376
.Linfo_string511:
	.asciz	"_STNt2|<void (udt::*)() volatile &>" # string offset=12412
.Linfo_string512:
	.asciz	"_STNt1|<void (udt::*)() const volatile &&>" # string offset=12448
.Linfo_string513:
	.asciz	"_STNt2|<void (udt::*)() const volatile &&>" # string offset=12491
.Linfo_string514:
	.asciz	"_STNt1|<void (*const)()>"      # string offset=12534
.Linfo_string515:
	.asciz	"_STNt2|<void (*const)()>"      # string offset=12559
.Linfo_string516:
	.asciz	"_STNt1|<const char (&)[1]>"    # string offset=12584
.Linfo_string517:
	.asciz	"_STNt2|<const char (&)[1]>"    # string offset=12611
.Linfo_string518:
	.asciz	"_STNt1|<void () const &>"      # string offset=12638
.Linfo_string519:
	.asciz	"_STNt2|<void () const &>"      # string offset=12663
.Linfo_string520:
	.asciz	"_STNt1|<void () volatile &&>"  # string offset=12688
.Linfo_string521:
	.asciz	"_STNt2|<void () volatile &&>"  # string offset=12717
.Linfo_string522:
	.asciz	"_STNt1|<void () const volatile>" # string offset=12746
.Linfo_string523:
	.asciz	"_STNt2|<void () const volatile>" # string offset=12778
.Linfo_string524:
	.asciz	"_STNt1|<int *const[1]>"        # string offset=12810
.Linfo_string525:
	.asciz	"_STNt2|<int *const[1]>"        # string offset=12833
.Linfo_string526:
	.asciz	"_STNt1|<int *const (&)[1]>"    # string offset=12856
.Linfo_string527:
	.asciz	"_STNt2|<int *const (&)[1]>"    # string offset=12883
.Linfo_string528:
	.asciz	"_STNt1|<void (udt::*const &)()>" # string offset=12910
.Linfo_string529:
	.asciz	"_STNt2|<void (udt::*const &)()>" # string offset=12942
.Linfo_string530:
	.asciz	"_STNt1|<void (*(int))(float)>" # string offset=12974
.Linfo_string531:
	.asciz	"_STNt2|<void (*(int))(float)>" # string offset=13004
.Linfo_string532:
	.asciz	"_STNt1|<t1<int>[1]>"           # string offset=13034
.Linfo_string533:
	.asciz	"_STNt2|<t1<int>[1]>"           # string offset=13054
.Linfo_string534:
	.asciz	"t1<void (*)() noexcept>"       # string offset=13074
.Linfo_string535:
	.asciz	"t2<void (*)() noexcept>"       # string offset=13098
.Linfo_string536:
	.asciz	"t1<void ((unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:183:3))>" # string offset=13122
.Linfo_string537:
	.asciz	"t2<void ((unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:183:3))>" # string offset=13246
.Linfo_string538:
	.asciz	"t1<void (t8, (unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:183:3))>" # string offset=13370
.Linfo_string539:
	.asciz	"t2<void (t8, (unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:183:3))>" # string offset=13498
.Linfo_string540:
	.asciz	"_STNt1|<void (t8)>"            # string offset=13626
.Linfo_string541:
	.asciz	"_STNt2|<void (t8)>"            # string offset=13645
.Linfo_string542:
	.asciz	"t1<_BitInt(3)>"                # string offset=13664
.Linfo_string543:
	.asciz	"t2<_BitInt(3)>"                # string offset=13679
.Linfo_string544:
	.asciz	"t1<const unsigned _BitInt(5)>" # string offset=13694
.Linfo_string545:
	.asciz	"t2<const unsigned _BitInt(5)>" # string offset=13724
.Linfo_string546:
	.asciz	"_STNt1|<void (t1<>, t1<>)>"    # string offset=13754
.Linfo_string547:
	.asciz	"_STNt2|<void (t1<>, t1<>)>"    # string offset=13781
.Linfo_string548:
	.asciz	"_STNt1|<int t1<>::*>"          # string offset=13808
.Linfo_string549:
	.asciz	"_STNt2|<int t1<>::*>"          # string offset=13829
.Linfo_string550:
	.asciz	"_STNt1|<void (t8::*)()>"       # string offset=13850
.Linfo_string551:
	.asciz	"_STNt2|<void (t8::*)()>"       # string offset=13874
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
.Ldebug_addr_end0:
	.ident	"clang version 15.0.0 (git@github.com:llvm/llvm-project.git 9980a3f8318cbfe2305b4ffc58a7256da1a3891d)"
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
	.addrsig_sym _Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv
	.addrsig_sym _Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv
	.addrsig_sym _Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv
	.addrsig_sym _Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv
	.addrsig_sym _Z2f3IN12_GLOBAL__N_19LocalEnumEJLS1_0EEEvv
	.addrsig_sym _Z2f3IPiJXadL_Z1iEEEEvv
	.addrsig_sym _Z2f3IPiJLS0_0EEEvv
	.addrsig_sym _Z2f3ImJLm1EEEvv
	.addrsig_sym _Z2f3IyJLy1EEEvv
	.addrsig_sym _Z2f3IlJLl1EEEvv
	.addrsig_sym _Z2f3IjJLj1EEEvv
	.addrsig_sym _Z2f3IsJLs1EEEvv
	.addrsig_sym _Z2f3IhJLh0EEEvv
	.addrsig_sym _Z2f3IaJLa0EEEvv
	.addrsig_sym _Z2f3ItJLt1ELt2EEEvv
	.addrsig_sym _Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv
	.addrsig_sym _Z2f3InJLn18446744073709551614EEEvv
	.addrsig_sym _Z2f4IjLj3EEvv
	.addrsig_sym _Z2f1IJ2t3IiLb0EEEEvv
	.addrsig_sym _Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv
	.addrsig_sym _Z2f1IJZ4mainE3$_1EEvv
	.addrsig_sym _Z2f1IJ2t3IS0_IZ4mainE3$_1Lb0EELb0EEEEvv
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
	.addrsig_sym _Z2f1IJPZ4mainE3$_1EEvv
	.addrsig_sym _Z2f1IJZ4mainE3$_2EEvv
	.addrsig_sym _Z2f1IJPZ4mainE3$_2EEvv
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
	.addrsig_sym _Z2f7IN2ns3inl2t9EEvv
	.addrsig_sym _Z2f1IJU7_AtomiciEEvv
	.addrsig_sym _Z2f1IJilVcEEvv
	.addrsig_sym _Z2f1IJDv2_iEEvv
	.addrsig_sym _Z2f1IJVKPiEEvv
	.addrsig_sym _Z2f1IJVKvEEvv
	.addrsig_sym _Z2f1IJ2t1IJZ4mainE3$_1EEEEvv
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
	.addrsig_sym _Z2f1IJFvZ4mainE3$_2EEEvv
	.addrsig_sym _Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv
	.addrsig_sym _Z2f1IJFvZ4mainE2t8EEEvv
	.addrsig_sym _Z19operator_not_reallyIiEvv
	.addrsig_sym _Z2f1IJDB3_EEvv
	.addrsig_sym _Z2f1IJKDU5_EEvv
	.addrsig_sym _Z2f1IJFv2t1IJEES1_EEEvv
	.addrsig_sym _Z2f1IJM2t1IJEEiEEvv
	.addrsig_sym _Z2f1IJZN2t83memEvE2t7EEvv
	.addrsig_sym _Z2f1IJM2t8FvvEEEvv
	.section	.debug_line,"",@progbits
.Lline_table_start0:
