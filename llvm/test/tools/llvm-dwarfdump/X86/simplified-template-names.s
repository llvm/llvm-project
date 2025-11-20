# RUN: llvm-mc < %s -filetype obj -triple x86_64 -o - \
# RUN:   | llvm-dwarfdump --verify - | FileCheck %s

# Checking the LLVM side of cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp
# Compile that file with `-g -Xclang -gsimple-template-names=mangled -Xclang -debug-forward-template-params -S -std=c++20 -target x86_64-linux`
# to (re)generate this assembly file - while it might be slightly overkill in
# some ways, it seems small/simple enough to keep this as an exact match for
# that end to end test.

# CHECK: No errors.
	.file	"simplified_template_names.cpp"
	.file	0 "/Users/michaelbuch/Git/llvm-worktrees/main" "cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp" md5 0x0d97d46dd4360733206888e9ffc9e7e6
	.text
	.globl	_Zli5_suffy                     # -- Begin function _Zli5_suffy
	.p2align	4
	.type	_Zli5_suffy,@function
_Zli5_suffy:                            # @_Zli5_suffy
.Lfunc_begin0:
	.loc	0 144 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:144:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp0:
	.loc	0 144 44 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:144:44
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
	.loc	0 190 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:190:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
.Ltmp2:
	.loc	0 192 8 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:192:8
	movb	.L__const.main.L(%rip), %al
	movb	%al, -2(%rbp)
	.loc	0 193 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:193:3
	callq	_Z2f1IJiEEvv
	.loc	0 194 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:194:3
	callq	_Z2f1IJfEEvv
	.loc	0 195 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:195:3
	callq	_Z2f1IJbEEvv
	.loc	0 196 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:196:3
	callq	_Z2f1IJdEEvv
	.loc	0 197 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:197:3
	callq	_Z2f1IJlEEvv
	.loc	0 198 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:198:3
	callq	_Z2f1IJsEEvv
	.loc	0 199 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:199:3
	callq	_Z2f1IJjEEvv
	.loc	0 200 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:200:3
	callq	_Z2f1IJyEEvv
	.loc	0 201 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:201:3
	callq	_Z2f1IJxEEvv
	.loc	0 202 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:202:3
	callq	_Z2f1IJ3udtEEvv
	.loc	0 203 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:203:3
	callq	_Z2f1IJN2ns3udtEEEvv
	.loc	0 204 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:204:3
	callq	_Z2f1IJPN2ns3udtEEEvv
	.loc	0 205 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:205:3
	callq	_Z2f1IJN2ns5inner3udtEEEvv
	.loc	0 206 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:206:3
	callq	_Z2f1IJ2t1IJiEEEEvv
	.loc	0 207 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:207:3
	callq	_Z2f1IJifEEvv
	.loc	0 208 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:208:3
	callq	_Z2f1IJPiEEvv
	.loc	0 209 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:209:3
	callq	_Z2f1IJRiEEvv
	.loc	0 210 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:210:3
	callq	_Z2f1IJOiEEvv
	.loc	0 211 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:211:3
	callq	_Z2f1IJKiEEvv
	.loc	0 212 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:212:3
	callq	_Z2f1IJA3_iEEvv
	.loc	0 213 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:213:3
	callq	_Z2f1IJvEEvv
	.loc	0 214 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:214:3
	callq	_Z2f1IJN11outer_class11inner_classEEEvv
	.loc	0 215 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:215:3
	callq	_Z2f1IJmEEvv
	.loc	0 216 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:216:3
	callq	_Z2f2ILb1ELi3EEvv
	.loc	0 217 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:217:3
	callq	_Z2f3IN2ns11EnumerationETpTnT_JLS1_1ELS1_2EEEvv
	.loc	0 218 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:218:3
	callq	_Z2f3IN2ns16EnumerationClassETpTnT_JLS1_1ELS1_2EEEvv
	.loc	0 219 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:219:3
	callq	_Z2f3IN2ns16EnumerationSmallETpTnT_JLS1_255EEEvv
	.loc	0 220 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:220:3
	callq	_Z2f3IN2ns3$_0ETpTnT_JLS1_1ELS1_2EEEvv
	.loc	0 221 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:221:3
	callq	_Z2f3IN12_GLOBAL__N_19LocalEnumETpTnT_JLS1_0EEEvv
	.loc	0 222 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:222:3
	callq	_Z2f3IPiTpTnT_JXadL_Z1iEEEEvv
	.loc	0 223 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:223:3
	callq	_Z2f3IPiTpTnT_JLS0_0EEEvv
	.loc	0 225 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:225:3
	callq	_Z2f3ImTpTnT_JLm1EEEvv
	.loc	0 226 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:226:3
	callq	_Z2f3IyTpTnT_JLy1EEEvv
	.loc	0 227 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:227:3
	callq	_Z2f3IlTpTnT_JLl1EEEvv
	.loc	0 228 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:228:3
	callq	_Z2f3IjTpTnT_JLj1EEEvv
	.loc	0 229 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:229:3
	callq	_Z2f3IsTpTnT_JLs1EEEvv
	.loc	0 230 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:230:3
	callq	_Z2f3IhTpTnT_JLh0EEEvv
	.loc	0 231 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:231:3
	callq	_Z2f3IaTpTnT_JLa0EEEvv
	.loc	0 232 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:232:3
	callq	_Z2f3ItTpTnT_JLt1ELt2EEEvv
	.loc	0 233 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:233:3
	callq	_Z2f3IcTpTnT_JLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv
	.loc	0 234 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:234:3
	callq	_Z2f3InTpTnT_JLn18446744073709551614EEEvv
	.loc	0 235 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:235:3
	callq	_Z2f4IjLj3EEvv
	.loc	0 236 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:236:3
	callq	_Z2f1IJ2t3IiLb0EEEEvv
	.loc	0 237 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:237:3
	callq	_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv
	.loc	0 238 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:238:3
	callq	_Z2f1IJZ4mainE3$_0EEvv
	.loc	0 240 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:240:3
	callq	_Z2f1IJ2t3IS0_IZ4mainE3$_0Lb0EELb0EEEEvv
	.loc	0 241 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:241:3
	callq	_Z2f1IJFifEEEvv
	.loc	0 242 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:242:3
	callq	_Z2f1IJFvzEEEvv
	.loc	0 243 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:243:3
	callq	_Z2f1IJFvizEEEvv
	.loc	0 244 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:244:3
	callq	_Z2f1IJRKiEEvv
	.loc	0 245 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:245:3
	callq	_Z2f1IJRPKiEEvv
	.loc	0 246 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:246:3
	callq	_Z2f1IJN12_GLOBAL__N_12t5EEEvv
	.loc	0 247 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:247:3
	callq	_Z2f1IJDnEEvv
	.loc	0 248 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:248:3
	callq	_Z2f1IJPlS0_EEvv
	.loc	0 249 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:249:3
	callq	_Z2f1IJPlP3udtEEvv
	.loc	0 250 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:250:3
	callq	_Z2f1IJKPvEEvv
	.loc	0 251 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:251:3
	callq	_Z2f1IJPKPKvEEvv
	.loc	0 252 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:252:3
	callq	_Z2f1IJFvvEEEvv
	.loc	0 253 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:253:3
	callq	_Z2f1IJPFvvEEEvv
	.loc	0 254 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:254:3
	callq	_Z2f1IJPZ4mainE3$_0EEvv
	.loc	0 255 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:255:3
	callq	_Z2f1IJZ4mainE3$_1EEvv
	.loc	0 256 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:256:3
	callq	_Z2f1IJPZ4mainE3$_1EEvv
	.loc	0 257 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:257:3
	callq	_Z2f5IJ2t1IJiEEEiEvv
	.loc	0 258 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:258:3
	callq	_Z2f5IJEiEvv
	.loc	0 259 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:259:3
	callq	_Z2f6I2t1IJiEEJEEvv
	.loc	0 260 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:260:3
	callq	_Z2f1IJEEvv
	.loc	0 261 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:261:3
	callq	_Z2f1IJPKvS1_EEvv
	.loc	0 262 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:262:3
	callq	_Z2f1IJP2t1IJPiEEEEvv
	.loc	0 263 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:263:3
	callq	_Z2f1IJA_PiEEvv
	.loc	0 265 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:265:6
	leaq	-5(%rbp), %rdi
	movl	$1, %esi
	callq	_ZN2t6lsIiEEvi
	.loc	0 266 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:266:6
	leaq	-5(%rbp), %rdi
	movl	$1, %esi
	callq	_ZN2t6ltIiEEvi
	.loc	0 267 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:267:6
	leaq	-5(%rbp), %rdi
	movl	$1, %esi
	callq	_ZN2t6leIiEEvi
	.loc	0 268 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:268:6
	leaq	-5(%rbp), %rdi
	callq	_ZN2t6cvP2t1IJfEEIiEEv
	.loc	0 269 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:269:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6miIiEEvi
	.loc	0 270 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:270:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6mlIiEEvi
	.loc	0 271 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:271:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6dvIiEEvi
	.loc	0 272 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:272:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6rmIiEEvi
	.loc	0 273 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:273:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6eoIiEEvi
	.loc	0 274 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:274:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6anIiEEvi
	.loc	0 275 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:275:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6orIiEEvi
	.loc	0 276 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:276:6
	leaq	-5(%rbp), %rdi
	callq	_ZN2t6coIiEEvv
	.loc	0 277 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:277:6
	leaq	-5(%rbp), %rdi
	callq	_ZN2t6ntIiEEvv
	.loc	0 278 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:278:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6aSIiEEvi
	.loc	0 279 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:279:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6gtIiEEvi
	.loc	0 280 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:280:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6cmIiEEvi
	.loc	0 281 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:281:6
	leaq	-5(%rbp), %rdi
	callq	_ZN2t6clIiEEvv
	.loc	0 282 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:282:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6ixIiEEvi
	.loc	0 283 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:283:6
	leaq	-5(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6ssIiEEvi
	.loc	0 284 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:284:3
	xorl	%eax, %eax
	movl	%eax, %edi
	xorl	%esi, %esi
	callq	_ZN2t6nwIiEEPvmT_
	.loc	0 285 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:285:3
	xorl	%eax, %eax
	movl	%eax, %edi
	xorl	%esi, %esi
	callq	_ZN2t6naIiEEPvmT_
	.loc	0 286 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:286:3
	xorl	%eax, %eax
	movl	%eax, %edi
	xorl	%esi, %esi
	callq	_ZN2t6dlIiEEvPvT_
	.loc	0 287 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:287:3
	xorl	%eax, %eax
	movl	%eax, %edi
	xorl	%esi, %esi
	callq	_ZN2t6daIiEEvPvT_
	.loc	0 288 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:288:6
	leaq	-5(%rbp), %rdi
	callq	_ZN2t6awIiEEiv
	.loc	0 289 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:289:3
	movl	$42, %edi
	callq	_Zli5_suffy
	.loc	0 291 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:291:3
	callq	_Z2f1IJZ4mainE2t7EEvv
	.loc	0 292 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:292:3
	callq	_Z2f1IJRA3_iEEvv
	.loc	0 293 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:293:3
	callq	_Z2f1IJPA3_iEEvv
	.loc	0 294 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:294:3
	callq	_Z2f7I2t1Evv
	.loc	0 295 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:295:3
	callq	_Z2f8I2t1iEvv
	.loc	0 297 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:297:3
	callq	_ZN2ns8ttp_userINS_5inner3ttpEEEvv
	.loc	0 298 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:298:3
	callq	_Z2f1IJPiPDnEEvv
	.loc	0 300 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:300:3
	callq	_Z2f1IJ2t7IiEEEvv
	.loc	0 301 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:301:3
	callq	_Z2f7ITtTpTyEN2ns3inl2t9EEvv
	.loc	0 302 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:302:3
	callq	_Z2f1IJU7_AtomiciEEvv
	.loc	0 303 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:303:3
	callq	_Z2f1IJilVcEEvv
	.loc	0 304 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:304:3
	callq	_Z2f1IJDv2_iEEvv
	.loc	0 305 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:305:3
	callq	_Z2f1IJVKPiEEvv
	.loc	0 306 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:306:3
	callq	_Z2f1IJVKvEEvv
	.loc	0 307 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:307:3
	callq	_Z2f1IJ2t1IJZ4mainE3$_0EEEEvv
	.loc	0 308 7                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:308:7
	leaq	-7(%rbp), %rdi
	callq	_ZN3t10C2IvEEv
	.loc	0 309 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:309:3
	callq	_Z2f1IJM3udtKFvvEEEvv
	.loc	0 310 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:310:3
	callq	_Z2f1IJM3udtVFvvREEEvv
	.loc	0 311 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:311:3
	callq	_Z2f1IJM3udtVKFvvOEEEvv
	.loc	0 312 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:312:3
	callq	_Z2f9IiEPFvvEv
	.loc	0 313 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:313:3
	callq	_Z2f1IJKPFvvEEEvv
	.loc	0 314 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:314:3
	callq	_Z2f1IJRA1_KcEEvv
	.loc	0 315 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:315:3
	callq	_Z2f1IJKFvvREEEvv
	.loc	0 316 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:316:3
	callq	_Z2f1IJVFvvOEEEvv
	.loc	0 317 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:317:3
	callq	_Z2f1IJVKFvvEEEvv
	.loc	0 318 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:318:3
	callq	_Z2f1IJA1_KPiEEvv
	.loc	0 319 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:319:3
	callq	_Z2f1IJRA1_KPiEEvv
	.loc	0 320 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:320:3
	callq	_Z2f1IJRKM3udtFvvEEEvv
	.loc	0 321 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:321:3
	callq	_Z2f1IJFPFvfEiEEEvv
	.loc	0 322 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:322:3
	callq	_Z2f1IJA1_2t1IJiEEEEvv
	.loc	0 323 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:323:3
	callq	_Z2f1IJPDoFvvEEEvv
	.loc	0 324 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:324:3
	callq	_Z2f1IJFvZ4mainE3$_1EEEvv
	.loc	0 326 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:326:3
	callq	_Z2f1IJFvZ4mainE2t8Z4mainE3$_1EEEvv
	.loc	0 327 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:327:3
	callq	_Z2f1IJFvZ4mainE2t8EEEvv
	.loc	0 328 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:328:3
	callq	_Z19operator_not_reallyIiEvv
	.loc	0 330 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:330:3
	callq	_Z3f11IDB3_TnT_LS0_2EEvv
	.loc	0 331 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:331:3
	callq	_Z3f11IKDU5_TnT_LS0_2EEvv
	.loc	0 332 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:332:3
	callq	_Z3f11IDB65_TnT_LS0_2EEvv
	.loc	0 333 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:333:3
	callq	_Z3f11IKDU65_TnT_LS0_2EEvv
	.loc	0 334 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:334:3
	callq	_Z2f1IJFv2t1IJEES1_EEEvv
	.loc	0 335 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:335:3
	callq	_Z2f1IJM2t1IJEEiEEvv
	.loc	0 337 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:337:3
	callq	_Z2f1IJU9swiftcallFvvEEEvv
	.loc	0 339 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:339:3
	callq	_Z2f1IJFivEEEvv
	.loc	0 340 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:340:3
	callq	_Z3f10ILN2ns3$_0E0EEvv
	.loc	0 341 1                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:341:1
	xorl	%eax, %eax
	.loc	0 341 1 epilogue_begin is_stmt 0 # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:341:1
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
	.loc	0 35 0 is_stmt 1                # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp4:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp6:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp8:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp10:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp12:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp14:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp16:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp18:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp20:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp22:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp24:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp26:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp28:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp30:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp32:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp34:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp36:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp38:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp40:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp42:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp44:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp46:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp48:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 40 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:40:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp50:
	.loc	0 41 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:41:1
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
	.loc	0 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp52:
	.loc	0 44 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:1
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
	.loc	0 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp54:
	.loc	0 44 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:1
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
	.loc	0 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp56:
	.loc	0 44 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:1
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
	.loc	0 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp58:
	.loc	0 44 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:1
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
	.loc	0 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp60:
	.loc	0 44 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:1
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
	.loc	0 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp62:
	.loc	0 44 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:1
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
	.loc	0 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp64:
	.loc	0 44 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:1
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
	.loc	0 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp66:
	.loc	0 44 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:1
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
	.loc	0 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp68:
	.loc	0 44 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:1
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
	.loc	0 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp70:
	.loc	0 44 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:1
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
	.loc	0 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp72:
	.loc	0 44 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:1
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
	.loc	0 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp74:
	.loc	0 44 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:1
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
	.loc	0 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp76:
	.loc	0 44 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:1
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
	.loc	0 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp78:
	.loc	0 44 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:1
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
	.loc	0 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp80:
	.loc	0 44 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:1
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
	.loc	0 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp82:
	.loc	0 44 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:1
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
	.loc	0 43 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:43:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp84:
	.loc	0 44 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:44:1
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
	.loc	0 46 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:46:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp86:
	.loc	0 47 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:47:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp88:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp90:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp92:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp94:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp96:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp98:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp100:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp102:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp104:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp106:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp108:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp110:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp112:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp114:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp116:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp118:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp120:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp122:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp124:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp126:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 64 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:64:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp128:
	.loc	0 64 13 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:64:13
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
	.loc	0 64 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:64:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp130:
	.loc	0 64 13 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:64:13
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
	.loc	0 66 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:66:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp132:
	.loc	0 66 13 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:66:13
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp134:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp136:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp138:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp140:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 69 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:69:0
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
	.loc	0 70 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:70:3
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
	.loc	0 72 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:72:0
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
	.loc	0 73 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:73:3
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
	.loc	0 75 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:75:0
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
	.loc	0 76 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:76:3
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
	.loc	0 78 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:78:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp148:
	.loc	0 79 5 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:79:5
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	.loc	0 79 5 epilogue_begin is_stmt 0 # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:79:5
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
	.loc	0 82 0 is_stmt 1                # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:82:0
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
	.loc	0 83 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:83:3
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
	.loc	0 85 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:85:0
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
	.loc	0 86 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:86:3
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
	.loc	0 88 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:88:0
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
	.loc	0 89 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:89:3
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
	.loc	0 91 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:91:0
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
	.loc	0 92 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:92:3
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
	.loc	0 94 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:94:0
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
	.loc	0 95 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:95:3
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
	.loc	0 97 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:97:0
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
	.loc	0 98 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:98:3
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
	.loc	0 100 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:100:0
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
	.loc	0 101 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:101:3
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
	.loc	0 103 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:103:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp164:
	.loc	0 104 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:104:3
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
	.loc	0 106 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:106:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp166:
	.loc	0 107 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:107:3
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
	.loc	0 109 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:109:0
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
	.loc	0 110 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:110:3
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
	.loc	0 112 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:112:0
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
	.loc	0 113 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:113:3
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
	.loc	0 115 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:115:0
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
	.loc	0 116 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:116:3
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
	.loc	0 118 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:118:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp174:
	.loc	0 119 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:119:3
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
	.loc	0 121 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:121:0
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
	.loc	0 122 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:122:3
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
	.loc	0 124 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:124:0
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
	.loc	0 125 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:125:3
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
	.loc	0 127 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:127:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	.loc	0 127 0 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:127:0
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Lfunc_end90:
	.size	_ZN2t6nwIiEEPvmT_, .Lfunc_end90-_ZN2t6nwIiEEPvmT_
	.cfi_endproc
	.file	1 "builds/release/bin/../include/c++/v1/__cstddef" "size_t.h"
                                        # -- End function
	.section	.text._ZN2t6naIiEEPvmT_,"axG",@progbits,_ZN2t6naIiEEPvmT_,comdat
	.weak	_ZN2t6naIiEEPvmT_               # -- Begin function _ZN2t6naIiEEPvmT_
	.p2align	4
	.type	_ZN2t6naIiEEPvmT_,@function
_ZN2t6naIiEEPvmT_:                      # @_ZN2t6naIiEEPvmT_
.Lfunc_begin91:
	.loc	0 134 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:134:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	.loc	0 134 0 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:134:0
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
	.loc	0 131 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:131:0
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
	.loc	0 132 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:132:3
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
	.loc	0 138 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:138:0
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
	.loc	0 139 3 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:139:3
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
	.loc	0 141 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:141:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	.loc	0 141 0 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:141:0
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp184:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp186:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp188:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 145 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:145:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp190:
	.loc	0 145 53 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:145:53
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
	.loc	0 146 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:146:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp192:
	.loc	0 146 66 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:146:66
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
	.loc	0 28 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:28:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp194:
	.loc	0 28 19 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:28:19
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp196:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp198:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 145 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:145:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp200:
	.loc	0 145 53 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:145:53
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp202:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp204:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp206:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp208:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp210:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp212:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 169 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:169:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp214:
	.loc	0 169 11 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:169:11
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp216:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp218:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp220:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 164 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:164:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp222:
	.loc	0 165 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:165:3
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	.loc	0 165 3 epilogue_begin is_stmt 0 # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:165:3
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
	.loc	0 35 0 is_stmt 1                # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp224:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp226:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp228:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp230:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp232:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp234:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp236:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp238:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp240:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp242:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp244:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp246:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp248:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp250:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 173 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:173:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp252:
	.loc	0 174 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:174:1
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
	.loc	0 188 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:188:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp254:
	.loc	0 188 40 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:188:40
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
	.loc	0 188 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:188:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp256:
	.loc	0 188 40 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:188:40
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
	.loc	0 188 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:188:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp258:
	.loc	0 188 40 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:188:40
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
	.loc	0 188 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:188:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp260:
	.loc	0 188 40 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:188:40
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp262:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp264:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp266:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp268:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 185 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:185:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp270:
	.loc	0 186 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:186:1
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
	.loc	0 342 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:342:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
.Ltmp272:
	.loc	0 344 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:344:3
	callq	_Z2f1IJZN2t83memEvE2t7EEvv
	.loc	0 345 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:345:3
	callq	_Z2f1IJM2t8FvvEEEvv
	.loc	0 346 1 epilogue_begin          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:346:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp274:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 35 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp276:
	.loc	0 38 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
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
	.loc	0 360 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:360:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp278:
	.loc	0 363 1 prologue_end epilogue_begin # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:363:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp279:
.Lfunc_end142:
	.size	_ZN18complex_type_units2f1Ev, .Lfunc_end142-_ZN18complex_type_units2f1Ev
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

	.file	2 "builds/release/lib/clang/22/include" "stdint.h" md5 0xa47a6a6c7fcbc62776237de6abac9549
	.file	3 "builds/release/bin/../include/c++/v1" "cstdint"
	.file	4 "builds/release/lib/clang/22/include" "__stddef_max_align_t.h" md5 0x3c0a2f19d136d39aa835c737c7105def
	.file	5 "builds/release/bin/../include/c++/v1/__cstddef" "max_align_t.h"
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
	.byte	18                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
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
	.byte	21                              # Abbreviation Code
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
	.byte	22                              # Abbreviation Code
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
	.byte	23                              # Abbreviation Code
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
	.byte	24                              # Abbreviation Code
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
	.byte	5                               # DW_FORM_data2
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
	.byte	5                               # DW_FORM_data2
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
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	30                              # Abbreviation Code
	.ascii	"\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	31                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	32                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
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
	.byte	15                              # DW_FORM_udata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	34                              # Abbreviation Code
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
	.byte	35                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	28                              # DW_AT_const_value
	.byte	13                              # DW_FORM_sdata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	36                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	28                              # DW_AT_const_value
	.byte	15                              # DW_FORM_udata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	37                              # Abbreviation Code
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
	.byte	38                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	39                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	28                              # DW_AT_const_value
	.byte	10                              # DW_FORM_block1
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
	.byte	38                              # DW_FORM_strx2
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	41                              # Abbreviation Code
	.ascii	"\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	42                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	43                              # Abbreviation Code
	.ascii	"\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	44                              # Abbreviation Code
	.ascii	"\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	45                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	46                              # Abbreviation Code
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
	.byte	47                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	48                              # Abbreviation Code
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
	.byte	49                              # Abbreviation Code
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
	.byte	50                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	51                              # Abbreviation Code
	.byte	57                              # DW_TAG_namespace
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.ascii	"\211\001"                      # DW_AT_export_symbols
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	52                              # Abbreviation Code
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
	.byte	53                              # Abbreviation Code
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
	.byte	54                              # Abbreviation Code
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
	.byte	55                              # Abbreviation Code
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
	.byte	56                              # Abbreviation Code
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
	.byte	57                              # Abbreviation Code
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
	.byte	58                              # Abbreviation Code
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
	.byte	59                              # Abbreviation Code
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
	.byte	59                              # DW_AT_decl_line
	.byte	5                               # DW_FORM_data2
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
	.byte	5                               # DW_FORM_data2
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	62                              # Abbreviation Code
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
	.byte	5                               # DW_FORM_data2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	63                              # Abbreviation Code
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
	.byte	64                              # Abbreviation Code
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
	.byte	65                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	30                              # DW_AT_default_value
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	66                              # Abbreviation Code
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
	.byte	67                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	68                              # Abbreviation Code
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
	.byte	69                              # Abbreviation Code
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
	.byte	70                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	71                              # Abbreviation Code
	.byte	16                              # DW_TAG_reference_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	72                              # Abbreviation Code
	.byte	66                              # DW_TAG_rvalue_reference_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	73                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	74                              # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	75                              # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	55                              # DW_AT_count
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	76                              # Abbreviation Code
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
	.byte	77                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	78                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	79                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	80                              # Abbreviation Code
	.byte	24                              # DW_TAG_unspecified_parameters
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	81                              # Abbreviation Code
	.byte	59                              # DW_TAG_unspecified_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	82                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	83                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	84                              # Abbreviation Code
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
	.byte	85                              # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	86                              # Abbreviation Code
	.byte	71                              # DW_TAG_atomic_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	87                              # Abbreviation Code
	.byte	53                              # DW_TAG_volatile_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	88                              # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.ascii	"\207B"                         # DW_AT_GNU_vector
	.byte	25                              # DW_FORM_flag_present
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	89                              # Abbreviation Code
	.byte	53                              # DW_TAG_volatile_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	90                              # Abbreviation Code
	.byte	31                              # DW_TAG_ptr_to_member_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	29                              # DW_AT_containing_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	91                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	119                             # DW_AT_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	92                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	120                             # DW_AT_rvalue_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	93                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	119                             # DW_AT_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	94                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	120                             # DW_AT_rvalue_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	95                              # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	38                              # DW_FORM_strx2
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	13                              # DW_AT_bit_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	96                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	97                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	98                              # Abbreviation Code
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
	.byte	99                              # Abbreviation Code
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
	.byte	1                               # Abbrev [1] 0xc:0x29ba DW_TAG_compile_unit
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
	.byte	56                              # DW_AT_decl_line
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
	.byte	61                              # DW_AT_decl_line
	.byte	6                               # Abbrev [6] 0x44:0x3 DW_TAG_enumerator
	.byte	6                               # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	7                               # Abbrev [7] 0x48:0x2 DW_TAG_structure_type
	.byte	240                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0x4b:0x4 DW_TAG_base_type
	.byte	5                               # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	8                               # Abbrev [8] 0x4f:0x63 DW_TAG_namespace
	.byte	8                               # DW_AT_name
	.byte	5                               # Abbrev [5] 0x51:0x13 DW_TAG_enumeration_type
	.long	54                              # DW_AT_type
	.byte	12                              # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	29                              # DW_AT_decl_line
	.byte	9                               # Abbrev [9] 0x5a:0x3 DW_TAG_enumerator
	.byte	9                               # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	9                               # Abbrev [9] 0x5d:0x3 DW_TAG_enumerator
	.byte	10                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	9                               # Abbrev [9] 0x60:0x3 DW_TAG_enumerator
	.byte	11                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x64:0x13 DW_TAG_enumeration_type
	.long	54                              # DW_AT_type
                                        # DW_AT_enum_class
	.byte	13                              # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	30                              # DW_AT_decl_line
	.byte	9                               # Abbrev [9] 0x6d:0x3 DW_TAG_enumerator
	.byte	9                               # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	9                               # Abbrev [9] 0x70:0x3 DW_TAG_enumerator
	.byte	10                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	9                               # Abbrev [9] 0x73:0x3 DW_TAG_enumerator
	.byte	11                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x77:0xe DW_TAG_enumeration_type
	.long	178                             # DW_AT_type
	.byte	16                              # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	32                              # DW_AT_decl_line
	.byte	6                               # Abbrev [6] 0x80:0x4 DW_TAG_enumerator
	.byte	15                              # DW_AT_name
	.ascii	"\377\001"                      # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	11                              # Abbrev [11] 0x85:0x12 DW_TAG_enumeration_type
	.long	54                              # DW_AT_type
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	31                              # DW_AT_decl_line
	.byte	9                               # Abbrev [9] 0x8d:0x3 DW_TAG_enumerator
	.byte	17                              # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	9                               # Abbrev [9] 0x90:0x3 DW_TAG_enumerator
	.byte	18                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	9                               # Abbrev [9] 0x93:0x3 DW_TAG_enumerator
	.byte	19                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x97:0x13 DW_TAG_subprogram
	.byte	101                             # DW_AT_low_pc
	.long	.Lfunc_end100-.Lfunc_begin100   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	294                             # DW_AT_linkage_name
	.short	295                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	13                              # Abbrev [13] 0xa5:0x4 DW_TAG_GNU_template_template_param
	.byte	20                              # DW_AT_name
	.short	293                             # DW_AT_GNU_template_name
	.byte	0                               # End Of Children Mark
	.byte	7                               # Abbrev [7] 0xaa:0x2 DW_TAG_structure_type
	.byte	144                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	8                               # Abbrev [8] 0xac:0x5 DW_TAG_namespace
	.byte	151                             # DW_AT_name
	.byte	7                               # Abbrev [7] 0xae:0x2 DW_TAG_structure_type
	.byte	144                             # DW_AT_name
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
	.byte	50                              # DW_AT_decl_line
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
	.byte	14                              # Abbrev [14] 0xce:0x14 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	24                              # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	167                             # DW_AT_decl_line
	.byte	17                              # Abbrev [17] 0xd4:0xd DW_TAG_subprogram
	.byte	81                              # DW_AT_linkage_name
	.byte	82                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	169                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	18                              # Abbrev [18] 0xd9:0x2 DW_TAG_template_type_parameter
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xdb:0x5 DW_TAG_formal_parameter
	.long	5528                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0xe2:0x17 DW_TAG_subprogram
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	120                             # DW_AT_linkage_name
	.byte	121                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	144                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	21                              # Abbrev [21] 0xee:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	0                               # DW_AT_decl_file
	.byte	144                             # DW_AT_decl_line
	.long	7298                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	22                              # Abbrev [22] 0xf9:0x8a DW_TAG_subprogram
	.byte	2                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	122                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	190                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
                                        # DW_AT_external
	.byte	23                              # Abbrev [23] 0x108:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.byte	180                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	191                             # DW_AT_decl_line
	.long	376                             # DW_AT_type
	.byte	24                              # Abbrev [24] 0x113:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	126
	.short	385                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	192                             # DW_AT_decl_line
	.long	371                             # DW_AT_type
	.byte	24                              # Abbrev [24] 0x11f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	125
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	224                             # DW_AT_decl_line
	.long	7963                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x12b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	239                             # DW_AT_decl_line
	.long	7413                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x137:0xd DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	123
	.short	390                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.short	264                             # DW_AT_decl_line
	.long	3243                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x144:0xd DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	122
	.short	391                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.short	299                             # DW_AT_decl_line
	.long	7979                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x151:0xd DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	121
	.short	393                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.short	308                             # DW_AT_decl_line
	.long	206                             # DW_AT_type
	.byte	25                              # Abbrev [25] 0x15e:0xd DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	394                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.short	329                             # DW_AT_decl_line
	.long	7988                            # DW_AT_type
	.byte	26                              # Abbrev [26] 0x16b:0x8 DW_TAG_imported_module
	.byte	0                               # DW_AT_decl_file
	.short	296                             # DW_AT_decl_line
	.long	79                              # DW_AT_import
	.byte	27                              # Abbrev [27] 0x173:0x5 DW_TAG_class_type
	.byte	5                               # DW_AT_calling_convention
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	192                             # DW_AT_decl_line
	.byte	28                              # Abbrev [28] 0x178:0x5 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	191                             # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x17d:0x3 DW_TAG_structure_type
	.short	281                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	7                               # Abbrev [7] 0x180:0x2 DW_TAG_structure_type
	.byte	85                              # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x183:0x2d DW_TAG_subprogram
	.byte	3                               # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	123                             # DW_AT_linkage_name
	.byte	124                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x18f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	7313                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x19b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8029                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1a7:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1a9:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x1b0:0x2d DW_TAG_subprogram
	.byte	4                               # DW_AT_low_pc
	.long	.Lfunc_end3-.Lfunc_begin3       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	125                             # DW_AT_linkage_name
	.byte	126                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1bc:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	3908                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1c8:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8046                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1d4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1d6:0x5 DW_TAG_template_type_parameter
	.long	3923                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x1dd:0x2d DW_TAG_subprogram
	.byte	5                               # DW_AT_low_pc
	.long	.Lfunc_end4-.Lfunc_begin4       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	127                             # DW_AT_linkage_name
	.byte	128                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1e9:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8063                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1f5:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8079                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x201:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x203:0x5 DW_TAG_template_type_parameter
	.long	202                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x20a:0x2d DW_TAG_subprogram
	.byte	6                               # DW_AT_low_pc
	.long	.Lfunc_end5-.Lfunc_begin5       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	130                             # DW_AT_linkage_name
	.byte	131                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x216:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8096                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x222:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8112                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x22e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x230:0x5 DW_TAG_template_type_parameter
	.long	7294                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x237:0x2d DW_TAG_subprogram
	.byte	7                               # DW_AT_low_pc
	.long	.Lfunc_end6-.Lfunc_begin6       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	132                             # DW_AT_linkage_name
	.byte	133                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x243:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8129                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x24f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8145                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x25b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x25d:0x5 DW_TAG_template_type_parameter
	.long	7072                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x264:0x2d DW_TAG_subprogram
	.byte	8                               # DW_AT_low_pc
	.long	.Lfunc_end7-.Lfunc_begin7       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	134                             # DW_AT_linkage_name
	.byte	135                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x270:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8162                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x27c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8178                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x288:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x28a:0x5 DW_TAG_template_type_parameter
	.long	7052                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x291:0x2d DW_TAG_subprogram
	.byte	9                               # DW_AT_low_pc
	.long	.Lfunc_end8-.Lfunc_begin8       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	136                             # DW_AT_linkage_name
	.byte	137                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x29d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8195                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x2a9:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8211                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2b5:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2b7:0x5 DW_TAG_template_type_parameter
	.long	75                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x2be:0x2d DW_TAG_subprogram
	.byte	10                              # DW_AT_low_pc
	.long	.Lfunc_end9-.Lfunc_begin9       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	139                             # DW_AT_linkage_name
	.byte	140                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x2ca:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8228                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x2d6:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8244                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2e2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2e4:0x5 DW_TAG_template_type_parameter
	.long	7298                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x2eb:0x2d DW_TAG_subprogram
	.byte	11                              # DW_AT_low_pc
	.long	.Lfunc_end10-.Lfunc_begin10     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	142                             # DW_AT_linkage_name
	.byte	143                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x2f7:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8261                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x303:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8277                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x30f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x311:0x5 DW_TAG_template_type_parameter
	.long	7302                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x318:0x2d DW_TAG_subprogram
	.byte	12                              # DW_AT_low_pc
	.long	.Lfunc_end11-.Lfunc_begin11     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	145                             # DW_AT_linkage_name
	.byte	146                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x324:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8294                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x330:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8310                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x33c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x33e:0x5 DW_TAG_template_type_parameter
	.long	7306                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x345:0x2d DW_TAG_subprogram
	.byte	13                              # DW_AT_low_pc
	.long	.Lfunc_end12-.Lfunc_begin12     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	147                             # DW_AT_linkage_name
	.byte	148                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x351:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8327                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x35d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8343                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x369:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x36b:0x5 DW_TAG_template_type_parameter
	.long	170                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x372:0x2d DW_TAG_subprogram
	.byte	14                              # DW_AT_low_pc
	.long	.Lfunc_end13-.Lfunc_begin13     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	149                             # DW_AT_linkage_name
	.byte	150                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x37e:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8360                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x38a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8376                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x396:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x398:0x5 DW_TAG_template_type_parameter
	.long	7308                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x39f:0x2d DW_TAG_subprogram
	.byte	15                              # DW_AT_low_pc
	.long	.Lfunc_end14-.Lfunc_begin14     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	152                             # DW_AT_linkage_name
	.byte	153                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x3ab:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8393                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x3b7:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8409                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x3c3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x3c5:0x5 DW_TAG_template_type_parameter
	.long	174                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x3cc:0x2d DW_TAG_subprogram
	.byte	16                              # DW_AT_low_pc
	.long	.Lfunc_end15-.Lfunc_begin15     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	155                             # DW_AT_linkage_name
	.byte	156                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x3d8:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8426                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x3e4:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8442                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x3f0:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x3f2:0x5 DW_TAG_template_type_parameter
	.long	7313                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x3f9:0x32 DW_TAG_subprogram
	.byte	17                              # DW_AT_low_pc
	.long	.Lfunc_end16-.Lfunc_begin16     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	157                             # DW_AT_linkage_name
	.byte	158                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x405:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8459                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x411:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8480                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x41d:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x41f:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	31                              # Abbrev [31] 0x424:0x5 DW_TAG_template_type_parameter
	.long	3923                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x42b:0x2d DW_TAG_subprogram
	.byte	18                              # DW_AT_low_pc
	.long	.Lfunc_end17-.Lfunc_begin17     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	159                             # DW_AT_linkage_name
	.byte	160                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x437:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	7524                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x443:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8502                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x44f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x451:0x5 DW_TAG_template_type_parameter
	.long	7328                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x458:0x2d DW_TAG_subprogram
	.byte	19                              # DW_AT_low_pc
	.long	.Lfunc_end18-.Lfunc_begin18     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	161                             # DW_AT_linkage_name
	.byte	162                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x464:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8519                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x470:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8535                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x47c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x47e:0x5 DW_TAG_template_type_parameter
	.long	7333                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x485:0x2d DW_TAG_subprogram
	.byte	20                              # DW_AT_low_pc
	.long	.Lfunc_end19-.Lfunc_begin19     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	163                             # DW_AT_linkage_name
	.byte	164                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x491:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8552                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x49d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8568                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x4a9:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x4ab:0x5 DW_TAG_template_type_parameter
	.long	7338                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x4b2:0x2d DW_TAG_subprogram
	.byte	21                              # DW_AT_low_pc
	.long	.Lfunc_end20-.Lfunc_begin20     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	165                             # DW_AT_linkage_name
	.byte	166                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x4be:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8585                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x4ca:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8601                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x4d6:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x4d8:0x5 DW_TAG_template_type_parameter
	.long	7343                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x4df:0x2d DW_TAG_subprogram
	.byte	22                              # DW_AT_low_pc
	.long	.Lfunc_end21-.Lfunc_begin21     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	168                             # DW_AT_linkage_name
	.byte	169                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x4eb:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8618                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x4f7:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8634                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x503:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x505:0x5 DW_TAG_template_type_parameter
	.long	7348                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x50c:0x29 DW_TAG_subprogram
	.byte	23                              # DW_AT_low_pc
	.long	.Lfunc_end22-.Lfunc_begin22     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	170                             # DW_AT_linkage_name
	.byte	171                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x518:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8651                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x524:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8663                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x530:0x4 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	32                              # Abbrev [32] 0x532:0x1 DW_TAG_template_type_parameter
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x535:0x2d DW_TAG_subprogram
	.byte	24                              # DW_AT_low_pc
	.long	.Lfunc_end23-.Lfunc_begin23     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	174                             # DW_AT_linkage_name
	.byte	175                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x541:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8676                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x54d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8692                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x559:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x55b:0x5 DW_TAG_template_type_parameter
	.long	7370                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x562:0x2d DW_TAG_subprogram
	.byte	25                              # DW_AT_low_pc
	.long	.Lfunc_end24-.Lfunc_begin24     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	176                             # DW_AT_linkage_name
	.byte	177                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x56e:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8709                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x57a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8725                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x586:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x588:0x5 DW_TAG_template_type_parameter
	.long	4793                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x58f:0x1b DW_TAG_subprogram
	.byte	26                              # DW_AT_low_pc
	.long	.Lfunc_end25-.Lfunc_begin25     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	178                             # DW_AT_linkage_name
	.byte	179                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	40                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	33                              # Abbrev [33] 0x59b:0x7 DW_TAG_template_value_parameter
	.long	202                             # DW_AT_type
	.byte	22                              # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0x5a2:0x7 DW_TAG_template_value_parameter
	.long	54                              # DW_AT_type
	.byte	3                               # DW_AT_name
	.byte	3                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x5aa:0x22 DW_TAG_subprogram
	.byte	27                              # DW_AT_low_pc
	.long	.Lfunc_end26-.Lfunc_begin26     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	181                             # DW_AT_linkage_name
	.byte	182                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x5b6:0x6 DW_TAG_template_type_parameter
	.long	81                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x5bc:0xf DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	35                              # Abbrev [35] 0x5be:0x6 DW_TAG_template_value_parameter
	.long	81                              # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	35                              # Abbrev [35] 0x5c4:0x6 DW_TAG_template_value_parameter
	.long	81                              # DW_AT_type
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x5cc:0x22 DW_TAG_subprogram
	.byte	28                              # DW_AT_low_pc
	.long	.Lfunc_end27-.Lfunc_begin27     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	183                             # DW_AT_linkage_name
	.byte	184                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x5d8:0x6 DW_TAG_template_type_parameter
	.long	100                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x5de:0xf DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	35                              # Abbrev [35] 0x5e0:0x6 DW_TAG_template_value_parameter
	.long	100                             # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	35                              # Abbrev [35] 0x5e6:0x6 DW_TAG_template_value_parameter
	.long	100                             # DW_AT_type
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x5ee:0x1d DW_TAG_subprogram
	.byte	29                              # DW_AT_low_pc
	.long	.Lfunc_end28-.Lfunc_begin28     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	185                             # DW_AT_linkage_name
	.byte	186                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x5fa:0x6 DW_TAG_template_type_parameter
	.long	119                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x600:0xa DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	36                              # Abbrev [36] 0x602:0x7 DW_TAG_template_value_parameter
	.long	119                             # DW_AT_type
	.ascii	"\377\001"                      # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	37                              # Abbrev [37] 0x60b:0x22 DW_TAG_subprogram
	.byte	30                              # DW_AT_low_pc
	.long	.Lfunc_end29-.Lfunc_begin29     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	187                             # DW_AT_linkage_name
	.byte	188                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
	.byte	15                              # Abbrev [15] 0x617:0x6 DW_TAG_template_type_parameter
	.long	133                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x61d:0xf DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	35                              # Abbrev [35] 0x61f:0x6 DW_TAG_template_value_parameter
	.long	133                             # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	35                              # Abbrev [35] 0x625:0x6 DW_TAG_template_value_parameter
	.long	133                             # DW_AT_type
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	37                              # Abbrev [37] 0x62d:0x1c DW_TAG_subprogram
	.byte	31                              # DW_AT_low_pc
	.long	.Lfunc_end30-.Lfunc_begin30     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	189                             # DW_AT_linkage_name
	.byte	190                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
	.byte	15                              # Abbrev [15] 0x639:0x6 DW_TAG_template_type_parameter
	.long	59                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x63f:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	36                              # Abbrev [36] 0x641:0x6 DW_TAG_template_value_parameter
	.long	59                              # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x649:0x1f DW_TAG_subprogram
	.byte	32                              # DW_AT_low_pc
	.long	.Lfunc_end31-.Lfunc_begin31     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	191                             # DW_AT_linkage_name
	.byte	192                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x655:0x6 DW_TAG_template_type_parameter
	.long	7328                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x65b:0xc DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	38                              # Abbrev [38] 0x65d:0x9 DW_TAG_template_value_parameter
	.long	7328                            # DW_AT_type
	.byte	3                               # DW_AT_location
	.byte	161
	.byte	0
	.byte	159
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x668:0x1c DW_TAG_subprogram
	.byte	33                              # DW_AT_low_pc
	.long	.Lfunc_end32-.Lfunc_begin32     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	193                             # DW_AT_linkage_name
	.byte	194                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x674:0x6 DW_TAG_template_type_parameter
	.long	7328                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x67a:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	36                              # Abbrev [36] 0x67c:0x6 DW_TAG_template_value_parameter
	.long	7328                            # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x684:0x1c DW_TAG_subprogram
	.byte	34                              # DW_AT_low_pc
	.long	.Lfunc_end33-.Lfunc_begin33     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	195                             # DW_AT_linkage_name
	.byte	196                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x690:0x6 DW_TAG_template_type_parameter
	.long	4793                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x696:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	36                              # Abbrev [36] 0x698:0x6 DW_TAG_template_value_parameter
	.long	4793                            # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x6a0:0x1c DW_TAG_subprogram
	.byte	35                              # DW_AT_low_pc
	.long	.Lfunc_end34-.Lfunc_begin34     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	197                             # DW_AT_linkage_name
	.byte	198                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x6ac:0x6 DW_TAG_template_type_parameter
	.long	7298                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x6b2:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	36                              # Abbrev [36] 0x6b4:0x6 DW_TAG_template_value_parameter
	.long	7298                            # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x6bc:0x1c DW_TAG_subprogram
	.byte	36                              # DW_AT_low_pc
	.long	.Lfunc_end35-.Lfunc_begin35     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	199                             # DW_AT_linkage_name
	.byte	200                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x6c8:0x6 DW_TAG_template_type_parameter
	.long	7072                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x6ce:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	35                              # Abbrev [35] 0x6d0:0x6 DW_TAG_template_value_parameter
	.long	7072                            # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x6d8:0x1c DW_TAG_subprogram
	.byte	37                              # DW_AT_low_pc
	.long	.Lfunc_end36-.Lfunc_begin36     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	201                             # DW_AT_linkage_name
	.byte	202                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x6e4:0x6 DW_TAG_template_type_parameter
	.long	75                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x6ea:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	36                              # Abbrev [36] 0x6ec:0x6 DW_TAG_template_value_parameter
	.long	75                              # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x6f4:0x1c DW_TAG_subprogram
	.byte	38                              # DW_AT_low_pc
	.long	.Lfunc_end37-.Lfunc_begin37     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	203                             # DW_AT_linkage_name
	.byte	204                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x700:0x6 DW_TAG_template_type_parameter
	.long	7052                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x706:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	35                              # Abbrev [35] 0x708:0x6 DW_TAG_template_value_parameter
	.long	7052                            # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x710:0x1c DW_TAG_subprogram
	.byte	39                              # DW_AT_low_pc
	.long	.Lfunc_end38-.Lfunc_begin38     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	205                             # DW_AT_linkage_name
	.byte	206                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x71c:0x6 DW_TAG_template_type_parameter
	.long	178                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x722:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	36                              # Abbrev [36] 0x724:0x6 DW_TAG_template_value_parameter
	.long	178                             # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x72c:0x1c DW_TAG_subprogram
	.byte	40                              # DW_AT_low_pc
	.long	.Lfunc_end39-.Lfunc_begin39     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	207                             # DW_AT_linkage_name
	.byte	208                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x738:0x6 DW_TAG_template_type_parameter
	.long	7040                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x73e:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	35                              # Abbrev [35] 0x740:0x6 DW_TAG_template_value_parameter
	.long	7040                            # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x748:0x22 DW_TAG_subprogram
	.byte	41                              # DW_AT_low_pc
	.long	.Lfunc_end40-.Lfunc_begin40     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	209                             # DW_AT_linkage_name
	.byte	210                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x754:0x6 DW_TAG_template_type_parameter
	.long	7093                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x75a:0xf DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	36                              # Abbrev [36] 0x75c:0x6 DW_TAG_template_value_parameter
	.long	7093                            # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	36                              # Abbrev [36] 0x762:0x6 DW_TAG_template_value_parameter
	.long	7093                            # DW_AT_type
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x76a:0x5a DW_TAG_subprogram
	.byte	42                              # DW_AT_low_pc
	.long	.Lfunc_end41-.Lfunc_begin41     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	212                             # DW_AT_linkage_name
	.byte	213                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x776:0x6 DW_TAG_template_type_parameter
	.long	7373                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x77c:0x47 DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	35                              # Abbrev [35] 0x77e:0x6 DW_TAG_template_value_parameter
	.long	7373                            # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	35                              # Abbrev [35] 0x784:0x6 DW_TAG_template_value_parameter
	.long	7373                            # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	35                              # Abbrev [35] 0x78a:0x6 DW_TAG_template_value_parameter
	.long	7373                            # DW_AT_type
	.byte	6                               # DW_AT_const_value
	.byte	35                              # Abbrev [35] 0x790:0x6 DW_TAG_template_value_parameter
	.long	7373                            # DW_AT_type
	.byte	7                               # DW_AT_const_value
	.byte	35                              # Abbrev [35] 0x796:0x6 DW_TAG_template_value_parameter
	.long	7373                            # DW_AT_type
	.byte	13                              # DW_AT_const_value
	.byte	35                              # Abbrev [35] 0x79c:0x6 DW_TAG_template_value_parameter
	.long	7373                            # DW_AT_type
	.byte	14                              # DW_AT_const_value
	.byte	35                              # Abbrev [35] 0x7a2:0x6 DW_TAG_template_value_parameter
	.long	7373                            # DW_AT_type
	.byte	31                              # DW_AT_const_value
	.byte	35                              # Abbrev [35] 0x7a8:0x6 DW_TAG_template_value_parameter
	.long	7373                            # DW_AT_type
	.byte	32                              # DW_AT_const_value
	.byte	35                              # Abbrev [35] 0x7ae:0x6 DW_TAG_template_value_parameter
	.long	7373                            # DW_AT_type
	.byte	33                              # DW_AT_const_value
	.byte	35                              # Abbrev [35] 0x7b4:0x7 DW_TAG_template_value_parameter
	.long	7373                            # DW_AT_type
	.asciz	"\377"                          # DW_AT_const_value
	.byte	35                              # Abbrev [35] 0x7bb:0x7 DW_TAG_template_value_parameter
	.long	7373                            # DW_AT_type
	.ascii	"\200\177"                      # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x7c4:0x2c DW_TAG_subprogram
	.byte	43                              # DW_AT_low_pc
	.long	.Lfunc_end42-.Lfunc_begin42     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	215                             # DW_AT_linkage_name
	.byte	216                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x7d0:0x6 DW_TAG_template_type_parameter
	.long	7377                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x7d6:0x19 DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	39                              # Abbrev [39] 0x7d8:0x16 DW_TAG_template_value_parameter
	.long	7377                            # DW_AT_type
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
	.byte	20                              # Abbrev [20] 0x7f0:0x19 DW_TAG_subprogram
	.byte	44                              # DW_AT_low_pc
	.long	.Lfunc_end43-.Lfunc_begin43     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	217                             # DW_AT_linkage_name
	.byte	218                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	46                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x7fc:0x6 DW_TAG_template_type_parameter
	.long	75                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	36                              # Abbrev [36] 0x802:0x6 DW_TAG_template_value_parameter
	.long	75                              # DW_AT_type
	.byte	3                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x809:0x2d DW_TAG_subprogram
	.byte	45                              # DW_AT_low_pc
	.long	.Lfunc_end44-.Lfunc_begin44     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	219                             # DW_AT_linkage_name
	.byte	220                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x815:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8742                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x821:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8758                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x82d:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x82f:0x5 DW_TAG_template_type_parameter
	.long	182                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x836:0x2d DW_TAG_subprogram
	.byte	46                              # DW_AT_low_pc
	.long	.Lfunc_end45-.Lfunc_begin45     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	222                             # DW_AT_linkage_name
	.byte	223                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x842:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8775                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x84e:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8791                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x85a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x85c:0x5 DW_TAG_template_type_parameter
	.long	7381                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	37                              # Abbrev [37] 0x863:0x2d DW_TAG_subprogram
	.byte	47                              # DW_AT_low_pc
	.long	.Lfunc_end46-.Lfunc_begin46     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	224                             # DW_AT_linkage_name
	.byte	225                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x86f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	7618                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x87b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8808                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x887:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x889:0x5 DW_TAG_template_type_parameter
	.long	371                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	37                              # Abbrev [37] 0x890:0x2d DW_TAG_subprogram
	.byte	48                              # DW_AT_low_pc
	.long	.Lfunc_end47-.Lfunc_begin47     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	228                             # DW_AT_linkage_name
	.byte	229                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x89c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8825                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x8a8:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8841                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x8b4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x8b6:0x5 DW_TAG_template_type_parameter
	.long	7397                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x8bd:0x2d DW_TAG_subprogram
	.byte	49                              # DW_AT_low_pc
	.long	.Lfunc_end48-.Lfunc_begin48     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	230                             # DW_AT_linkage_name
	.byte	231                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x8c9:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8858                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x8d5:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8874                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x8e1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x8e3:0x5 DW_TAG_template_type_parameter
	.long	7433                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x8ea:0x2d DW_TAG_subprogram
	.byte	50                              # DW_AT_low_pc
	.long	.Lfunc_end49-.Lfunc_begin49     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	232                             # DW_AT_linkage_name
	.byte	233                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x8f6:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8891                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x902:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8907                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x90e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x910:0x5 DW_TAG_template_type_parameter
	.long	7444                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x917:0x2d DW_TAG_subprogram
	.byte	51                              # DW_AT_low_pc
	.long	.Lfunc_end50-.Lfunc_begin50     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	234                             # DW_AT_linkage_name
	.byte	235                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x923:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8924                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x92f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8940                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x93b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x93d:0x5 DW_TAG_template_type_parameter
	.long	7447                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x944:0x2d DW_TAG_subprogram
	.byte	52                              # DW_AT_low_pc
	.long	.Lfunc_end51-.Lfunc_begin51     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	236                             # DW_AT_linkage_name
	.byte	237                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x950:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8957                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x95c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	8973                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x968:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x96a:0x5 DW_TAG_template_type_parameter
	.long	7455                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x971:0x2d DW_TAG_subprogram
	.byte	53                              # DW_AT_low_pc
	.long	.Lfunc_end52-.Lfunc_begin52     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	238                             # DW_AT_linkage_name
	.byte	239                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x97d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	8990                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x989:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9006                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x995:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x997:0x5 DW_TAG_template_type_parameter
	.long	7460                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	37                              # Abbrev [37] 0x99e:0x2d DW_TAG_subprogram
	.byte	54                              # DW_AT_low_pc
	.long	.Lfunc_end53-.Lfunc_begin53     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	241                             # DW_AT_linkage_name
	.byte	242                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x9aa:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9023                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x9b6:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9039                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x9c2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x9c4:0x5 DW_TAG_template_type_parameter
	.long	72                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x9cb:0x2d DW_TAG_subprogram
	.byte	55                              # DW_AT_low_pc
	.long	.Lfunc_end54-.Lfunc_begin54     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	244                             # DW_AT_linkage_name
	.byte	245                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x9d7:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9056                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x9e3:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9072                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x9ef:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x9f1:0x5 DW_TAG_template_type_parameter
	.long	7470                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x9f8:0x32 DW_TAG_subprogram
	.byte	56                              # DW_AT_low_pc
	.long	.Lfunc_end55-.Lfunc_begin55     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	246                             # DW_AT_linkage_name
	.byte	247                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0xa04:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9089                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0xa10:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9110                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0xa1c:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0xa1e:0x5 DW_TAG_template_type_parameter
	.long	7472                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0xa23:0x5 DW_TAG_template_type_parameter
	.long	7472                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0xa2a:0x32 DW_TAG_subprogram
	.byte	57                              # DW_AT_low_pc
	.long	.Lfunc_end56-.Lfunc_begin56     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	248                             # DW_AT_linkage_name
	.byte	249                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0xa36:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9132                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0xa42:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9153                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0xa4e:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0xa50:0x5 DW_TAG_template_type_parameter
	.long	7472                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0xa55:0x5 DW_TAG_template_type_parameter
	.long	7477                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0xa5c:0x2d DW_TAG_subprogram
	.byte	58                              # DW_AT_low_pc
	.long	.Lfunc_end57-.Lfunc_begin57     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	250                             # DW_AT_linkage_name
	.byte	251                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0xa68:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9175                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0xa74:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9191                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0xa80:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0xa82:0x5 DW_TAG_template_type_parameter
	.long	7482                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0xa89:0x2d DW_TAG_subprogram
	.byte	59                              # DW_AT_low_pc
	.long	.Lfunc_end58-.Lfunc_begin58     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	252                             # DW_AT_linkage_name
	.byte	253                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0xa95:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9208                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0xaa1:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9224                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0xaad:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0xaaf:0x5 DW_TAG_template_type_parameter
	.long	7487                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0xab6:0x2d DW_TAG_subprogram
	.byte	60                              # DW_AT_low_pc
	.long	.Lfunc_end59-.Lfunc_begin59     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	254                             # DW_AT_linkage_name
	.byte	255                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0xac2:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9241                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0xace:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9257                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0xada:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0xadc:0x5 DW_TAG_template_type_parameter
	.long	7503                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xae3:0x2f DW_TAG_subprogram
	.byte	61                              # DW_AT_low_pc
	.long	.Lfunc_end60-.Lfunc_begin60     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	256                             # DW_AT_linkage_name
	.short	257                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0xaf1:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9274                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0xafd:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9290                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0xb09:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0xb0b:0x5 DW_TAG_template_type_parameter
	.long	7504                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	40                              # Abbrev [40] 0xb12:0x2f DW_TAG_subprogram
	.byte	62                              # DW_AT_low_pc
	.long	.Lfunc_end61-.Lfunc_begin61     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	258                             # DW_AT_linkage_name
	.short	259                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0xb20:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9307                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0xb2c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9323                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0xb38:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0xb3a:0x5 DW_TAG_template_type_parameter
	.long	7509                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	40                              # Abbrev [40] 0xb41:0x2f DW_TAG_subprogram
	.byte	63                              # DW_AT_low_pc
	.long	.Lfunc_end62-.Lfunc_begin62     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	260                             # DW_AT_linkage_name
	.short	261                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0xb4f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9340                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0xb5b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9356                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0xb67:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0xb69:0x5 DW_TAG_template_type_parameter
	.long	376                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	40                              # Abbrev [40] 0xb70:0x2f DW_TAG_subprogram
	.byte	64                              # DW_AT_low_pc
	.long	.Lfunc_end63-.Lfunc_begin63     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	262                             # DW_AT_linkage_name
	.short	263                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0xb7e:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9373                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0xb8a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9389                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0xb96:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0xb98:0x5 DW_TAG_template_type_parameter
	.long	7514                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xb9f:0x1f DW_TAG_subprogram
	.byte	65                              # DW_AT_low_pc
	.long	.Lfunc_end64-.Lfunc_begin64     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	266                             # DW_AT_linkage_name
	.short	267                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	64                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	41                              # Abbrev [41] 0xbad:0x9 DW_TAG_GNU_template_parameter_pack
	.short	264                             # DW_AT_name
	.byte	31                              # Abbrev [31] 0xbb0:0x5 DW_TAG_template_type_parameter
	.long	7313                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0xbb6:0x7 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.short	265                             # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xbbe:0x19 DW_TAG_subprogram
	.byte	66                              # DW_AT_low_pc
	.long	.Lfunc_end65-.Lfunc_begin65     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	268                             # DW_AT_linkage_name
	.short	269                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	64                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	43                              # Abbrev [43] 0xbcc:0x3 DW_TAG_GNU_template_parameter_pack
	.short	264                             # DW_AT_name
	.byte	42                              # Abbrev [42] 0xbcf:0x7 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.short	265                             # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xbd7:0x19 DW_TAG_subprogram
	.byte	67                              # DW_AT_low_pc
	.long	.Lfunc_end66-.Lfunc_begin66     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	270                             # DW_AT_linkage_name
	.short	271                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	66                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	42                              # Abbrev [42] 0xbe5:0x7 DW_TAG_template_type_parameter
	.long	7313                            # DW_AT_type
	.short	264                             # DW_AT_name
	.byte	43                              # Abbrev [43] 0xbec:0x3 DW_TAG_GNU_template_parameter_pack
	.short	265                             # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xbf0:0x29 DW_TAG_subprogram
	.byte	68                              # DW_AT_low_pc
	.long	.Lfunc_end67-.Lfunc_begin67     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	272                             # DW_AT_linkage_name
	.short	273                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0xbfe:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	7921                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0xc0a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9406                            # DW_AT_type
	.byte	44                              # Abbrev [44] 0xc16:0x2 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xc19:0x34 DW_TAG_subprogram
	.byte	69                              # DW_AT_low_pc
	.long	.Lfunc_end68-.Lfunc_begin68     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	274                             # DW_AT_linkage_name
	.short	275                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0xc27:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9417                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0xc33:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9438                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0xc3f:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0xc41:0x5 DW_TAG_template_type_parameter
	.long	7497                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0xc46:0x5 DW_TAG_template_type_parameter
	.long	7497                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xc4d:0x2f DW_TAG_subprogram
	.byte	70                              # DW_AT_low_pc
	.long	.Lfunc_end69-.Lfunc_begin69     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	277                             # DW_AT_linkage_name
	.short	278                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0xc5b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9460                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0xc67:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9476                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0xc73:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0xc75:0x5 DW_TAG_template_type_parameter
	.long	7519                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xc7c:0x2f DW_TAG_subprogram
	.byte	71                              # DW_AT_low_pc
	.long	.Lfunc_end70-.Lfunc_begin70     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	279                             # DW_AT_linkage_name
	.short	280                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0xc8a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9493                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0xc96:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9509                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0xca2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0xca4:0x5 DW_TAG_template_type_parameter
	.long	7540                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	14                              # Abbrev [14] 0xcab:0x20e DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	25                              # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	67                              # DW_AT_decl_line
	.byte	17                              # Abbrev [17] 0xcb1:0x16 DW_TAG_subprogram
	.byte	26                              # DW_AT_linkage_name
	.byte	27                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	69                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xcb6:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xcbc:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	45                              # Abbrev [45] 0xcc1:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xcc7:0x16 DW_TAG_subprogram
	.byte	28                              # DW_AT_linkage_name
	.byte	29                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	72                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xccc:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xcd2:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	45                              # Abbrev [45] 0xcd7:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xcdd:0x16 DW_TAG_subprogram
	.byte	30                              # DW_AT_linkage_name
	.byte	31                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	75                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xce2:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xce8:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	45                              # Abbrev [45] 0xced:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0xcf3:0x15 DW_TAG_subprogram
	.byte	32                              # DW_AT_linkage_name
	.byte	33                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	78                              # DW_AT_decl_line
	.long	3903                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xcfc:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xd02:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xd08:0x16 DW_TAG_subprogram
	.byte	37                              # DW_AT_linkage_name
	.byte	38                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	82                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xd0d:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xd13:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	45                              # Abbrev [45] 0xd18:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xd1e:0x16 DW_TAG_subprogram
	.byte	39                              # DW_AT_linkage_name
	.byte	40                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	85                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xd23:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xd29:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	45                              # Abbrev [45] 0xd2e:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xd34:0x16 DW_TAG_subprogram
	.byte	41                              # DW_AT_linkage_name
	.byte	42                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	88                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xd39:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xd3f:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	45                              # Abbrev [45] 0xd44:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xd4a:0x16 DW_TAG_subprogram
	.byte	43                              # DW_AT_linkage_name
	.byte	44                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	91                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xd4f:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xd55:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	45                              # Abbrev [45] 0xd5a:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xd60:0x16 DW_TAG_subprogram
	.byte	45                              # DW_AT_linkage_name
	.byte	46                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	94                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xd65:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xd6b:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	45                              # Abbrev [45] 0xd70:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xd76:0x16 DW_TAG_subprogram
	.byte	47                              # DW_AT_linkage_name
	.byte	48                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	97                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xd7b:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xd81:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	45                              # Abbrev [45] 0xd86:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xd8c:0x16 DW_TAG_subprogram
	.byte	49                              # DW_AT_linkage_name
	.byte	50                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	100                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xd91:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xd97:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	45                              # Abbrev [45] 0xd9c:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xda2:0x11 DW_TAG_subprogram
	.byte	51                              # DW_AT_linkage_name
	.byte	52                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	103                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xda7:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xdad:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xdb3:0x11 DW_TAG_subprogram
	.byte	53                              # DW_AT_linkage_name
	.byte	54                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	106                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xdb8:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xdbe:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xdc4:0x16 DW_TAG_subprogram
	.byte	55                              # DW_AT_linkage_name
	.byte	56                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	109                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xdc9:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xdcf:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	45                              # Abbrev [45] 0xdd4:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xdda:0x16 DW_TAG_subprogram
	.byte	57                              # DW_AT_linkage_name
	.byte	58                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	112                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xddf:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xde5:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	45                              # Abbrev [45] 0xdea:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xdf0:0x16 DW_TAG_subprogram
	.byte	59                              # DW_AT_linkage_name
	.byte	60                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	115                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xdf5:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xdfb:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	45                              # Abbrev [45] 0xe00:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xe06:0x11 DW_TAG_subprogram
	.byte	61                              # DW_AT_linkage_name
	.byte	62                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	118                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xe0b:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xe11:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xe17:0x16 DW_TAG_subprogram
	.byte	63                              # DW_AT_linkage_name
	.byte	64                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	121                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xe1c:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xe22:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	45                              # Abbrev [45] 0xe27:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xe2d:0x16 DW_TAG_subprogram
	.byte	65                              # DW_AT_linkage_name
	.byte	66                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	124                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xe32:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xe38:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	45                              # Abbrev [45] 0xe3d:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0xe43:0x1a DW_TAG_subprogram
	.byte	67                              # DW_AT_linkage_name
	.byte	68                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	127                             # DW_AT_decl_line
	.long	4575                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xe4c:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	45                              # Abbrev [45] 0xe52:0x5 DW_TAG_formal_parameter
	.long	4580                            # DW_AT_type
	.byte	45                              # Abbrev [45] 0xe57:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0xe5d:0x1a DW_TAG_subprogram
	.byte	73                              # DW_AT_linkage_name
	.byte	74                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	134                             # DW_AT_decl_line
	.long	4575                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xe66:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	45                              # Abbrev [45] 0xe6c:0x5 DW_TAG_formal_parameter
	.long	4580                            # DW_AT_type
	.byte	45                              # Abbrev [45] 0xe71:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xe77:0x16 DW_TAG_subprogram
	.byte	75                              # DW_AT_linkage_name
	.byte	76                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	131                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xe7c:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	45                              # Abbrev [45] 0xe82:0x5 DW_TAG_formal_parameter
	.long	4575                            # DW_AT_type
	.byte	45                              # Abbrev [45] 0xe87:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0xe8d:0x16 DW_TAG_subprogram
	.byte	77                              # DW_AT_linkage_name
	.byte	78                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	138                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xe92:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	45                              # Abbrev [45] 0xe98:0x5 DW_TAG_formal_parameter
	.long	4575                            # DW_AT_type
	.byte	45                              # Abbrev [45] 0xe9d:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0xea3:0x15 DW_TAG_subprogram
	.byte	79                              # DW_AT_linkage_name
	.byte	80                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	141                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xeac:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	19                              # Abbrev [19] 0xeb2:0x5 DW_TAG_formal_parameter
	.long	3769                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0xeb9:0x5 DW_TAG_pointer_type
	.long	3243                            # DW_AT_type
	.byte	48                              # Abbrev [48] 0xebe:0x2b DW_TAG_subprogram
	.byte	72                              # DW_AT_low_pc
	.long	.Lfunc_end71-.Lfunc_begin71     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3790                            # DW_AT_object_pointer
	.long	3249                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0xece:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	21                              # Abbrev [21] 0xed8:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	69                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0xee2:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0xee9:0x2b DW_TAG_subprogram
	.byte	73                              # DW_AT_low_pc
	.long	.Lfunc_end72-.Lfunc_begin72     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3833                            # DW_AT_object_pointer
	.long	3271                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0xef9:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	21                              # Abbrev [21] 0xf03:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	72                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0xf0d:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0xf14:0x2b DW_TAG_subprogram
	.byte	74                              # DW_AT_low_pc
	.long	.Lfunc_end73-.Lfunc_begin73     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3876                            # DW_AT_object_pointer
	.long	3293                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0xf24:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	21                              # Abbrev [21] 0xf2e:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	75                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0xf38:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0xf3f:0x5 DW_TAG_pointer_type
	.long	3908                            # DW_AT_type
	.byte	14                              # Abbrev [14] 0xf44:0xf DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	36                              # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0xf4a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0xf4c:0x5 DW_TAG_template_type_parameter
	.long	3923                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0xf53:0x4 DW_TAG_base_type
	.byte	35                              # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	48                              # Abbrev [48] 0xf57:0x21 DW_TAG_subprogram
	.byte	75                              # DW_AT_low_pc
	.long	.Lfunc_end74-.Lfunc_begin74     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3943                            # DW_AT_object_pointer
	.long	3315                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0xf67:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	15                              # Abbrev [15] 0xf71:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0xf78:0x2b DW_TAG_subprogram
	.byte	76                              # DW_AT_low_pc
	.long	.Lfunc_end75-.Lfunc_begin75     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3976                            # DW_AT_object_pointer
	.long	3336                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0xf88:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	21                              # Abbrev [21] 0xf92:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	82                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0xf9c:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0xfa3:0x2b DW_TAG_subprogram
	.byte	77                              # DW_AT_low_pc
	.long	.Lfunc_end76-.Lfunc_begin76     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4019                            # DW_AT_object_pointer
	.long	3358                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0xfb3:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	21                              # Abbrev [21] 0xfbd:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	85                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0xfc7:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0xfce:0x2b DW_TAG_subprogram
	.byte	78                              # DW_AT_low_pc
	.long	.Lfunc_end77-.Lfunc_begin77     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4062                            # DW_AT_object_pointer
	.long	3380                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0xfde:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	21                              # Abbrev [21] 0xfe8:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	88                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0xff2:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0xff9:0x2b DW_TAG_subprogram
	.byte	79                              # DW_AT_low_pc
	.long	.Lfunc_end78-.Lfunc_begin78     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4105                            # DW_AT_object_pointer
	.long	3402                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0x1009:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	21                              # Abbrev [21] 0x1013:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	91                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x101d:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0x1024:0x2b DW_TAG_subprogram
	.byte	80                              # DW_AT_low_pc
	.long	.Lfunc_end79-.Lfunc_begin79     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4148                            # DW_AT_object_pointer
	.long	3424                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0x1034:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	21                              # Abbrev [21] 0x103e:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	94                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x1048:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0x104f:0x2b DW_TAG_subprogram
	.byte	81                              # DW_AT_low_pc
	.long	.Lfunc_end80-.Lfunc_begin80     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4191                            # DW_AT_object_pointer
	.long	3446                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0x105f:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	21                              # Abbrev [21] 0x1069:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	97                              # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x1073:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0x107a:0x2b DW_TAG_subprogram
	.byte	82                              # DW_AT_low_pc
	.long	.Lfunc_end81-.Lfunc_begin81     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4234                            # DW_AT_object_pointer
	.long	3468                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0x108a:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	21                              # Abbrev [21] 0x1094:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	100                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x109e:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0x10a5:0x21 DW_TAG_subprogram
	.byte	83                              # DW_AT_low_pc
	.long	.Lfunc_end82-.Lfunc_begin82     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4277                            # DW_AT_object_pointer
	.long	3490                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0x10b5:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	15                              # Abbrev [15] 0x10bf:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0x10c6:0x21 DW_TAG_subprogram
	.byte	84                              # DW_AT_low_pc
	.long	.Lfunc_end83-.Lfunc_begin83     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4310                            # DW_AT_object_pointer
	.long	3507                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0x10d6:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	15                              # Abbrev [15] 0x10e0:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0x10e7:0x2b DW_TAG_subprogram
	.byte	85                              # DW_AT_low_pc
	.long	.Lfunc_end84-.Lfunc_begin84     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4343                            # DW_AT_object_pointer
	.long	3524                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0x10f7:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	21                              # Abbrev [21] 0x1101:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	109                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x110b:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0x1112:0x2b DW_TAG_subprogram
	.byte	86                              # DW_AT_low_pc
	.long	.Lfunc_end85-.Lfunc_begin85     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4386                            # DW_AT_object_pointer
	.long	3546                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0x1122:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	21                              # Abbrev [21] 0x112c:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	112                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x1136:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0x113d:0x2b DW_TAG_subprogram
	.byte	87                              # DW_AT_low_pc
	.long	.Lfunc_end86-.Lfunc_begin86     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4429                            # DW_AT_object_pointer
	.long	3568                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0x114d:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	21                              # Abbrev [21] 0x1157:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	115                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x1161:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0x1168:0x21 DW_TAG_subprogram
	.byte	88                              # DW_AT_low_pc
	.long	.Lfunc_end87-.Lfunc_begin87     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4472                            # DW_AT_object_pointer
	.long	3590                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0x1178:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	15                              # Abbrev [15] 0x1182:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0x1189:0x2b DW_TAG_subprogram
	.byte	89                              # DW_AT_low_pc
	.long	.Lfunc_end88-.Lfunc_begin88     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4505                            # DW_AT_object_pointer
	.long	3607                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0x1199:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	21                              # Abbrev [21] 0x11a3:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	121                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x11ad:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0x11b4:0x2b DW_TAG_subprogram
	.byte	90                              # DW_AT_low_pc
	.long	.Lfunc_end89-.Lfunc_begin89     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	4548                            # DW_AT_object_pointer
	.long	3629                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0x11c4:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9526                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	21                              # Abbrev [21] 0x11ce:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	124                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x11d8:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	50                              # Abbrev [50] 0x11df:0x1 DW_TAG_pointer_type
	.byte	8                               # Abbrev [8] 0x11e0:0xd9 DW_TAG_namespace
	.byte	69                              # DW_AT_name
	.byte	51                              # Abbrev [51] 0x11e2:0xd6 DW_TAG_namespace
	.byte	70                              # DW_AT_name
                                        # DW_AT_export_symbols
	.byte	52                              # Abbrev [52] 0x11e4:0x8 DW_TAG_typedef
	.long	4793                            # DW_AT_type
	.byte	72                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.byte	53                              # Abbrev [53] 0x11ec:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	158                             # DW_AT_decl_line
	.long	7031                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x11f3:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	159                             # DW_AT_decl_line
	.long	7044                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x11fa:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	160                             # DW_AT_decl_line
	.long	7056                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x1201:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	161                             # DW_AT_decl_line
	.long	7064                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x1208:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	163                             # DW_AT_decl_line
	.long	7076                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x120f:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	164                             # DW_AT_decl_line
	.long	7085                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x1216:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	165                             # DW_AT_decl_line
	.long	7097                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x121d:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	166                             # DW_AT_decl_line
	.long	7105                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x1224:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	168                             # DW_AT_decl_line
	.long	7113                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x122b:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	169                             # DW_AT_decl_line
	.long	7122                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x1232:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	170                             # DW_AT_decl_line
	.long	7131                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x1239:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	171                             # DW_AT_decl_line
	.long	7139                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x1240:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	173                             # DW_AT_decl_line
	.long	7147                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x1247:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	174                             # DW_AT_decl_line
	.long	7156                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x124e:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	175                             # DW_AT_decl_line
	.long	7165                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x1255:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	176                             # DW_AT_decl_line
	.long	7173                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x125c:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	178                             # DW_AT_decl_line
	.long	7181                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x1263:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	179                             # DW_AT_decl_line
	.long	7190                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x126a:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	180                             # DW_AT_decl_line
	.long	7199                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x1271:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	181                             # DW_AT_decl_line
	.long	7207                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x1278:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	183                             # DW_AT_decl_line
	.long	7215                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x127f:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	184                             # DW_AT_decl_line
	.long	7224                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x1286:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	185                             # DW_AT_decl_line
	.long	7233                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x128d:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	186                             # DW_AT_decl_line
	.long	7241                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x1294:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	188                             # DW_AT_decl_line
	.long	7249                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x129b:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	189                             # DW_AT_decl_line
	.long	7258                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x12a2:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	191                             # DW_AT_decl_line
	.long	7267                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x12a9:0x7 DW_TAG_imported_declaration
	.byte	3                               # DW_AT_decl_file
	.byte	192                             # DW_AT_decl_line
	.long	7276                            # DW_AT_import
	.byte	53                              # Abbrev [53] 0x12b0:0x7 DW_TAG_imported_declaration
	.byte	5                               # DW_AT_decl_file
	.byte	22                              # DW_AT_decl_line
	.long	7285                            # DW_AT_import
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0x12b9:0x4 DW_TAG_base_type
	.byte	71                              # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	54                              # Abbrev [54] 0x12bd:0x13 DW_TAG_subprogram
	.byte	91                              # DW_AT_low_pc
	.long	.Lfunc_end90-.Lfunc_begin90     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3651                            # DW_AT_specification
	.byte	15                              # Abbrev [15] 0x12c9:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	54                              # Abbrev [54] 0x12d0:0x13 DW_TAG_subprogram
	.byte	92                              # DW_AT_low_pc
	.long	.Lfunc_end91-.Lfunc_begin91     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3677                            # DW_AT_specification
	.byte	15                              # Abbrev [15] 0x12dc:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	54                              # Abbrev [54] 0x12e3:0x27 DW_TAG_subprogram
	.byte	93                              # DW_AT_low_pc
	.long	.Lfunc_end92-.Lfunc_begin92     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3703                            # DW_AT_specification
	.byte	21                              # Abbrev [21] 0x12ef:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	0                               # DW_AT_decl_file
	.byte	131                             # DW_AT_decl_line
	.long	4575                            # DW_AT_type
	.byte	21                              # Abbrev [21] 0x12f9:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	131                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x1303:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	54                              # Abbrev [54] 0x130a:0x27 DW_TAG_subprogram
	.byte	94                              # DW_AT_low_pc
	.long	.Lfunc_end93-.Lfunc_begin93     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3725                            # DW_AT_specification
	.byte	21                              # Abbrev [21] 0x1316:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	0                               # DW_AT_decl_file
	.byte	138                             # DW_AT_decl_line
	.long	4575                            # DW_AT_type
	.byte	21                              # Abbrev [21] 0x1320:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	0                               # DW_AT_decl_file
	.byte	138                             # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x132a:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	54                              # Abbrev [54] 0x1331:0x13 DW_TAG_subprogram
	.byte	95                              # DW_AT_low_pc
	.long	.Lfunc_end94-.Lfunc_begin94     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	3747                            # DW_AT_specification
	.byte	15                              # Abbrev [15] 0x133d:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	40                              # Abbrev [40] 0x1344:0x2f DW_TAG_subprogram
	.byte	96                              # DW_AT_low_pc
	.long	.Lfunc_end95-.Lfunc_begin95     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	282                             # DW_AT_linkage_name
	.short	283                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x1352:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9531                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x135e:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9547                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x136a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x136c:0x5 DW_TAG_template_type_parameter
	.long	381                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1373:0x2f DW_TAG_subprogram
	.byte	97                              # DW_AT_low_pc
	.long	.Lfunc_end96-.Lfunc_begin96     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	284                             # DW_AT_linkage_name
	.short	285                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1381:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9564                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x138d:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9580                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1399:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x139b:0x5 DW_TAG_template_type_parameter
	.long	7551                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x13a2:0x2f DW_TAG_subprogram
	.byte	98                              # DW_AT_low_pc
	.long	.Lfunc_end97-.Lfunc_begin97     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	286                             # DW_AT_linkage_name
	.short	287                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x13b0:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9597                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x13bc:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9613                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x13c8:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x13ca:0x5 DW_TAG_template_type_parameter
	.long	7556                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x13d1:0x13 DW_TAG_subprogram
	.byte	99                              # DW_AT_low_pc
	.long	.Lfunc_end98-.Lfunc_begin98     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	289                             # DW_AT_linkage_name
	.short	290                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	145                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	13                              # Abbrev [13] 0x13df:0x4 DW_TAG_GNU_template_template_param
	.byte	20                              # DW_AT_name
	.short	288                             # DW_AT_GNU_template_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x13e4:0x1a DW_TAG_subprogram
	.byte	100                             # DW_AT_low_pc
	.long	.Lfunc_end99-.Lfunc_begin99     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	291                             # DW_AT_linkage_name
	.short	292                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	146                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	13                              # Abbrev [13] 0x13f2:0x4 DW_TAG_GNU_template_template_param
	.byte	20                              # DW_AT_name
	.short	288                             # DW_AT_GNU_template_name
	.byte	42                              # Abbrev [42] 0x13f6:0x7 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.short	265                             # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x13fe:0x34 DW_TAG_subprogram
	.byte	102                             # DW_AT_low_pc
	.long	.Lfunc_end101-.Lfunc_begin101   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	296                             # DW_AT_linkage_name
	.short	297                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x140c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9630                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1418:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9651                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1424:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1426:0x5 DW_TAG_template_type_parameter
	.long	7328                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0x142b:0x5 DW_TAG_template_type_parameter
	.long	7561                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1432:0x2f DW_TAG_subprogram
	.byte	103                             # DW_AT_low_pc
	.long	.Lfunc_end102-.Lfunc_begin102   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	299                             # DW_AT_linkage_name
	.short	300                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1440:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9673                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x144c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9689                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1458:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x145a:0x5 DW_TAG_template_type_parameter
	.long	7566                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1461:0x13 DW_TAG_subprogram
	.byte	104                             # DW_AT_low_pc
	.long	.Lfunc_end103-.Lfunc_begin103   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	302                             # DW_AT_linkage_name
	.short	303                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	145                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	13                              # Abbrev [13] 0x146f:0x4 DW_TAG_GNU_template_template_param
	.byte	20                              # DW_AT_name
	.short	301                             # DW_AT_GNU_template_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1474:0x2f DW_TAG_subprogram
	.byte	105                             # DW_AT_low_pc
	.long	.Lfunc_end104-.Lfunc_begin104   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	304                             # DW_AT_linkage_name
	.short	305                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1482:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9706                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x148e:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9722                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x149a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x149c:0x5 DW_TAG_template_type_parameter
	.long	7580                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x14a3:0x39 DW_TAG_subprogram
	.byte	106                             # DW_AT_low_pc
	.long	.Lfunc_end105-.Lfunc_begin105   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	306                             # DW_AT_linkage_name
	.short	307                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x14b1:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9739                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x14bd:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9765                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x14c9:0x12 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x14cb:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	31                              # Abbrev [31] 0x14d0:0x5 DW_TAG_template_type_parameter
	.long	7072                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0x14d5:0x5 DW_TAG_template_type_parameter
	.long	7585                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x14dc:0x2f DW_TAG_subprogram
	.byte	107                             # DW_AT_low_pc
	.long	.Lfunc_end106-.Lfunc_begin106   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	308                             # DW_AT_linkage_name
	.short	309                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x14ea:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9792                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x14f6:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9808                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1502:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1504:0x5 DW_TAG_template_type_parameter
	.long	7590                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x150b:0x2f DW_TAG_subprogram
	.byte	108                             # DW_AT_low_pc
	.long	.Lfunc_end107-.Lfunc_begin107   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	310                             # DW_AT_linkage_name
	.short	311                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1519:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9825                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1525:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9841                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1531:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1533:0x5 DW_TAG_template_type_parameter
	.long	7602                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x153a:0x2f DW_TAG_subprogram
	.byte	109                             # DW_AT_low_pc
	.long	.Lfunc_end108-.Lfunc_begin108   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	312                             # DW_AT_linkage_name
	.short	313                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1548:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9858                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1554:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9874                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1560:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1562:0x5 DW_TAG_template_type_parameter
	.long	7612                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	40                              # Abbrev [40] 0x1569:0x2f DW_TAG_subprogram
	.byte	110                             # DW_AT_low_pc
	.long	.Lfunc_end109-.Lfunc_begin109   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	315                             # DW_AT_linkage_name
	.short	316                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x1577:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9891                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1583:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9907                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x158f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1591:0x5 DW_TAG_template_type_parameter
	.long	7618                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1598:0x5 DW_TAG_pointer_type
	.long	206                             # DW_AT_type
	.byte	55                              # Abbrev [55] 0x159d:0x1f DW_TAG_subprogram
	.byte	111                             # DW_AT_low_pc
	.long	.Lfunc_end110-.Lfunc_begin110   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	5551                            # DW_AT_object_pointer
	.short	317                             # DW_AT_linkage_name
	.long	212                             # DW_AT_specification
	.byte	49                              # Abbrev [49] 0x15af:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	9924                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	18                              # Abbrev [18] 0x15b9:0x2 DW_TAG_template_type_parameter
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x15bc:0x2f DW_TAG_subprogram
	.byte	112                             # DW_AT_low_pc
	.long	.Lfunc_end111-.Lfunc_begin111   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	318                             # DW_AT_linkage_name
	.short	319                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x15ca:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9929                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x15d6:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9945                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x15e2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x15e4:0x5 DW_TAG_template_type_parameter
	.long	7634                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x15eb:0x2f DW_TAG_subprogram
	.byte	113                             # DW_AT_low_pc
	.long	.Lfunc_end112-.Lfunc_begin112   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	320                             # DW_AT_linkage_name
	.short	321                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x15f9:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9962                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1605:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	9978                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1611:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1613:0x5 DW_TAG_template_type_parameter
	.long	7660                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x161a:0x2f DW_TAG_subprogram
	.byte	114                             # DW_AT_low_pc
	.long	.Lfunc_end113-.Lfunc_begin113   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	322                             # DW_AT_linkage_name
	.short	323                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1628:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	9995                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1634:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10011                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1640:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1642:0x5 DW_TAG_template_type_parameter
	.long	7686                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x1649:0x19 DW_TAG_subprogram
	.byte	115                             # DW_AT_low_pc
	.long	.Lfunc_end114-.Lfunc_begin114   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	324                             # DW_AT_linkage_name
	.short	325                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	164                             # DW_AT_decl_line
	.long	7504                            # DW_AT_type
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x165b:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1662:0x2f DW_TAG_subprogram
	.byte	116                             # DW_AT_low_pc
	.long	.Lfunc_end115-.Lfunc_begin115   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	326                             # DW_AT_linkage_name
	.short	327                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1670:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10028                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x167c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10044                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1688:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x168a:0x5 DW_TAG_template_type_parameter
	.long	7712                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1691:0x2f DW_TAG_subprogram
	.byte	117                             # DW_AT_low_pc
	.long	.Lfunc_end116-.Lfunc_begin116   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	328                             # DW_AT_linkage_name
	.short	329                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x169f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10061                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x16ab:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10077                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x16b7:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x16b9:0x5 DW_TAG_template_type_parameter
	.long	7717                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x16c0:0x2f DW_TAG_subprogram
	.byte	118                             # DW_AT_low_pc
	.long	.Lfunc_end117-.Lfunc_begin117   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	330                             # DW_AT_linkage_name
	.short	331                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x16ce:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10094                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x16da:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10110                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x16e6:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x16e8:0x5 DW_TAG_template_type_parameter
	.long	7739                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x16ef:0x2f DW_TAG_subprogram
	.byte	119                             # DW_AT_low_pc
	.long	.Lfunc_end118-.Lfunc_begin118   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	332                             # DW_AT_linkage_name
	.short	333                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x16fd:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10127                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1709:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10143                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1715:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1717:0x5 DW_TAG_template_type_parameter
	.long	7745                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x171e:0x2f DW_TAG_subprogram
	.byte	120                             # DW_AT_low_pc
	.long	.Lfunc_end119-.Lfunc_begin119   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	334                             # DW_AT_linkage_name
	.short	335                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x172c:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10160                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1738:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10176                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1744:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1746:0x5 DW_TAG_template_type_parameter
	.long	7751                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x174d:0x2f DW_TAG_subprogram
	.byte	121                             # DW_AT_low_pc
	.long	.Lfunc_end120-.Lfunc_begin120   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	336                             # DW_AT_linkage_name
	.short	337                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x175b:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10193                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1767:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10209                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1773:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1775:0x5 DW_TAG_template_type_parameter
	.long	7761                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x177c:0x2f DW_TAG_subprogram
	.byte	122                             # DW_AT_low_pc
	.long	.Lfunc_end121-.Lfunc_begin121   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	338                             # DW_AT_linkage_name
	.short	339                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x178a:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10226                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1796:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10242                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x17a2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x17a4:0x5 DW_TAG_template_type_parameter
	.long	7778                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x17ab:0x2f DW_TAG_subprogram
	.byte	123                             # DW_AT_low_pc
	.long	.Lfunc_end122-.Lfunc_begin122   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	340                             # DW_AT_linkage_name
	.short	341                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x17b9:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10259                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x17c5:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10275                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x17d1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x17d3:0x5 DW_TAG_template_type_parameter
	.long	7783                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x17da:0x2f DW_TAG_subprogram
	.byte	124                             # DW_AT_low_pc
	.long	.Lfunc_end123-.Lfunc_begin123   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	342                             # DW_AT_linkage_name
	.short	343                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x17e8:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10292                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x17f4:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10308                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1800:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1802:0x5 DW_TAG_template_type_parameter
	.long	7814                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1809:0x2f DW_TAG_subprogram
	.byte	125                             # DW_AT_low_pc
	.long	.Lfunc_end124-.Lfunc_begin124   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	344                             # DW_AT_linkage_name
	.short	345                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1817:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10325                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1823:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10341                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x182f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1831:0x5 DW_TAG_template_type_parameter
	.long	7837                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1838:0x2f DW_TAG_subprogram
	.byte	126                             # DW_AT_low_pc
	.long	.Lfunc_end125-.Lfunc_begin125   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	346                             # DW_AT_linkage_name
	.short	347                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1846:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10358                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1852:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10374                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x185e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1860:0x5 DW_TAG_template_type_parameter
	.long	7504                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	40                              # Abbrev [40] 0x1867:0x2f DW_TAG_subprogram
	.byte	127                             # DW_AT_low_pc
	.long	.Lfunc_end126-.Lfunc_begin126   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	348                             # DW_AT_linkage_name
	.short	349                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x1875:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10391                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1881:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10407                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x188d:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x188f:0x5 DW_TAG_template_type_parameter
	.long	7849                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	40                              # Abbrev [40] 0x1896:0x30 DW_TAG_subprogram
	.ascii	"\200\001"                      # DW_AT_low_pc
	.long	.Lfunc_end127-.Lfunc_begin127   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	350                             # DW_AT_linkage_name
	.short	351                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x18a5:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10424                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x18b1:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10440                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x18bd:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x18bf:0x5 DW_TAG_template_type_parameter
	.long	7856                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	40                              # Abbrev [40] 0x18c6:0x30 DW_TAG_subprogram
	.ascii	"\201\001"                      # DW_AT_low_pc
	.long	.Lfunc_end128-.Lfunc_begin128   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	352                             # DW_AT_linkage_name
	.short	353                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x18d5:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10457                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x18e1:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10473                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x18ed:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x18ef:0x5 DW_TAG_template_type_parameter
	.long	7868                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x18f6:0x16 DW_TAG_subprogram
	.ascii	"\202\001"                      # DW_AT_low_pc
	.long	.Lfunc_end129-.Lfunc_begin129   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	354                             # DW_AT_linkage_name
	.short	355                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	173                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x1905:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x190c:0x1e DW_TAG_subprogram
	.ascii	"\203\001"                      # DW_AT_low_pc
	.long	.Lfunc_end130-.Lfunc_begin130   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	358                             # DW_AT_linkage_name
	.short	359                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	188                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x191b:0x6 DW_TAG_template_type_parameter
	.long	7875                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	57                              # Abbrev [57] 0x1921:0x8 DW_TAG_template_value_parameter
	.long	7875                            # DW_AT_type
	.short	357                             # DW_AT_name
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x192a:0x1e DW_TAG_subprogram
	.ascii	"\204\001"                      # DW_AT_low_pc
	.long	.Lfunc_end131-.Lfunc_begin131   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	361                             # DW_AT_linkage_name
	.short	362                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	188                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x1939:0x6 DW_TAG_template_type_parameter
	.long	7881                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	58                              # Abbrev [58] 0x193f:0x8 DW_TAG_template_value_parameter
	.long	7886                            # DW_AT_type
	.short	357                             # DW_AT_name
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1948:0x26 DW_TAG_subprogram
	.ascii	"\205\001"                      # DW_AT_low_pc
	.long	.Lfunc_end132-.Lfunc_begin132   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	364                             # DW_AT_linkage_name
	.short	365                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	188                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x1957:0x6 DW_TAG_template_type_parameter
	.long	7892                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	59                              # Abbrev [59] 0x195d:0x10 DW_TAG_template_value_parameter
	.long	7892                            # DW_AT_type
	.short	357                             # DW_AT_name
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
	.byte	12                              # Abbrev [12] 0x196e:0x26 DW_TAG_subprogram
	.ascii	"\206\001"                      # DW_AT_low_pc
	.long	.Lfunc_end133-.Lfunc_begin133   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	367                             # DW_AT_linkage_name
	.short	368                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	188                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0x197d:0x6 DW_TAG_template_type_parameter
	.long	7898                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	59                              # Abbrev [59] 0x1983:0x10 DW_TAG_template_value_parameter
	.long	7903                            # DW_AT_type
	.short	357                             # DW_AT_name
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
	.byte	12                              # Abbrev [12] 0x1994:0x30 DW_TAG_subprogram
	.ascii	"\207\001"                      # DW_AT_low_pc
	.long	.Lfunc_end134-.Lfunc_begin134   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	370                             # DW_AT_linkage_name
	.short	371                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x19a3:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10490                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x19af:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10506                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x19bb:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x19bd:0x5 DW_TAG_template_type_parameter
	.long	7909                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x19c4:0x30 DW_TAG_subprogram
	.ascii	"\210\001"                      # DW_AT_low_pc
	.long	.Lfunc_end135-.Lfunc_begin135   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	372                             # DW_AT_linkage_name
	.short	373                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x19d3:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10523                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x19df:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10539                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x19eb:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x19ed:0x5 DW_TAG_template_type_parameter
	.long	7931                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x19f4:0x30 DW_TAG_subprogram
	.ascii	"\211\001"                      # DW_AT_low_pc
	.long	.Lfunc_end136-.Lfunc_begin136   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	374                             # DW_AT_linkage_name
	.short	375                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1a03:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10556                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1a0f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10572                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1a1b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1a1d:0x5 DW_TAG_template_type_parameter
	.long	7940                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1a24:0x30 DW_TAG_subprogram
	.ascii	"\212\001"                      # DW_AT_low_pc
	.long	.Lfunc_end137-.Lfunc_begin137   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	376                             # DW_AT_linkage_name
	.short	377                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1a33:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10589                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1a3f:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10605                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1a4b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1a4d:0x5 DW_TAG_template_type_parameter
	.long	7942                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	40                              # Abbrev [40] 0x1a54:0x16 DW_TAG_subprogram
	.ascii	"\213\001"                      # DW_AT_low_pc
	.long	.Lfunc_end138-.Lfunc_begin138   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	378                             # DW_AT_linkage_name
	.short	379                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	185                             # DW_AT_decl_line
	.byte	35                              # Abbrev [35] 0x1a63:0x6 DW_TAG_template_value_parameter
	.long	133                             # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	14                              # Abbrev [14] 0x1a6a:0x12 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	85                              # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	155                             # DW_AT_decl_line
	.byte	17                              # Abbrev [17] 0x1a70:0xb DW_TAG_subprogram
	.byte	83                              # DW_AT_linkage_name
	.byte	84                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	156                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	19                              # Abbrev [19] 0x1a75:0x5 DW_TAG_formal_parameter
	.long	6780                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1a7c:0x5 DW_TAG_pointer_type
	.long	6762                            # DW_AT_type
	.byte	60                              # Abbrev [60] 0x1a81:0x21 DW_TAG_subprogram
	.ascii	"\214\001"                      # DW_AT_low_pc
	.long	.Lfunc_end139-.Lfunc_begin139   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6804                            # DW_AT_object_pointer
	.short	342                             # DW_AT_decl_line
	.long	6768                            # DW_AT_specification
	.byte	49                              # Abbrev [49] 0x1a94:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.short	486                             # DW_AT_name
	.long	10622                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	29                              # Abbrev [29] 0x1a9e:0x3 DW_TAG_structure_type
	.short	281                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	40                              # Abbrev [40] 0x1aa2:0x30 DW_TAG_subprogram
	.ascii	"\215\001"                      # DW_AT_low_pc
	.long	.Lfunc_end140-.Lfunc_begin140   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	380                             # DW_AT_linkage_name
	.short	283                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x1ab1:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10627                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1abd:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10643                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1ac9:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1acb:0x5 DW_TAG_template_type_parameter
	.long	6814                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1ad2:0x30 DW_TAG_subprogram
	.ascii	"\216\001"                      # DW_AT_low_pc
	.long	.Lfunc_end141-.Lfunc_begin141   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	381                             # DW_AT_linkage_name
	.short	382                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1ae1:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	36                              # DW_AT_decl_line
	.long	10660                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1aed:0xc DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.long	10676                           # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1af9:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1afb:0x5 DW_TAG_template_type_parameter
	.long	7947                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	8                               # Abbrev [8] 0x1b02:0x75 DW_TAG_namespace
	.byte	86                              # DW_AT_name
	.byte	61                              # Abbrev [61] 0x1b04:0x2b DW_TAG_subprogram
	.ascii	"\217\001"                      # DW_AT_low_pc
	.long	.Lfunc_end142-.Lfunc_begin142   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.short	383                             # DW_AT_linkage_name
	.short	384                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.short	360                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	25                              # Abbrev [25] 0x1b14:0xd DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.short	389                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.short	361                             # DW_AT_decl_line
	.long	6959                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1b21:0xd DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	126
	.short	386                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.short	362                             # DW_AT_decl_line
	.long	7002                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	62                              # Abbrev [62] 0x1b2f:0x13 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	240                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.short	357                             # DW_AT_decl_line
	.byte	63                              # Abbrev [63] 0x1b36:0xb DW_TAG_member
	.short	389                             # DW_AT_name
	.long	6978                            # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.short	358                             # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	64                              # Abbrev [64] 0x1b42:0x13 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	556                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.short	355                             # DW_AT_decl_line
	.byte	65                              # Abbrev [65] 0x1b4a:0x5 DW_TAG_template_type_parameter
	.long	6998                            # DW_AT_type
                                        # DW_AT_default_value
	.byte	65                              # Abbrev [65] 0x1b4f:0x5 DW_TAG_template_type_parameter
	.long	7002                            # DW_AT_type
                                        # DW_AT_default_value
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x1b55:0x5 DW_TAG_namespace
	.byte	29                              # Abbrev [29] 0x1b56:0x3 DW_TAG_structure_type
	.short	553                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	66                              # Abbrev [66] 0x1b5a:0xe DW_TAG_class_type
	.byte	5                               # DW_AT_calling_convention
	.short	555                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.short	353                             # DW_AT_decl_line
	.byte	65                              # Abbrev [65] 0x1b62:0x5 DW_TAG_template_type_parameter
	.long	7016                            # DW_AT_type
                                        # DW_AT_default_value
	.byte	0                               # End Of Children Mark
	.byte	67                              # Abbrev [67] 0x1b68:0xe DW_TAG_structure_type
	.short	554                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	68                              # Abbrev [68] 0x1b6b:0xa DW_TAG_template_value_parameter
	.long	7504                            # DW_AT_type
                                        # DW_AT_default_value
	.byte	4                               # DW_AT_location
	.byte	161
	.ascii	"\220\001"
	.byte	159
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	69                              # Abbrev [69] 0x1b77:0x9 DW_TAG_typedef
	.long	7040                            # DW_AT_type
	.byte	88                              # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	268                             # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x1b80:0x4 DW_TAG_base_type
	.byte	87                              # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	52                              # Abbrev [52] 0x1b84:0x8 DW_TAG_typedef
	.long	7052                            # DW_AT_type
	.byte	90                              # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	245                             # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x1b8c:0x4 DW_TAG_base_type
	.byte	89                              # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	2                               # DW_AT_byte_size
	.byte	52                              # Abbrev [52] 0x1b90:0x8 DW_TAG_typedef
	.long	54                              # DW_AT_type
	.byte	91                              # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	197                             # DW_AT_decl_line
	.byte	52                              # Abbrev [52] 0x1b98:0x8 DW_TAG_typedef
	.long	7072                            # DW_AT_type
	.byte	93                              # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	100                             # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x1ba0:0x4 DW_TAG_base_type
	.byte	92                              # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	69                              # Abbrev [69] 0x1ba4:0x9 DW_TAG_typedef
	.long	178                             # DW_AT_type
	.byte	94                              # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	270                             # DW_AT_decl_line
	.byte	52                              # Abbrev [52] 0x1bad:0x8 DW_TAG_typedef
	.long	7093                            # DW_AT_type
	.byte	96                              # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	247                             # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x1bb5:0x4 DW_TAG_base_type
	.byte	95                              # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	2                               # DW_AT_byte_size
	.byte	52                              # Abbrev [52] 0x1bb9:0x8 DW_TAG_typedef
	.long	75                              # DW_AT_type
	.byte	97                              # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	202                             # DW_AT_decl_line
	.byte	52                              # Abbrev [52] 0x1bc1:0x8 DW_TAG_typedef
	.long	4793                            # DW_AT_type
	.byte	98                              # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	102                             # DW_AT_decl_line
	.byte	69                              # Abbrev [69] 0x1bc9:0x9 DW_TAG_typedef
	.long	7031                            # DW_AT_type
	.byte	99                              # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	278                             # DW_AT_decl_line
	.byte	69                              # Abbrev [69] 0x1bd2:0x9 DW_TAG_typedef
	.long	7044                            # DW_AT_type
	.byte	100                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	259                             # DW_AT_decl_line
	.byte	52                              # Abbrev [52] 0x1bdb:0x8 DW_TAG_typedef
	.long	7056                            # DW_AT_type
	.byte	101                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	220                             # DW_AT_decl_line
	.byte	52                              # Abbrev [52] 0x1be3:0x8 DW_TAG_typedef
	.long	7064                            # DW_AT_type
	.byte	102                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	122                             # DW_AT_decl_line
	.byte	69                              # Abbrev [69] 0x1beb:0x9 DW_TAG_typedef
	.long	7076                            # DW_AT_type
	.byte	103                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	279                             # DW_AT_decl_line
	.byte	69                              # Abbrev [69] 0x1bf4:0x9 DW_TAG_typedef
	.long	7085                            # DW_AT_type
	.byte	104                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	260                             # DW_AT_decl_line
	.byte	52                              # Abbrev [52] 0x1bfd:0x8 DW_TAG_typedef
	.long	7097                            # DW_AT_type
	.byte	105                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	221                             # DW_AT_decl_line
	.byte	52                              # Abbrev [52] 0x1c05:0x8 DW_TAG_typedef
	.long	7105                            # DW_AT_type
	.byte	106                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	123                             # DW_AT_decl_line
	.byte	69                              # Abbrev [69] 0x1c0d:0x9 DW_TAG_typedef
	.long	7031                            # DW_AT_type
	.byte	107                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	280                             # DW_AT_decl_line
	.byte	69                              # Abbrev [69] 0x1c16:0x9 DW_TAG_typedef
	.long	7044                            # DW_AT_type
	.byte	108                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	261                             # DW_AT_decl_line
	.byte	52                              # Abbrev [52] 0x1c1f:0x8 DW_TAG_typedef
	.long	7056                            # DW_AT_type
	.byte	109                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	222                             # DW_AT_decl_line
	.byte	52                              # Abbrev [52] 0x1c27:0x8 DW_TAG_typedef
	.long	7064                            # DW_AT_type
	.byte	110                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	124                             # DW_AT_decl_line
	.byte	69                              # Abbrev [69] 0x1c2f:0x9 DW_TAG_typedef
	.long	7076                            # DW_AT_type
	.byte	111                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	281                             # DW_AT_decl_line
	.byte	69                              # Abbrev [69] 0x1c38:0x9 DW_TAG_typedef
	.long	7085                            # DW_AT_type
	.byte	112                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	262                             # DW_AT_decl_line
	.byte	52                              # Abbrev [52] 0x1c41:0x8 DW_TAG_typedef
	.long	7097                            # DW_AT_type
	.byte	113                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	223                             # DW_AT_decl_line
	.byte	52                              # Abbrev [52] 0x1c49:0x8 DW_TAG_typedef
	.long	7105                            # DW_AT_type
	.byte	114                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	125                             # DW_AT_decl_line
	.byte	69                              # Abbrev [69] 0x1c51:0x9 DW_TAG_typedef
	.long	7072                            # DW_AT_type
	.byte	115                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	295                             # DW_AT_decl_line
	.byte	69                              # Abbrev [69] 0x1c5a:0x9 DW_TAG_typedef
	.long	4793                            # DW_AT_type
	.byte	116                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	302                             # DW_AT_decl_line
	.byte	69                              # Abbrev [69] 0x1c63:0x9 DW_TAG_typedef
	.long	7072                            # DW_AT_type
	.byte	117                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	308                             # DW_AT_decl_line
	.byte	69                              # Abbrev [69] 0x1c6c:0x9 DW_TAG_typedef
	.long	4793                            # DW_AT_type
	.byte	118                             # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	309                             # DW_AT_decl_line
	.byte	52                              # Abbrev [52] 0x1c75:0x8 DW_TAG_typedef
	.long	7293                            # DW_AT_type
	.byte	119                             # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	24                              # DW_AT_decl_line
	.byte	70                              # Abbrev [70] 0x1c7d:0x1 DW_TAG_structure_type
                                        # DW_AT_declaration
	.byte	3                               # Abbrev [3] 0x1c7e:0x4 DW_TAG_base_type
	.byte	129                             # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x1c82:0x4 DW_TAG_base_type
	.byte	138                             # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x1c86:0x4 DW_TAG_base_type
	.byte	141                             # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	7                               # Abbrev [7] 0x1c8a:0x2 DW_TAG_structure_type
	.byte	144                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	47                              # Abbrev [47] 0x1c8c:0x5 DW_TAG_pointer_type
	.long	170                             # DW_AT_type
	.byte	14                              # Abbrev [14] 0x1c91:0xf DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	154                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x1c97:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1c99:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1ca0:0x5 DW_TAG_pointer_type
	.long	54                              # DW_AT_type
	.byte	71                              # Abbrev [71] 0x1ca5:0x5 DW_TAG_reference_type
	.long	54                              # DW_AT_type
	.byte	72                              # Abbrev [72] 0x1caa:0x5 DW_TAG_rvalue_reference_type
	.long	54                              # DW_AT_type
	.byte	73                              # Abbrev [73] 0x1caf:0x5 DW_TAG_const_type
	.long	54                              # DW_AT_type
	.byte	74                              # Abbrev [74] 0x1cb4:0xc DW_TAG_array_type
	.long	54                              # DW_AT_type
	.byte	75                              # Abbrev [75] 0x1cb9:0x6 DW_TAG_subrange_type
	.long	7360                            # DW_AT_type
	.byte	3                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	76                              # Abbrev [76] 0x1cc0:0x4 DW_TAG_base_type
	.byte	167                             # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	7                               # DW_AT_encoding
	.byte	14                              # Abbrev [14] 0x1cc4:0x9 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	172                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	52                              # DW_AT_decl_line
	.byte	7                               # Abbrev [7] 0x1cca:0x2 DW_TAG_structure_type
	.byte	173                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0x1ccd:0x4 DW_TAG_base_type
	.byte	211                             # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x1cd1:0x4 DW_TAG_base_type
	.byte	214                             # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	16                              # DW_AT_byte_size
	.byte	77                              # Abbrev [77] 0x1cd5:0x10 DW_TAG_structure_type
	.byte	221                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	15                              # Abbrev [15] 0x1cd7:0x6 DW_TAG_template_type_parameter
	.long	182                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	16                              # Abbrev [16] 0x1cdd:0x7 DW_TAG_template_value_parameter
	.long	202                             # DW_AT_type
	.byte	22                              # DW_AT_name
                                        # DW_AT_default_value
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	77                              # Abbrev [77] 0x1ce5:0x10 DW_TAG_structure_type
	.byte	227                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	15                              # Abbrev [15] 0x1ce7:0x6 DW_TAG_template_type_parameter
	.long	7413                            # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	16                              # Abbrev [16] 0x1ced:0x7 DW_TAG_template_value_parameter
	.long	202                             # DW_AT_type
	.byte	22                              # DW_AT_name
                                        # DW_AT_default_value
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	14                              # Abbrev [14] 0x1cf5:0x14 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	226                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	49                              # DW_AT_decl_line
	.byte	15                              # Abbrev [15] 0x1cfb:0x6 DW_TAG_template_type_parameter
	.long	371                             # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	16                              # Abbrev [16] 0x1d01:0x7 DW_TAG_template_value_parameter
	.long	202                             # DW_AT_type
	.byte	22                              # DW_AT_name
                                        # DW_AT_default_value
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	78                              # Abbrev [78] 0x1d09:0xb DW_TAG_subroutine_type
	.long	54                              # DW_AT_type
	.byte	45                              # Abbrev [45] 0x1d0e:0x5 DW_TAG_formal_parameter
	.long	3923                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	79                              # Abbrev [79] 0x1d14:0x3 DW_TAG_subroutine_type
	.byte	80                              # Abbrev [80] 0x1d15:0x1 DW_TAG_unspecified_parameters
	.byte	0                               # End Of Children Mark
	.byte	79                              # Abbrev [79] 0x1d17:0x8 DW_TAG_subroutine_type
	.byte	45                              # Abbrev [45] 0x1d18:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	80                              # Abbrev [80] 0x1d1d:0x1 DW_TAG_unspecified_parameters
	.byte	0                               # End Of Children Mark
	.byte	71                              # Abbrev [71] 0x1d1f:0x5 DW_TAG_reference_type
	.long	7343                            # DW_AT_type
	.byte	71                              # Abbrev [71] 0x1d24:0x5 DW_TAG_reference_type
	.long	7465                            # DW_AT_type
	.byte	47                              # Abbrev [47] 0x1d29:0x5 DW_TAG_pointer_type
	.long	7343                            # DW_AT_type
	.byte	81                              # Abbrev [81] 0x1d2e:0x2 DW_TAG_unspecified_type
	.byte	243                             # DW_AT_name
	.byte	47                              # Abbrev [47] 0x1d30:0x5 DW_TAG_pointer_type
	.long	7072                            # DW_AT_type
	.byte	47                              # Abbrev [47] 0x1d35:0x5 DW_TAG_pointer_type
	.long	7306                            # DW_AT_type
	.byte	73                              # Abbrev [73] 0x1d3a:0x5 DW_TAG_const_type
	.long	4575                            # DW_AT_type
	.byte	47                              # Abbrev [47] 0x1d3f:0x5 DW_TAG_pointer_type
	.long	7492                            # DW_AT_type
	.byte	73                              # Abbrev [73] 0x1d44:0x5 DW_TAG_const_type
	.long	7497                            # DW_AT_type
	.byte	47                              # Abbrev [47] 0x1d49:0x5 DW_TAG_pointer_type
	.long	7502                            # DW_AT_type
	.byte	82                              # Abbrev [82] 0x1d4e:0x1 DW_TAG_const_type
	.byte	83                              # Abbrev [83] 0x1d4f:0x1 DW_TAG_subroutine_type
	.byte	47                              # Abbrev [47] 0x1d50:0x5 DW_TAG_pointer_type
	.long	7503                            # DW_AT_type
	.byte	47                              # Abbrev [47] 0x1d55:0x5 DW_TAG_pointer_type
	.long	371                             # DW_AT_type
	.byte	47                              # Abbrev [47] 0x1d5a:0x5 DW_TAG_pointer_type
	.long	376                             # DW_AT_type
	.byte	47                              # Abbrev [47] 0x1d5f:0x5 DW_TAG_pointer_type
	.long	7524                            # DW_AT_type
	.byte	84                              # Abbrev [84] 0x1d64:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	276                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x1d6b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1d6d:0x5 DW_TAG_template_type_parameter
	.long	7328                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	74                              # Abbrev [74] 0x1d74:0xb DW_TAG_array_type
	.long	7328                            # DW_AT_type
	.byte	85                              # Abbrev [85] 0x1d79:0x5 DW_TAG_subrange_type
	.long	7360                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	71                              # Abbrev [71] 0x1d7f:0x5 DW_TAG_reference_type
	.long	7348                            # DW_AT_type
	.byte	47                              # Abbrev [47] 0x1d84:0x5 DW_TAG_pointer_type
	.long	7348                            # DW_AT_type
	.byte	47                              # Abbrev [47] 0x1d89:0x5 DW_TAG_pointer_type
	.long	7470                            # DW_AT_type
	.byte	84                              # Abbrev [84] 0x1d8e:0xe DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	298                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	153                             # DW_AT_decl_line
	.byte	15                              # Abbrev [15] 0x1d95:0x6 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	86                              # Abbrev [86] 0x1d9c:0x5 DW_TAG_atomic_type
	.long	54                              # DW_AT_type
	.byte	87                              # Abbrev [87] 0x1da1:0x5 DW_TAG_volatile_type
	.long	7373                            # DW_AT_type
	.byte	88                              # Abbrev [88] 0x1da6:0xc DW_TAG_array_type
                                        # DW_AT_GNU_vector
	.long	54                              # DW_AT_type
	.byte	75                              # Abbrev [75] 0x1dab:0x6 DW_TAG_subrange_type
	.long	7360                            # DW_AT_type
	.byte	2                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	73                              # Abbrev [73] 0x1db2:0x5 DW_TAG_const_type
	.long	7607                            # DW_AT_type
	.byte	87                              # Abbrev [87] 0x1db7:0x5 DW_TAG_volatile_type
	.long	7328                            # DW_AT_type
	.byte	73                              # Abbrev [73] 0x1dbc:0x5 DW_TAG_const_type
	.long	7617                            # DW_AT_type
	.byte	89                              # Abbrev [89] 0x1dc1:0x1 DW_TAG_volatile_type
	.byte	84                              # Abbrev [84] 0x1dc2:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	314                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x1dc9:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1dcb:0x5 DW_TAG_template_type_parameter
	.long	371                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	90                              # Abbrev [90] 0x1dd2:0x9 DW_TAG_ptr_to_member_type
	.long	7643                            # DW_AT_type
	.long	7306                            # DW_AT_containing_type
	.byte	79                              # Abbrev [79] 0x1ddb:0x7 DW_TAG_subroutine_type
	.byte	19                              # Abbrev [19] 0x1ddc:0x5 DW_TAG_formal_parameter
	.long	7650                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1de2:0x5 DW_TAG_pointer_type
	.long	7655                            # DW_AT_type
	.byte	73                              # Abbrev [73] 0x1de7:0x5 DW_TAG_const_type
	.long	7306                            # DW_AT_type
	.byte	90                              # Abbrev [90] 0x1dec:0x9 DW_TAG_ptr_to_member_type
	.long	7669                            # DW_AT_type
	.long	7306                            # DW_AT_containing_type
	.byte	91                              # Abbrev [91] 0x1df5:0x7 DW_TAG_subroutine_type
                                        # DW_AT_reference
	.byte	19                              # Abbrev [19] 0x1df6:0x5 DW_TAG_formal_parameter
	.long	7676                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1dfc:0x5 DW_TAG_pointer_type
	.long	7681                            # DW_AT_type
	.byte	87                              # Abbrev [87] 0x1e01:0x5 DW_TAG_volatile_type
	.long	7306                            # DW_AT_type
	.byte	90                              # Abbrev [90] 0x1e06:0x9 DW_TAG_ptr_to_member_type
	.long	7695                            # DW_AT_type
	.long	7306                            # DW_AT_containing_type
	.byte	92                              # Abbrev [92] 0x1e0f:0x7 DW_TAG_subroutine_type
                                        # DW_AT_rvalue_reference
	.byte	19                              # Abbrev [19] 0x1e10:0x5 DW_TAG_formal_parameter
	.long	7702                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1e16:0x5 DW_TAG_pointer_type
	.long	7707                            # DW_AT_type
	.byte	73                              # Abbrev [73] 0x1e1b:0x5 DW_TAG_const_type
	.long	7681                            # DW_AT_type
	.byte	73                              # Abbrev [73] 0x1e20:0x5 DW_TAG_const_type
	.long	7504                            # DW_AT_type
	.byte	71                              # Abbrev [71] 0x1e25:0x5 DW_TAG_reference_type
	.long	7722                            # DW_AT_type
	.byte	73                              # Abbrev [73] 0x1e2a:0x5 DW_TAG_const_type
	.long	7727                            # DW_AT_type
	.byte	74                              # Abbrev [74] 0x1e2f:0xc DW_TAG_array_type
	.long	7373                            # DW_AT_type
	.byte	75                              # Abbrev [75] 0x1e34:0x6 DW_TAG_subrange_type
	.long	7360                            # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	73                              # Abbrev [73] 0x1e3b:0x5 DW_TAG_const_type
	.long	7744                            # DW_AT_type
	.byte	93                              # Abbrev [93] 0x1e40:0x1 DW_TAG_subroutine_type
                                        # DW_AT_reference
	.byte	87                              # Abbrev [87] 0x1e41:0x5 DW_TAG_volatile_type
	.long	7750                            # DW_AT_type
	.byte	94                              # Abbrev [94] 0x1e46:0x1 DW_TAG_subroutine_type
                                        # DW_AT_rvalue_reference
	.byte	73                              # Abbrev [73] 0x1e47:0x5 DW_TAG_const_type
	.long	7756                            # DW_AT_type
	.byte	87                              # Abbrev [87] 0x1e4c:0x5 DW_TAG_volatile_type
	.long	7503                            # DW_AT_type
	.byte	73                              # Abbrev [73] 0x1e51:0x5 DW_TAG_const_type
	.long	7766                            # DW_AT_type
	.byte	74                              # Abbrev [74] 0x1e56:0xc DW_TAG_array_type
	.long	7328                            # DW_AT_type
	.byte	75                              # Abbrev [75] 0x1e5b:0x6 DW_TAG_subrange_type
	.long	7360                            # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	71                              # Abbrev [71] 0x1e62:0x5 DW_TAG_reference_type
	.long	7761                            # DW_AT_type
	.byte	71                              # Abbrev [71] 0x1e67:0x5 DW_TAG_reference_type
	.long	7788                            # DW_AT_type
	.byte	73                              # Abbrev [73] 0x1e6c:0x5 DW_TAG_const_type
	.long	7793                            # DW_AT_type
	.byte	90                              # Abbrev [90] 0x1e71:0x9 DW_TAG_ptr_to_member_type
	.long	7802                            # DW_AT_type
	.long	7306                            # DW_AT_containing_type
	.byte	79                              # Abbrev [79] 0x1e7a:0x7 DW_TAG_subroutine_type
	.byte	19                              # Abbrev [19] 0x1e7b:0x5 DW_TAG_formal_parameter
	.long	7809                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1e81:0x5 DW_TAG_pointer_type
	.long	7306                            # DW_AT_type
	.byte	78                              # Abbrev [78] 0x1e86:0xb DW_TAG_subroutine_type
	.long	7825                            # DW_AT_type
	.byte	45                              # Abbrev [45] 0x1e8b:0x5 DW_TAG_formal_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1e91:0x5 DW_TAG_pointer_type
	.long	7830                            # DW_AT_type
	.byte	79                              # Abbrev [79] 0x1e96:0x7 DW_TAG_subroutine_type
	.byte	45                              # Abbrev [45] 0x1e97:0x5 DW_TAG_formal_parameter
	.long	3923                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	74                              # Abbrev [74] 0x1e9d:0xc DW_TAG_array_type
	.long	7313                            # DW_AT_type
	.byte	75                              # Abbrev [75] 0x1ea2:0x6 DW_TAG_subrange_type
	.long	7360                            # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	79                              # Abbrev [79] 0x1ea9:0x7 DW_TAG_subroutine_type
	.byte	45                              # Abbrev [45] 0x1eaa:0x5 DW_TAG_formal_parameter
	.long	376                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	79                              # Abbrev [79] 0x1eb0:0xc DW_TAG_subroutine_type
	.byte	45                              # Abbrev [45] 0x1eb1:0x5 DW_TAG_formal_parameter
	.long	384                             # DW_AT_type
	.byte	45                              # Abbrev [45] 0x1eb6:0x5 DW_TAG_formal_parameter
	.long	376                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	79                              # Abbrev [79] 0x1ebc:0x7 DW_TAG_subroutine_type
	.byte	45                              # Abbrev [45] 0x1ebd:0x5 DW_TAG_formal_parameter
	.long	384                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	95                              # Abbrev [95] 0x1ec3:0x6 DW_TAG_base_type
	.short	356                             # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	3                               # DW_AT_bit_size
	.byte	73                              # Abbrev [73] 0x1ec9:0x5 DW_TAG_const_type
	.long	7886                            # DW_AT_type
	.byte	95                              # Abbrev [95] 0x1ece:0x6 DW_TAG_base_type
	.short	360                             # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	5                               # DW_AT_bit_size
	.byte	95                              # Abbrev [95] 0x1ed4:0x6 DW_TAG_base_type
	.short	363                             # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	16                              # DW_AT_byte_size
	.byte	65                              # DW_AT_bit_size
	.byte	73                              # Abbrev [73] 0x1eda:0x5 DW_TAG_const_type
	.long	7903                            # DW_AT_type
	.byte	95                              # Abbrev [95] 0x1edf:0x6 DW_TAG_base_type
	.short	366                             # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	16                              # DW_AT_byte_size
	.byte	65                              # DW_AT_bit_size
	.byte	79                              # Abbrev [79] 0x1ee5:0xc DW_TAG_subroutine_type
	.byte	45                              # Abbrev [45] 0x1ee6:0x5 DW_TAG_formal_parameter
	.long	7921                            # DW_AT_type
	.byte	45                              # Abbrev [45] 0x1eeb:0x5 DW_TAG_formal_parameter
	.long	7921                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x1ef1:0xa DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	369                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	44                              # Abbrev [44] 0x1ef8:0x2 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	90                              # Abbrev [90] 0x1efb:0x9 DW_TAG_ptr_to_member_type
	.long	54                              # DW_AT_type
	.long	7921                            # DW_AT_containing_type
	.byte	96                              # Abbrev [96] 0x1f04:0x2 DW_TAG_subroutine_type
	.byte	200                             # DW_AT_calling_convention
	.byte	97                              # Abbrev [97] 0x1f06:0x5 DW_TAG_subroutine_type
	.long	54                              # DW_AT_type
	.byte	90                              # Abbrev [90] 0x1f0b:0x9 DW_TAG_ptr_to_member_type
	.long	7956                            # DW_AT_type
	.long	6762                            # DW_AT_containing_type
	.byte	79                              # Abbrev [79] 0x1f14:0x7 DW_TAG_subroutine_type
	.byte	19                              # Abbrev [19] 0x1f15:0x5 DW_TAG_formal_parameter
	.long	6780                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x1f1b:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	388                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.byte	58                              # Abbrev [58] 0x1f22:0x8 DW_TAG_template_value_parameter
	.long	75                              # DW_AT_type
	.short	387                             # DW_AT_name
	.byte	3                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	98                              # Abbrev [98] 0x1f2b:0x9 DW_TAG_typedef
	.long	7566                            # DW_AT_type
	.short	392                             # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	149                             # DW_AT_decl_line
	.byte	84                              # Abbrev [84] 0x1f34:0x12 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	396                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	180                             # DW_AT_decl_line
	.byte	99                              # Abbrev [99] 0x1f3b:0xa DW_TAG_member
	.short	389                             # DW_AT_name
	.long	8006                            # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	181                             # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x1f46:0x17 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	395                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	177                             # DW_AT_decl_line
	.byte	15                              # Abbrev [15] 0x1f4d:0x6 DW_TAG_template_type_parameter
	.long	59                              # DW_AT_type
	.byte	20                              # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1f53:0x9 DW_TAG_GNU_template_parameter_pack
	.byte	180                             # DW_AT_name
	.byte	36                              # Abbrev [36] 0x1f55:0x6 DW_TAG_template_value_parameter
	.long	59                              # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1f5d:0x5 DW_TAG_pointer_type
	.long	8034                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x1f62:0xc DW_TAG_structure_type
	.short	397                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x1f65:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1f67:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1f6e:0x5 DW_TAG_pointer_type
	.long	8051                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x1f73:0xc DW_TAG_structure_type
	.short	398                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x1f76:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1f78:0x5 DW_TAG_template_type_parameter
	.long	3923                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x1f7f:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	399                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x1f86:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1f88:0x5 DW_TAG_template_type_parameter
	.long	202                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1f8f:0x5 DW_TAG_pointer_type
	.long	8084                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x1f94:0xc DW_TAG_structure_type
	.short	400                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x1f97:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1f99:0x5 DW_TAG_template_type_parameter
	.long	202                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x1fa0:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	401                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x1fa7:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1fa9:0x5 DW_TAG_template_type_parameter
	.long	7294                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1fb0:0x5 DW_TAG_pointer_type
	.long	8117                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x1fb5:0xc DW_TAG_structure_type
	.short	402                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x1fb8:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1fba:0x5 DW_TAG_template_type_parameter
	.long	7294                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x1fc1:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	403                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x1fc8:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1fca:0x5 DW_TAG_template_type_parameter
	.long	7072                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1fd1:0x5 DW_TAG_pointer_type
	.long	8150                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x1fd6:0xc DW_TAG_structure_type
	.short	404                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x1fd9:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1fdb:0x5 DW_TAG_template_type_parameter
	.long	7072                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x1fe2:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	405                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x1fe9:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1feb:0x5 DW_TAG_template_type_parameter
	.long	7052                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x1ff2:0x5 DW_TAG_pointer_type
	.long	8183                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x1ff7:0xc DW_TAG_structure_type
	.short	406                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x1ffa:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x1ffc:0x5 DW_TAG_template_type_parameter
	.long	7052                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2003:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	407                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x200a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x200c:0x5 DW_TAG_template_type_parameter
	.long	75                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2013:0x5 DW_TAG_pointer_type
	.long	8216                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2018:0xc DW_TAG_structure_type
	.short	408                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x201b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x201d:0x5 DW_TAG_template_type_parameter
	.long	75                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2024:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	409                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x202b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x202d:0x5 DW_TAG_template_type_parameter
	.long	7298                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2034:0x5 DW_TAG_pointer_type
	.long	8249                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2039:0xc DW_TAG_structure_type
	.short	410                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x203c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x203e:0x5 DW_TAG_template_type_parameter
	.long	7298                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2045:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	411                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x204c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x204e:0x5 DW_TAG_template_type_parameter
	.long	7302                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2055:0x5 DW_TAG_pointer_type
	.long	8282                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x205a:0xc DW_TAG_structure_type
	.short	412                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x205d:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x205f:0x5 DW_TAG_template_type_parameter
	.long	7302                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2066:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	413                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x206d:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x206f:0x5 DW_TAG_template_type_parameter
	.long	7306                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2076:0x5 DW_TAG_pointer_type
	.long	8315                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x207b:0xc DW_TAG_structure_type
	.short	414                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x207e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2080:0x5 DW_TAG_template_type_parameter
	.long	7306                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2087:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	415                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x208e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2090:0x5 DW_TAG_template_type_parameter
	.long	170                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2097:0x5 DW_TAG_pointer_type
	.long	8348                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x209c:0xc DW_TAG_structure_type
	.short	416                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x209f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x20a1:0x5 DW_TAG_template_type_parameter
	.long	170                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x20a8:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	417                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x20af:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x20b1:0x5 DW_TAG_template_type_parameter
	.long	7308                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x20b8:0x5 DW_TAG_pointer_type
	.long	8381                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x20bd:0xc DW_TAG_structure_type
	.short	418                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x20c0:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x20c2:0x5 DW_TAG_template_type_parameter
	.long	7308                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x20c9:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	419                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x20d0:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x20d2:0x5 DW_TAG_template_type_parameter
	.long	174                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x20d9:0x5 DW_TAG_pointer_type
	.long	8414                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x20de:0xc DW_TAG_structure_type
	.short	420                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x20e1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x20e3:0x5 DW_TAG_template_type_parameter
	.long	174                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x20ea:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	421                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x20f1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x20f3:0x5 DW_TAG_template_type_parameter
	.long	7313                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x20fa:0x5 DW_TAG_pointer_type
	.long	8447                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x20ff:0xc DW_TAG_structure_type
	.short	422                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2102:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2104:0x5 DW_TAG_template_type_parameter
	.long	7313                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x210b:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	423                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2112:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2114:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	31                              # Abbrev [31] 0x2119:0x5 DW_TAG_template_type_parameter
	.long	3923                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2120:0x5 DW_TAG_pointer_type
	.long	8485                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2125:0x11 DW_TAG_structure_type
	.short	424                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2128:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x212a:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	31                              # Abbrev [31] 0x212f:0x5 DW_TAG_template_type_parameter
	.long	3923                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2136:0x5 DW_TAG_pointer_type
	.long	8507                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x213b:0xc DW_TAG_structure_type
	.short	425                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x213e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2140:0x5 DW_TAG_template_type_parameter
	.long	7328                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2147:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	426                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x214e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2150:0x5 DW_TAG_template_type_parameter
	.long	7333                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2157:0x5 DW_TAG_pointer_type
	.long	8540                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x215c:0xc DW_TAG_structure_type
	.short	427                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x215f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2161:0x5 DW_TAG_template_type_parameter
	.long	7333                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2168:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	428                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x216f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2171:0x5 DW_TAG_template_type_parameter
	.long	7338                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2178:0x5 DW_TAG_pointer_type
	.long	8573                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x217d:0xc DW_TAG_structure_type
	.short	429                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2180:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2182:0x5 DW_TAG_template_type_parameter
	.long	7338                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2189:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	430                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2190:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2192:0x5 DW_TAG_template_type_parameter
	.long	7343                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2199:0x5 DW_TAG_pointer_type
	.long	8606                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x219e:0xc DW_TAG_structure_type
	.short	431                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x21a1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x21a3:0x5 DW_TAG_template_type_parameter
	.long	7343                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x21aa:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	432                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x21b1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x21b3:0x5 DW_TAG_template_type_parameter
	.long	7348                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x21ba:0x5 DW_TAG_pointer_type
	.long	8639                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x21bf:0xc DW_TAG_structure_type
	.short	433                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x21c2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x21c4:0x5 DW_TAG_template_type_parameter
	.long	7348                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x21cb:0xc DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	434                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x21d2:0x4 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	32                              # Abbrev [32] 0x21d4:0x1 DW_TAG_template_type_parameter
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x21d7:0x5 DW_TAG_pointer_type
	.long	8668                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x21dc:0x8 DW_TAG_structure_type
	.short	435                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x21df:0x4 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	32                              # Abbrev [32] 0x21e1:0x1 DW_TAG_template_type_parameter
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x21e4:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	436                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x21eb:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x21ed:0x5 DW_TAG_template_type_parameter
	.long	7370                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x21f4:0x5 DW_TAG_pointer_type
	.long	8697                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x21f9:0xc DW_TAG_structure_type
	.short	437                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x21fc:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x21fe:0x5 DW_TAG_template_type_parameter
	.long	7370                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2205:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	438                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x220c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x220e:0x5 DW_TAG_template_type_parameter
	.long	4793                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2215:0x5 DW_TAG_pointer_type
	.long	8730                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x221a:0xc DW_TAG_structure_type
	.short	439                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x221d:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x221f:0x5 DW_TAG_template_type_parameter
	.long	4793                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2226:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	440                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x222d:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x222f:0x5 DW_TAG_template_type_parameter
	.long	182                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2236:0x5 DW_TAG_pointer_type
	.long	8763                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x223b:0xc DW_TAG_structure_type
	.short	441                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x223e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2240:0x5 DW_TAG_template_type_parameter
	.long	182                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2247:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	442                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x224e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2250:0x5 DW_TAG_template_type_parameter
	.long	7381                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2257:0x5 DW_TAG_pointer_type
	.long	8796                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x225c:0xc DW_TAG_structure_type
	.short	443                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x225f:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2261:0x5 DW_TAG_template_type_parameter
	.long	7381                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2268:0x5 DW_TAG_pointer_type
	.long	8813                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x226d:0xc DW_TAG_structure_type
	.short	444                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2270:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2272:0x5 DW_TAG_template_type_parameter
	.long	371                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2279:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	445                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2280:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2282:0x5 DW_TAG_template_type_parameter
	.long	7397                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2289:0x5 DW_TAG_pointer_type
	.long	8846                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x228e:0xc DW_TAG_structure_type
	.short	446                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2291:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2293:0x5 DW_TAG_template_type_parameter
	.long	7397                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x229a:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	447                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x22a1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x22a3:0x5 DW_TAG_template_type_parameter
	.long	7433                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x22aa:0x5 DW_TAG_pointer_type
	.long	8879                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x22af:0xc DW_TAG_structure_type
	.short	448                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x22b2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x22b4:0x5 DW_TAG_template_type_parameter
	.long	7433                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x22bb:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	449                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x22c2:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x22c4:0x5 DW_TAG_template_type_parameter
	.long	7444                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x22cb:0x5 DW_TAG_pointer_type
	.long	8912                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x22d0:0xc DW_TAG_structure_type
	.short	450                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x22d3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x22d5:0x5 DW_TAG_template_type_parameter
	.long	7444                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x22dc:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	451                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x22e3:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x22e5:0x5 DW_TAG_template_type_parameter
	.long	7447                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x22ec:0x5 DW_TAG_pointer_type
	.long	8945                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x22f1:0xc DW_TAG_structure_type
	.short	452                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x22f4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x22f6:0x5 DW_TAG_template_type_parameter
	.long	7447                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x22fd:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	453                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2304:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2306:0x5 DW_TAG_template_type_parameter
	.long	7455                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x230d:0x5 DW_TAG_pointer_type
	.long	8978                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2312:0xc DW_TAG_structure_type
	.short	454                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2315:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2317:0x5 DW_TAG_template_type_parameter
	.long	7455                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x231e:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	455                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2325:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2327:0x5 DW_TAG_template_type_parameter
	.long	7460                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x232e:0x5 DW_TAG_pointer_type
	.long	9011                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2333:0xc DW_TAG_structure_type
	.short	456                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2336:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2338:0x5 DW_TAG_template_type_parameter
	.long	7460                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x233f:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	457                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2346:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2348:0x5 DW_TAG_template_type_parameter
	.long	72                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x234f:0x5 DW_TAG_pointer_type
	.long	9044                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2354:0xc DW_TAG_structure_type
	.short	458                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2357:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2359:0x5 DW_TAG_template_type_parameter
	.long	72                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2360:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	459                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2367:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2369:0x5 DW_TAG_template_type_parameter
	.long	7470                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2370:0x5 DW_TAG_pointer_type
	.long	9077                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2375:0xc DW_TAG_structure_type
	.short	460                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2378:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x237a:0x5 DW_TAG_template_type_parameter
	.long	7470                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2381:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	461                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2388:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x238a:0x5 DW_TAG_template_type_parameter
	.long	7472                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0x238f:0x5 DW_TAG_template_type_parameter
	.long	7472                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2396:0x5 DW_TAG_pointer_type
	.long	9115                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x239b:0x11 DW_TAG_structure_type
	.short	462                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x239e:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x23a0:0x5 DW_TAG_template_type_parameter
	.long	7472                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0x23a5:0x5 DW_TAG_template_type_parameter
	.long	7472                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x23ac:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	463                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x23b3:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x23b5:0x5 DW_TAG_template_type_parameter
	.long	7472                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0x23ba:0x5 DW_TAG_template_type_parameter
	.long	7477                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x23c1:0x5 DW_TAG_pointer_type
	.long	9158                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x23c6:0x11 DW_TAG_structure_type
	.short	464                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x23c9:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x23cb:0x5 DW_TAG_template_type_parameter
	.long	7472                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0x23d0:0x5 DW_TAG_template_type_parameter
	.long	7477                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x23d7:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	465                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x23de:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x23e0:0x5 DW_TAG_template_type_parameter
	.long	7482                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x23e7:0x5 DW_TAG_pointer_type
	.long	9196                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x23ec:0xc DW_TAG_structure_type
	.short	466                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x23ef:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x23f1:0x5 DW_TAG_template_type_parameter
	.long	7482                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x23f8:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	467                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x23ff:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2401:0x5 DW_TAG_template_type_parameter
	.long	7487                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2408:0x5 DW_TAG_pointer_type
	.long	9229                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x240d:0xc DW_TAG_structure_type
	.short	468                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2410:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2412:0x5 DW_TAG_template_type_parameter
	.long	7487                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2419:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	469                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2420:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2422:0x5 DW_TAG_template_type_parameter
	.long	7503                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2429:0x5 DW_TAG_pointer_type
	.long	9262                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x242e:0xc DW_TAG_structure_type
	.short	470                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2431:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2433:0x5 DW_TAG_template_type_parameter
	.long	7503                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x243a:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	471                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2441:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2443:0x5 DW_TAG_template_type_parameter
	.long	7504                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x244a:0x5 DW_TAG_pointer_type
	.long	9295                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x244f:0xc DW_TAG_structure_type
	.short	472                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2452:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2454:0x5 DW_TAG_template_type_parameter
	.long	7504                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x245b:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	473                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2462:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2464:0x5 DW_TAG_template_type_parameter
	.long	7509                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x246b:0x5 DW_TAG_pointer_type
	.long	9328                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2470:0xc DW_TAG_structure_type
	.short	474                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2473:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2475:0x5 DW_TAG_template_type_parameter
	.long	7509                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x247c:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	475                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2483:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2485:0x5 DW_TAG_template_type_parameter
	.long	376                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x248c:0x5 DW_TAG_pointer_type
	.long	9361                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2491:0xc DW_TAG_structure_type
	.short	476                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2494:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2496:0x5 DW_TAG_template_type_parameter
	.long	376                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x249d:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	477                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x24a4:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x24a6:0x5 DW_TAG_template_type_parameter
	.long	7514                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x24ad:0x5 DW_TAG_pointer_type
	.long	9394                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x24b2:0xc DW_TAG_structure_type
	.short	478                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x24b5:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x24b7:0x5 DW_TAG_template_type_parameter
	.long	7514                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x24be:0x5 DW_TAG_pointer_type
	.long	9411                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x24c3:0x6 DW_TAG_structure_type
	.short	479                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	44                              # Abbrev [44] 0x24c6:0x2 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x24c9:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	480                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x24d0:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x24d2:0x5 DW_TAG_template_type_parameter
	.long	7497                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0x24d7:0x5 DW_TAG_template_type_parameter
	.long	7497                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x24de:0x5 DW_TAG_pointer_type
	.long	9443                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x24e3:0x11 DW_TAG_structure_type
	.short	481                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x24e6:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x24e8:0x5 DW_TAG_template_type_parameter
	.long	7497                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0x24ed:0x5 DW_TAG_template_type_parameter
	.long	7497                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x24f4:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	482                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x24fb:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x24fd:0x5 DW_TAG_template_type_parameter
	.long	7519                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2504:0x5 DW_TAG_pointer_type
	.long	9481                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2509:0xc DW_TAG_structure_type
	.short	483                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x250c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x250e:0x5 DW_TAG_template_type_parameter
	.long	7519                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2515:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	484                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x251c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x251e:0x5 DW_TAG_template_type_parameter
	.long	7540                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2525:0x5 DW_TAG_pointer_type
	.long	9514                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x252a:0xc DW_TAG_structure_type
	.short	485                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x252d:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x252f:0x5 DW_TAG_template_type_parameter
	.long	7540                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2536:0x5 DW_TAG_pointer_type
	.long	3243                            # DW_AT_type
	.byte	84                              # Abbrev [84] 0x253b:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	487                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2542:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2544:0x5 DW_TAG_template_type_parameter
	.long	381                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x254b:0x5 DW_TAG_pointer_type
	.long	9552                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2550:0xc DW_TAG_structure_type
	.short	488                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2553:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2555:0x5 DW_TAG_template_type_parameter
	.long	381                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x255c:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	489                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2563:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2565:0x5 DW_TAG_template_type_parameter
	.long	7551                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x256c:0x5 DW_TAG_pointer_type
	.long	9585                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2571:0xc DW_TAG_structure_type
	.short	490                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2574:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2576:0x5 DW_TAG_template_type_parameter
	.long	7551                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x257d:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	491                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2584:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2586:0x5 DW_TAG_template_type_parameter
	.long	7556                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x258d:0x5 DW_TAG_pointer_type
	.long	9618                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2592:0xc DW_TAG_structure_type
	.short	492                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2595:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2597:0x5 DW_TAG_template_type_parameter
	.long	7556                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x259e:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	493                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x25a5:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x25a7:0x5 DW_TAG_template_type_parameter
	.long	7328                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0x25ac:0x5 DW_TAG_template_type_parameter
	.long	7561                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x25b3:0x5 DW_TAG_pointer_type
	.long	9656                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x25b8:0x11 DW_TAG_structure_type
	.short	494                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x25bb:0xd DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x25bd:0x5 DW_TAG_template_type_parameter
	.long	7328                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0x25c2:0x5 DW_TAG_template_type_parameter
	.long	7561                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x25c9:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	495                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x25d0:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x25d2:0x5 DW_TAG_template_type_parameter
	.long	7566                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x25d9:0x5 DW_TAG_pointer_type
	.long	9694                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x25de:0xc DW_TAG_structure_type
	.short	496                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x25e1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x25e3:0x5 DW_TAG_template_type_parameter
	.long	7566                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x25ea:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	497                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x25f1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x25f3:0x5 DW_TAG_template_type_parameter
	.long	7580                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x25fa:0x5 DW_TAG_pointer_type
	.long	9727                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x25ff:0xc DW_TAG_structure_type
	.short	498                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2602:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2604:0x5 DW_TAG_template_type_parameter
	.long	7580                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x260b:0x1a DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	499                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2612:0x12 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2614:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	31                              # Abbrev [31] 0x2619:0x5 DW_TAG_template_type_parameter
	.long	7072                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0x261e:0x5 DW_TAG_template_type_parameter
	.long	7585                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2625:0x5 DW_TAG_pointer_type
	.long	9770                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x262a:0x16 DW_TAG_structure_type
	.short	500                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x262d:0x12 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x262f:0x5 DW_TAG_template_type_parameter
	.long	54                              # DW_AT_type
	.byte	31                              # Abbrev [31] 0x2634:0x5 DW_TAG_template_type_parameter
	.long	7072                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0x2639:0x5 DW_TAG_template_type_parameter
	.long	7585                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2640:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	501                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2647:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2649:0x5 DW_TAG_template_type_parameter
	.long	7590                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2650:0x5 DW_TAG_pointer_type
	.long	9813                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2655:0xc DW_TAG_structure_type
	.short	502                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2658:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x265a:0x5 DW_TAG_template_type_parameter
	.long	7590                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2661:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	503                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2668:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x266a:0x5 DW_TAG_template_type_parameter
	.long	7602                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2671:0x5 DW_TAG_pointer_type
	.long	9846                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2676:0xc DW_TAG_structure_type
	.short	504                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2679:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x267b:0x5 DW_TAG_template_type_parameter
	.long	7602                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2682:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	505                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2689:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x268b:0x5 DW_TAG_template_type_parameter
	.long	7612                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2692:0x5 DW_TAG_pointer_type
	.long	9879                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2697:0xc DW_TAG_structure_type
	.short	506                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x269a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x269c:0x5 DW_TAG_template_type_parameter
	.long	7612                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x26a3:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	507                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x26aa:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x26ac:0x5 DW_TAG_template_type_parameter
	.long	7618                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x26b3:0x5 DW_TAG_pointer_type
	.long	9912                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x26b8:0xc DW_TAG_structure_type
	.short	508                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x26bb:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x26bd:0x5 DW_TAG_template_type_parameter
	.long	7618                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x26c4:0x5 DW_TAG_pointer_type
	.long	206                             # DW_AT_type
	.byte	84                              # Abbrev [84] 0x26c9:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	509                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x26d0:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x26d2:0x5 DW_TAG_template_type_parameter
	.long	7634                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x26d9:0x5 DW_TAG_pointer_type
	.long	9950                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x26de:0xc DW_TAG_structure_type
	.short	510                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x26e1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x26e3:0x5 DW_TAG_template_type_parameter
	.long	7634                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x26ea:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	511                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x26f1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x26f3:0x5 DW_TAG_template_type_parameter
	.long	7660                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x26fa:0x5 DW_TAG_pointer_type
	.long	9983                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x26ff:0xc DW_TAG_structure_type
	.short	512                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2702:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2704:0x5 DW_TAG_template_type_parameter
	.long	7660                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x270b:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	513                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2712:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2714:0x5 DW_TAG_template_type_parameter
	.long	7686                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x271b:0x5 DW_TAG_pointer_type
	.long	10016                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2720:0xc DW_TAG_structure_type
	.short	514                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2723:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2725:0x5 DW_TAG_template_type_parameter
	.long	7686                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x272c:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	515                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2733:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2735:0x5 DW_TAG_template_type_parameter
	.long	7712                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x273c:0x5 DW_TAG_pointer_type
	.long	10049                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2741:0xc DW_TAG_structure_type
	.short	516                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2744:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2746:0x5 DW_TAG_template_type_parameter
	.long	7712                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x274d:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	517                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2754:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2756:0x5 DW_TAG_template_type_parameter
	.long	7717                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x275d:0x5 DW_TAG_pointer_type
	.long	10082                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2762:0xc DW_TAG_structure_type
	.short	518                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2765:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2767:0x5 DW_TAG_template_type_parameter
	.long	7717                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x276e:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	519                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2775:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2777:0x5 DW_TAG_template_type_parameter
	.long	7739                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x277e:0x5 DW_TAG_pointer_type
	.long	10115                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2783:0xc DW_TAG_structure_type
	.short	520                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2786:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2788:0x5 DW_TAG_template_type_parameter
	.long	7739                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x278f:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	521                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2796:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2798:0x5 DW_TAG_template_type_parameter
	.long	7745                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x279f:0x5 DW_TAG_pointer_type
	.long	10148                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x27a4:0xc DW_TAG_structure_type
	.short	522                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x27a7:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x27a9:0x5 DW_TAG_template_type_parameter
	.long	7745                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x27b0:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	523                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x27b7:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x27b9:0x5 DW_TAG_template_type_parameter
	.long	7751                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x27c0:0x5 DW_TAG_pointer_type
	.long	10181                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x27c5:0xc DW_TAG_structure_type
	.short	524                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x27c8:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x27ca:0x5 DW_TAG_template_type_parameter
	.long	7751                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x27d1:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	525                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x27d8:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x27da:0x5 DW_TAG_template_type_parameter
	.long	7761                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x27e1:0x5 DW_TAG_pointer_type
	.long	10214                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x27e6:0xc DW_TAG_structure_type
	.short	526                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x27e9:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x27eb:0x5 DW_TAG_template_type_parameter
	.long	7761                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x27f2:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	527                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x27f9:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x27fb:0x5 DW_TAG_template_type_parameter
	.long	7778                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2802:0x5 DW_TAG_pointer_type
	.long	10247                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2807:0xc DW_TAG_structure_type
	.short	528                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x280a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x280c:0x5 DW_TAG_template_type_parameter
	.long	7778                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2813:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	529                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x281a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x281c:0x5 DW_TAG_template_type_parameter
	.long	7783                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2823:0x5 DW_TAG_pointer_type
	.long	10280                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2828:0xc DW_TAG_structure_type
	.short	530                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x282b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x282d:0x5 DW_TAG_template_type_parameter
	.long	7783                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2834:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	531                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x283b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x283d:0x5 DW_TAG_template_type_parameter
	.long	7814                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2844:0x5 DW_TAG_pointer_type
	.long	10313                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2849:0xc DW_TAG_structure_type
	.short	532                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x284c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x284e:0x5 DW_TAG_template_type_parameter
	.long	7814                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2855:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	533                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x285c:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x285e:0x5 DW_TAG_template_type_parameter
	.long	7837                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2865:0x5 DW_TAG_pointer_type
	.long	10346                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x286a:0xc DW_TAG_structure_type
	.short	534                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x286d:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x286f:0x5 DW_TAG_template_type_parameter
	.long	7837                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2876:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	535                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x287d:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x287f:0x5 DW_TAG_template_type_parameter
	.long	7504                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2886:0x5 DW_TAG_pointer_type
	.long	10379                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x288b:0xc DW_TAG_structure_type
	.short	536                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x288e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2890:0x5 DW_TAG_template_type_parameter
	.long	7504                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x2897:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	537                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x289e:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x28a0:0x5 DW_TAG_template_type_parameter
	.long	7849                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x28a7:0x5 DW_TAG_pointer_type
	.long	10412                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x28ac:0xc DW_TAG_structure_type
	.short	538                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x28af:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x28b1:0x5 DW_TAG_template_type_parameter
	.long	7849                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x28b8:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	539                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x28bf:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x28c1:0x5 DW_TAG_template_type_parameter
	.long	7856                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x28c8:0x5 DW_TAG_pointer_type
	.long	10445                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x28cd:0xc DW_TAG_structure_type
	.short	540                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x28d0:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x28d2:0x5 DW_TAG_template_type_parameter
	.long	7856                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x28d9:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	541                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x28e0:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x28e2:0x5 DW_TAG_template_type_parameter
	.long	7868                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x28e9:0x5 DW_TAG_pointer_type
	.long	10478                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x28ee:0xc DW_TAG_structure_type
	.short	542                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x28f1:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x28f3:0x5 DW_TAG_template_type_parameter
	.long	7868                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x28fa:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	543                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2901:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2903:0x5 DW_TAG_template_type_parameter
	.long	7909                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x290a:0x5 DW_TAG_pointer_type
	.long	10511                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x290f:0xc DW_TAG_structure_type
	.short	544                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2912:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2914:0x5 DW_TAG_template_type_parameter
	.long	7909                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x291b:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	545                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2922:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2924:0x5 DW_TAG_template_type_parameter
	.long	7931                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x292b:0x5 DW_TAG_pointer_type
	.long	10544                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2930:0xc DW_TAG_structure_type
	.short	546                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2933:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2935:0x5 DW_TAG_template_type_parameter
	.long	7931                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x293c:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	547                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2943:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2945:0x5 DW_TAG_template_type_parameter
	.long	7940                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x294c:0x5 DW_TAG_pointer_type
	.long	10577                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2951:0xc DW_TAG_structure_type
	.short	548                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2954:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2956:0x5 DW_TAG_template_type_parameter
	.long	7940                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x295d:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	549                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x2964:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2966:0x5 DW_TAG_template_type_parameter
	.long	7942                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x296d:0x5 DW_TAG_pointer_type
	.long	10610                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2972:0xc DW_TAG_structure_type
	.short	550                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x2975:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2977:0x5 DW_TAG_template_type_parameter
	.long	7942                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x297e:0x5 DW_TAG_pointer_type
	.long	6762                            # DW_AT_type
	.byte	84                              # Abbrev [84] 0x2983:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	487                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x298a:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x298c:0x5 DW_TAG_template_type_parameter
	.long	6814                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x2993:0x5 DW_TAG_pointer_type
	.long	10648                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2998:0xc DW_TAG_structure_type
	.short	488                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x299b:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x299d:0x5 DW_TAG_template_type_parameter
	.long	6814                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	84                              # Abbrev [84] 0x29a4:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.short	551                             # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	30                              # Abbrev [30] 0x29ab:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x29ad:0x5 DW_TAG_template_type_parameter
	.long	7947                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	47                              # Abbrev [47] 0x29b4:0x5 DW_TAG_pointer_type
	.long	10681                           # DW_AT_type
	.byte	67                              # Abbrev [67] 0x29b9:0xc DW_TAG_structure_type
	.short	552                             # DW_AT_name
                                        # DW_AT_declaration
	.byte	30                              # Abbrev [30] 0x29bc:0x8 DW_TAG_GNU_template_parameter_pack
	.byte	34                              # DW_AT_name
	.byte	31                              # Abbrev [31] 0x29be:0x5 DW_TAG_template_type_parameter
	.long	7947                            # DW_AT_type
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
	.uleb128 .Lfunc_end142-.Lfunc_begin0    #   ending offset
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
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	2232                            # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 22.0.0git (git@github.com:Michael137/llvm-project.git f45bb984e6f21e702b4d65f1eeea1429f43c800e)" # string offset=0
.Linfo_string1:
	.asciz	"cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp" # string offset=110
.Linfo_string2:
	.asciz	"/Users/michaelbuch/Git/llvm-worktrees/main" # string offset=197
.Linfo_string3:
	.asciz	"i"                             # string offset=240
.Linfo_string4:
	.asciz	"int"                           # string offset=242
.Linfo_string5:
	.asciz	"unsigned int"                  # string offset=246
.Linfo_string6:
	.asciz	"LocalEnum1"                    # string offset=259
.Linfo_string7:
	.asciz	"LocalEnum"                     # string offset=270
.Linfo_string8:
	.asciz	"ns"                            # string offset=280
.Linfo_string9:
	.asciz	"Enumerator1"                   # string offset=283
.Linfo_string10:
	.asciz	"Enumerator2"                   # string offset=295
.Linfo_string11:
	.asciz	"Enumerator3"                   # string offset=307
.Linfo_string12:
	.asciz	"Enumeration"                   # string offset=319
.Linfo_string13:
	.asciz	"EnumerationClass"              # string offset=331
.Linfo_string14:
	.asciz	"unsigned char"                 # string offset=348
.Linfo_string15:
	.asciz	"kNeg"                          # string offset=362
.Linfo_string16:
	.asciz	"EnumerationSmall"              # string offset=367
.Linfo_string17:
	.asciz	"AnonEnum1"                     # string offset=384
.Linfo_string18:
	.asciz	"AnonEnum2"                     # string offset=394
.Linfo_string19:
	.asciz	"AnonEnum3"                     # string offset=404
.Linfo_string20:
	.asciz	"T"                             # string offset=414
.Linfo_string21:
	.asciz	"bool"                          # string offset=416
.Linfo_string22:
	.asciz	"b"                             # string offset=421
.Linfo_string23:
	.asciz	"_STN|t3|<int, false>"          # string offset=423
.Linfo_string24:
	.asciz	"t10"                           # string offset=444
.Linfo_string25:
	.asciz	"t6"                            # string offset=448
.Linfo_string26:
	.asciz	"_ZN2t6lsIiEEvi"                # string offset=451
.Linfo_string27:
	.asciz	"operator<<<int>"               # string offset=466
.Linfo_string28:
	.asciz	"_ZN2t6ltIiEEvi"                # string offset=482
.Linfo_string29:
	.asciz	"operator<<int>"                # string offset=497
.Linfo_string30:
	.asciz	"_ZN2t6leIiEEvi"                # string offset=512
.Linfo_string31:
	.asciz	"operator<=<int>"               # string offset=527
.Linfo_string32:
	.asciz	"_ZN2t6cvP2t1IJfEEIiEEv"        # string offset=543
.Linfo_string33:
	.asciz	"operator t1<float> *<int>"     # string offset=566
.Linfo_string34:
	.asciz	"Ts"                            # string offset=592
.Linfo_string35:
	.asciz	"float"                         # string offset=595
.Linfo_string36:
	.asciz	"_STN|t1|<float>"               # string offset=601
.Linfo_string37:
	.asciz	"_ZN2t6miIiEEvi"                # string offset=617
.Linfo_string38:
	.asciz	"operator-<int>"                # string offset=632
.Linfo_string39:
	.asciz	"_ZN2t6mlIiEEvi"                # string offset=647
.Linfo_string40:
	.asciz	"operator*<int>"                # string offset=662
.Linfo_string41:
	.asciz	"_ZN2t6dvIiEEvi"                # string offset=677
.Linfo_string42:
	.asciz	"operator/<int>"                # string offset=692
.Linfo_string43:
	.asciz	"_ZN2t6rmIiEEvi"                # string offset=707
.Linfo_string44:
	.asciz	"operator%<int>"                # string offset=722
.Linfo_string45:
	.asciz	"_ZN2t6eoIiEEvi"                # string offset=737
.Linfo_string46:
	.asciz	"operator^<int>"                # string offset=752
.Linfo_string47:
	.asciz	"_ZN2t6anIiEEvi"                # string offset=767
.Linfo_string48:
	.asciz	"operator&<int>"                # string offset=782
.Linfo_string49:
	.asciz	"_ZN2t6orIiEEvi"                # string offset=797
.Linfo_string50:
	.asciz	"operator|<int>"                # string offset=812
.Linfo_string51:
	.asciz	"_ZN2t6coIiEEvv"                # string offset=827
.Linfo_string52:
	.asciz	"operator~<int>"                # string offset=842
.Linfo_string53:
	.asciz	"_ZN2t6ntIiEEvv"                # string offset=857
.Linfo_string54:
	.asciz	"operator!<int>"                # string offset=872
.Linfo_string55:
	.asciz	"_ZN2t6aSIiEEvi"                # string offset=887
.Linfo_string56:
	.asciz	"operator=<int>"                # string offset=902
.Linfo_string57:
	.asciz	"_ZN2t6gtIiEEvi"                # string offset=917
.Linfo_string58:
	.asciz	"operator><int>"                # string offset=932
.Linfo_string59:
	.asciz	"_ZN2t6cmIiEEvi"                # string offset=947
.Linfo_string60:
	.asciz	"operator,<int>"                # string offset=962
.Linfo_string61:
	.asciz	"_ZN2t6clIiEEvv"                # string offset=977
.Linfo_string62:
	.asciz	"operator()<int>"               # string offset=992
.Linfo_string63:
	.asciz	"_ZN2t6ixIiEEvi"                # string offset=1008
.Linfo_string64:
	.asciz	"operator[]<int>"               # string offset=1023
.Linfo_string65:
	.asciz	"_ZN2t6ssIiEEvi"                # string offset=1039
.Linfo_string66:
	.asciz	"operator<=><int>"              # string offset=1054
.Linfo_string67:
	.asciz	"_ZN2t6nwIiEEPvmT_"             # string offset=1071
.Linfo_string68:
	.asciz	"operator new<int>"             # string offset=1089
.Linfo_string69:
	.asciz	"std"                           # string offset=1107
.Linfo_string70:
	.asciz	"__1"                           # string offset=1111
.Linfo_string71:
	.asciz	"unsigned long"                 # string offset=1115
.Linfo_string72:
	.asciz	"size_t"                        # string offset=1129
.Linfo_string73:
	.asciz	"_ZN2t6naIiEEPvmT_"             # string offset=1136
.Linfo_string74:
	.asciz	"operator new[]<int>"           # string offset=1154
.Linfo_string75:
	.asciz	"_ZN2t6dlIiEEvPvT_"             # string offset=1174
.Linfo_string76:
	.asciz	"operator delete<int>"          # string offset=1192
.Linfo_string77:
	.asciz	"_ZN2t6daIiEEvPvT_"             # string offset=1213
.Linfo_string78:
	.asciz	"operator delete[]<int>"        # string offset=1231
.Linfo_string79:
	.asciz	"_ZN2t6awIiEEiv"                # string offset=1254
.Linfo_string80:
	.asciz	"operator co_await<int>"        # string offset=1269
.Linfo_string81:
	.asciz	"_ZN3t10C4IvEEv"                # string offset=1292
.Linfo_string82:
	.asciz	"_STN|t10|<void>"               # string offset=1307
.Linfo_string83:
	.asciz	"_ZN2t83memEv"                  # string offset=1323
.Linfo_string84:
	.asciz	"mem"                           # string offset=1336
.Linfo_string85:
	.asciz	"t8"                            # string offset=1340
.Linfo_string86:
	.asciz	"complex_type_units"            # string offset=1343
.Linfo_string87:
	.asciz	"signed char"                   # string offset=1362
.Linfo_string88:
	.asciz	"int8_t"                        # string offset=1374
.Linfo_string89:
	.asciz	"short"                         # string offset=1381
.Linfo_string90:
	.asciz	"int16_t"                       # string offset=1387
.Linfo_string91:
	.asciz	"int32_t"                       # string offset=1395
.Linfo_string92:
	.asciz	"long"                          # string offset=1403
.Linfo_string93:
	.asciz	"int64_t"                       # string offset=1408
.Linfo_string94:
	.asciz	"uint8_t"                       # string offset=1416
.Linfo_string95:
	.asciz	"unsigned short"                # string offset=1424
.Linfo_string96:
	.asciz	"uint16_t"                      # string offset=1439
.Linfo_string97:
	.asciz	"uint32_t"                      # string offset=1448
.Linfo_string98:
	.asciz	"uint64_t"                      # string offset=1457
.Linfo_string99:
	.asciz	"int_least8_t"                  # string offset=1466
.Linfo_string100:
	.asciz	"int_least16_t"                 # string offset=1479
.Linfo_string101:
	.asciz	"int_least32_t"                 # string offset=1493
.Linfo_string102:
	.asciz	"int_least64_t"                 # string offset=1507
.Linfo_string103:
	.asciz	"uint_least8_t"                 # string offset=1521
.Linfo_string104:
	.asciz	"uint_least16_t"                # string offset=1535
.Linfo_string105:
	.asciz	"uint_least32_t"                # string offset=1550
.Linfo_string106:
	.asciz	"uint_least64_t"                # string offset=1565
.Linfo_string107:
	.asciz	"int_fast8_t"                   # string offset=1580
.Linfo_string108:
	.asciz	"int_fast16_t"                  # string offset=1592
.Linfo_string109:
	.asciz	"int_fast32_t"                  # string offset=1605
.Linfo_string110:
	.asciz	"int_fast64_t"                  # string offset=1618
.Linfo_string111:
	.asciz	"uint_fast8_t"                  # string offset=1631
.Linfo_string112:
	.asciz	"uint_fast16_t"                 # string offset=1644
.Linfo_string113:
	.asciz	"uint_fast32_t"                 # string offset=1658
.Linfo_string114:
	.asciz	"uint_fast64_t"                 # string offset=1672
.Linfo_string115:
	.asciz	"intptr_t"                      # string offset=1686
.Linfo_string116:
	.asciz	"uintptr_t"                     # string offset=1695
.Linfo_string117:
	.asciz	"intmax_t"                      # string offset=1705
.Linfo_string118:
	.asciz	"uintmax_t"                     # string offset=1714
.Linfo_string119:
	.asciz	"max_align_t"                   # string offset=1724
.Linfo_string120:
	.asciz	"_Zli5_suffy"                   # string offset=1736
.Linfo_string121:
	.asciz	"operator\"\"_suff"             # string offset=1748
.Linfo_string122:
	.asciz	"main"                          # string offset=1764
.Linfo_string123:
	.asciz	"_Z2f1IJiEEvv"                  # string offset=1769
.Linfo_string124:
	.asciz	"_STN|f1|<int>"                 # string offset=1782
.Linfo_string125:
	.asciz	"_Z2f1IJfEEvv"                  # string offset=1796
.Linfo_string126:
	.asciz	"_STN|f1|<float>"               # string offset=1809
.Linfo_string127:
	.asciz	"_Z2f1IJbEEvv"                  # string offset=1825
.Linfo_string128:
	.asciz	"_STN|f1|<bool>"                # string offset=1838
.Linfo_string129:
	.asciz	"double"                        # string offset=1853
.Linfo_string130:
	.asciz	"_Z2f1IJdEEvv"                  # string offset=1860
.Linfo_string131:
	.asciz	"_STN|f1|<double>"              # string offset=1873
.Linfo_string132:
	.asciz	"_Z2f1IJlEEvv"                  # string offset=1890
.Linfo_string133:
	.asciz	"_STN|f1|<long>"                # string offset=1903
.Linfo_string134:
	.asciz	"_Z2f1IJsEEvv"                  # string offset=1918
.Linfo_string135:
	.asciz	"_STN|f1|<short>"               # string offset=1931
.Linfo_string136:
	.asciz	"_Z2f1IJjEEvv"                  # string offset=1947
.Linfo_string137:
	.asciz	"_STN|f1|<unsigned int>"        # string offset=1960
.Linfo_string138:
	.asciz	"unsigned long long"            # string offset=1983
.Linfo_string139:
	.asciz	"_Z2f1IJyEEvv"                  # string offset=2002
.Linfo_string140:
	.asciz	"_STN|f1|<unsigned long long>"  # string offset=2015
.Linfo_string141:
	.asciz	"long long"                     # string offset=2044
.Linfo_string142:
	.asciz	"_Z2f1IJxEEvv"                  # string offset=2054
.Linfo_string143:
	.asciz	"_STN|f1|<long long>"           # string offset=2067
.Linfo_string144:
	.asciz	"udt"                           # string offset=2087
.Linfo_string145:
	.asciz	"_Z2f1IJ3udtEEvv"               # string offset=2091
.Linfo_string146:
	.asciz	"_STN|f1|<udt>"                 # string offset=2107
.Linfo_string147:
	.asciz	"_Z2f1IJN2ns3udtEEEvv"          # string offset=2121
.Linfo_string148:
	.asciz	"_STN|f1|<ns::udt>"             # string offset=2142
.Linfo_string149:
	.asciz	"_Z2f1IJPN2ns3udtEEEvv"         # string offset=2160
.Linfo_string150:
	.asciz	"_STN|f1|<ns::udt *>"           # string offset=2182
.Linfo_string151:
	.asciz	"inner"                         # string offset=2202
.Linfo_string152:
	.asciz	"_Z2f1IJN2ns5inner3udtEEEvv"    # string offset=2208
.Linfo_string153:
	.asciz	"_STN|f1|<ns::inner::udt>"      # string offset=2235
.Linfo_string154:
	.asciz	"_STN|t1|<int>"                 # string offset=2260
.Linfo_string155:
	.asciz	"_Z2f1IJ2t1IJiEEEEvv"           # string offset=2274
.Linfo_string156:
	.asciz	"_STN|f1|<t1<int> >"            # string offset=2294
.Linfo_string157:
	.asciz	"_Z2f1IJifEEvv"                 # string offset=2313
.Linfo_string158:
	.asciz	"_STN|f1|<int, float>"          # string offset=2327
.Linfo_string159:
	.asciz	"_Z2f1IJPiEEvv"                 # string offset=2348
.Linfo_string160:
	.asciz	"_STN|f1|<int *>"               # string offset=2362
.Linfo_string161:
	.asciz	"_Z2f1IJRiEEvv"                 # string offset=2378
.Linfo_string162:
	.asciz	"_STN|f1|<int &>"               # string offset=2392
.Linfo_string163:
	.asciz	"_Z2f1IJOiEEvv"                 # string offset=2408
.Linfo_string164:
	.asciz	"_STN|f1|<int &&>"              # string offset=2422
.Linfo_string165:
	.asciz	"_Z2f1IJKiEEvv"                 # string offset=2439
.Linfo_string166:
	.asciz	"_STN|f1|<const int>"           # string offset=2453
.Linfo_string167:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=2473
.Linfo_string168:
	.asciz	"_Z2f1IJA3_iEEvv"               # string offset=2493
.Linfo_string169:
	.asciz	"_STN|f1|<int[3]>"              # string offset=2509
.Linfo_string170:
	.asciz	"_Z2f1IJvEEvv"                  # string offset=2526
.Linfo_string171:
	.asciz	"_STN|f1|<void>"                # string offset=2539
.Linfo_string172:
	.asciz	"outer_class"                   # string offset=2554
.Linfo_string173:
	.asciz	"inner_class"                   # string offset=2566
.Linfo_string174:
	.asciz	"_Z2f1IJN11outer_class11inner_classEEEvv" # string offset=2578
.Linfo_string175:
	.asciz	"_STN|f1|<outer_class::inner_class>" # string offset=2618
.Linfo_string176:
	.asciz	"_Z2f1IJmEEvv"                  # string offset=2653
.Linfo_string177:
	.asciz	"_STN|f1|<unsigned long>"       # string offset=2666
.Linfo_string178:
	.asciz	"_Z2f2ILb1ELi3EEvv"             # string offset=2690
.Linfo_string179:
	.asciz	"_STN|f2|<true, 3>"             # string offset=2708
.Linfo_string180:
	.asciz	"A"                             # string offset=2726
.Linfo_string181:
	.asciz	"_Z2f3IN2ns11EnumerationETpTnT_JLS1_1ELS1_2EEEvv" # string offset=2728
.Linfo_string182:
	.asciz	"_STN|f3|<ns::Enumeration, (ns::Enumeration)1, (ns::Enumeration)2>" # string offset=2776
.Linfo_string183:
	.asciz	"_Z2f3IN2ns16EnumerationClassETpTnT_JLS1_1ELS1_2EEEvv" # string offset=2842
.Linfo_string184:
	.asciz	"_STN|f3|<ns::EnumerationClass, (ns::EnumerationClass)1, (ns::EnumerationClass)2>" # string offset=2895
.Linfo_string185:
	.asciz	"_Z2f3IN2ns16EnumerationSmallETpTnT_JLS1_255EEEvv" # string offset=2976
.Linfo_string186:
	.asciz	"_STN|f3|<ns::EnumerationSmall, (ns::EnumerationSmall)255>" # string offset=3025
.Linfo_string187:
	.asciz	"_Z2f3IN2ns3$_0ETpTnT_JLS1_1ELS1_2EEEvv" # string offset=3083
.Linfo_string188:
	.asciz	"f3<ns::(unnamed enum at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:31:1), (ns::(unnamed enum at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:31:1))1, (ns::(unnamed enum at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:31:1))2>" # string offset=3122
.Linfo_string189:
	.asciz	"_Z2f3IN12_GLOBAL__N_19LocalEnumETpTnT_JLS1_0EEEvv" # string offset=3476
.Linfo_string190:
	.asciz	"f3<(anonymous namespace)::LocalEnum, ((anonymous namespace)::LocalEnum)0>" # string offset=3526
.Linfo_string191:
	.asciz	"_Z2f3IPiTpTnT_JXadL_Z1iEEEEvv" # string offset=3600
.Linfo_string192:
	.asciz	"f3<int *, &i>"                 # string offset=3630
.Linfo_string193:
	.asciz	"_Z2f3IPiTpTnT_JLS0_0EEEvv"     # string offset=3644
.Linfo_string194:
	.asciz	"f3<int *, nullptr>"            # string offset=3670
.Linfo_string195:
	.asciz	"_Z2f3ImTpTnT_JLm1EEEvv"        # string offset=3689
.Linfo_string196:
	.asciz	"_STN|f3|<unsigned long, 1UL>"  # string offset=3712
.Linfo_string197:
	.asciz	"_Z2f3IyTpTnT_JLy1EEEvv"        # string offset=3741
.Linfo_string198:
	.asciz	"_STN|f3|<unsigned long long, 1ULL>" # string offset=3764
.Linfo_string199:
	.asciz	"_Z2f3IlTpTnT_JLl1EEEvv"        # string offset=3799
.Linfo_string200:
	.asciz	"_STN|f3|<long, 1L>"            # string offset=3822
.Linfo_string201:
	.asciz	"_Z2f3IjTpTnT_JLj1EEEvv"        # string offset=3841
.Linfo_string202:
	.asciz	"_STN|f3|<unsigned int, 1U>"    # string offset=3864
.Linfo_string203:
	.asciz	"_Z2f3IsTpTnT_JLs1EEEvv"        # string offset=3891
.Linfo_string204:
	.asciz	"_STN|f3|<short, (short)1>"     # string offset=3914
.Linfo_string205:
	.asciz	"_Z2f3IhTpTnT_JLh0EEEvv"        # string offset=3940
.Linfo_string206:
	.asciz	"_STN|f3|<unsigned char, (unsigned char)'\\x00'>" # string offset=3963
.Linfo_string207:
	.asciz	"_Z2f3IaTpTnT_JLa0EEEvv"        # string offset=4010
.Linfo_string208:
	.asciz	"_STN|f3|<signed char, (signed char)'\\x00'>" # string offset=4033
.Linfo_string209:
	.asciz	"_Z2f3ItTpTnT_JLt1ELt2EEEvv"    # string offset=4076
.Linfo_string210:
	.asciz	"_STN|f3|<unsigned short, (unsigned short)1, (unsigned short)2>" # string offset=4103
.Linfo_string211:
	.asciz	"char"                          # string offset=4166
.Linfo_string212:
	.asciz	"_Z2f3IcTpTnT_JLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv" # string offset=4171
.Linfo_string213:
	.asciz	"_STN|f3|<char, '\\x00', '\\x01', '\\x06', '\\a', '\\r', '\\x0e', '\\x1f', ' ', '!', '\\x7f', '\\x80'>" # string offset=4244
.Linfo_string214:
	.asciz	"__int128"                      # string offset=4337
.Linfo_string215:
	.asciz	"_Z2f3InTpTnT_JLn18446744073709551614EEEvv" # string offset=4346
.Linfo_string216:
	.asciz	"f3<__int128, (__int128)18446744073709551614>" # string offset=4388
.Linfo_string217:
	.asciz	"_Z2f4IjLj3EEvv"                # string offset=4433
.Linfo_string218:
	.asciz	"_STN|f4|<unsigned int, 3U>"    # string offset=4448
.Linfo_string219:
	.asciz	"_Z2f1IJ2t3IiLb0EEEEvv"         # string offset=4475
.Linfo_string220:
	.asciz	"_STN|f1|<t3<int, false> >"     # string offset=4497
.Linfo_string221:
	.asciz	"_STN|t3|<t3<int, false>, false>" # string offset=4523
.Linfo_string222:
	.asciz	"_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv" # string offset=4555
.Linfo_string223:
	.asciz	"_STN|f1|<t3<t3<int, false>, false> >" # string offset=4586
.Linfo_string224:
	.asciz	"_Z2f1IJZ4mainE3$_0EEvv"        # string offset=4623
.Linfo_string225:
	.asciz	"f1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:192:12)>" # string offset=4646
.Linfo_string226:
	.asciz	"t3<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:192:12), false>" # string offset=4756
.Linfo_string227:
	.asciz	"t3<t3<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:192:12), false>, false>" # string offset=4873
.Linfo_string228:
	.asciz	"_Z2f1IJ2t3IS0_IZ4mainE3$_0Lb0EELb0EEEEvv" # string offset=5001
.Linfo_string229:
	.asciz	"f1<t3<t3<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:192:12), false>, false> >" # string offset=5042
.Linfo_string230:
	.asciz	"_Z2f1IJFifEEEvv"               # string offset=5175
.Linfo_string231:
	.asciz	"_STN|f1|<int (float)>"         # string offset=5191
.Linfo_string232:
	.asciz	"_Z2f1IJFvzEEEvv"               # string offset=5213
.Linfo_string233:
	.asciz	"_STN|f1|<void (...)>"          # string offset=5229
.Linfo_string234:
	.asciz	"_Z2f1IJFvizEEEvv"              # string offset=5250
.Linfo_string235:
	.asciz	"_STN|f1|<void (int, ...)>"     # string offset=5267
.Linfo_string236:
	.asciz	"_Z2f1IJRKiEEvv"                # string offset=5293
.Linfo_string237:
	.asciz	"_STN|f1|<const int &>"         # string offset=5308
.Linfo_string238:
	.asciz	"_Z2f1IJRPKiEEvv"               # string offset=5330
.Linfo_string239:
	.asciz	"_STN|f1|<const int *&>"        # string offset=5346
.Linfo_string240:
	.asciz	"t5"                            # string offset=5369
.Linfo_string241:
	.asciz	"_Z2f1IJN12_GLOBAL__N_12t5EEEvv" # string offset=5372
.Linfo_string242:
	.asciz	"_STN|f1|<(anonymous namespace)::t5>" # string offset=5403
.Linfo_string243:
	.asciz	"decltype(nullptr)"             # string offset=5439
.Linfo_string244:
	.asciz	"_Z2f1IJDnEEvv"                 # string offset=5457
.Linfo_string245:
	.asciz	"_STN|f1|<std::nullptr_t>"      # string offset=5471
.Linfo_string246:
	.asciz	"_Z2f1IJPlS0_EEvv"              # string offset=5496
.Linfo_string247:
	.asciz	"_STN|f1|<long *, long *>"      # string offset=5513
.Linfo_string248:
	.asciz	"_Z2f1IJPlP3udtEEvv"            # string offset=5538
.Linfo_string249:
	.asciz	"_STN|f1|<long *, udt *>"       # string offset=5557
.Linfo_string250:
	.asciz	"_Z2f1IJKPvEEvv"                # string offset=5581
.Linfo_string251:
	.asciz	"_STN|f1|<void *const>"         # string offset=5596
.Linfo_string252:
	.asciz	"_Z2f1IJPKPKvEEvv"              # string offset=5618
.Linfo_string253:
	.asciz	"_STN|f1|<const void *const *>" # string offset=5635
.Linfo_string254:
	.asciz	"_Z2f1IJFvvEEEvv"               # string offset=5665
.Linfo_string255:
	.asciz	"_STN|f1|<void ()>"             # string offset=5681
.Linfo_string256:
	.asciz	"_Z2f1IJPFvvEEEvv"              # string offset=5699
.Linfo_string257:
	.asciz	"_STN|f1|<void (*)()>"          # string offset=5716
.Linfo_string258:
	.asciz	"_Z2f1IJPZ4mainE3$_0EEvv"       # string offset=5737
.Linfo_string259:
	.asciz	"f1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:192:12) *>" # string offset=5761
.Linfo_string260:
	.asciz	"_Z2f1IJZ4mainE3$_1EEvv"        # string offset=5873
.Linfo_string261:
	.asciz	"f1<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:191:3)>" # string offset=5896
.Linfo_string262:
	.asciz	"_Z2f1IJPZ4mainE3$_1EEvv"       # string offset=6013
.Linfo_string263:
	.asciz	"f1<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:191:3) *>" # string offset=6037
.Linfo_string264:
	.asciz	"T1"                            # string offset=6156
.Linfo_string265:
	.asciz	"T2"                            # string offset=6159
.Linfo_string266:
	.asciz	"_Z2f5IJ2t1IJiEEEiEvv"          # string offset=6162
.Linfo_string267:
	.asciz	"_STN|f5|<t1<int>, int>"        # string offset=6183
.Linfo_string268:
	.asciz	"_Z2f5IJEiEvv"                  # string offset=6206
.Linfo_string269:
	.asciz	"_STN|f5|<int>"                 # string offset=6219
.Linfo_string270:
	.asciz	"_Z2f6I2t1IJiEEJEEvv"           # string offset=6233
.Linfo_string271:
	.asciz	"_STN|f6|<t1<int> >"            # string offset=6253
.Linfo_string272:
	.asciz	"_Z2f1IJEEvv"                   # string offset=6272
.Linfo_string273:
	.asciz	"_STN|f1|<>"                    # string offset=6284
.Linfo_string274:
	.asciz	"_Z2f1IJPKvS1_EEvv"             # string offset=6295
.Linfo_string275:
	.asciz	"_STN|f1|<const void *, const void *>" # string offset=6313
.Linfo_string276:
	.asciz	"_STN|t1|<int *>"               # string offset=6350
.Linfo_string277:
	.asciz	"_Z2f1IJP2t1IJPiEEEEvv"         # string offset=6366
.Linfo_string278:
	.asciz	"_STN|f1|<t1<int *> *>"         # string offset=6388
.Linfo_string279:
	.asciz	"_Z2f1IJA_PiEEvv"               # string offset=6410
.Linfo_string280:
	.asciz	"_STN|f1|<int *[]>"             # string offset=6426
.Linfo_string281:
	.asciz	"t7"                            # string offset=6444
.Linfo_string282:
	.asciz	"_Z2f1IJZ4mainE2t7EEvv"         # string offset=6447
.Linfo_string283:
	.asciz	"_STN|f1|<t7>"                  # string offset=6469
.Linfo_string284:
	.asciz	"_Z2f1IJRA3_iEEvv"              # string offset=6482
.Linfo_string285:
	.asciz	"_STN|f1|<int (&)[3]>"          # string offset=6499
.Linfo_string286:
	.asciz	"_Z2f1IJPA3_iEEvv"              # string offset=6520
.Linfo_string287:
	.asciz	"_STN|f1|<int (*)[3]>"          # string offset=6537
.Linfo_string288:
	.asciz	"t1"                            # string offset=6558
.Linfo_string289:
	.asciz	"_Z2f7I2t1Evv"                  # string offset=6561
.Linfo_string290:
	.asciz	"_STN|f7|<t1>"                  # string offset=6574
.Linfo_string291:
	.asciz	"_Z2f8I2t1iEvv"                 # string offset=6587
.Linfo_string292:
	.asciz	"_STN|f8|<t1, int>"             # string offset=6601
.Linfo_string293:
	.asciz	"ns::inner::ttp"                # string offset=6619
.Linfo_string294:
	.asciz	"_ZN2ns8ttp_userINS_5inner3ttpEEEvv" # string offset=6634
.Linfo_string295:
	.asciz	"_STN|ttp_user|<ns::inner::ttp>" # string offset=6669
.Linfo_string296:
	.asciz	"_Z2f1IJPiPDnEEvv"              # string offset=6700
.Linfo_string297:
	.asciz	"_STN|f1|<int *, std::nullptr_t *>" # string offset=6717
.Linfo_string298:
	.asciz	"_STN|t7|<int>"                 # string offset=6751
.Linfo_string299:
	.asciz	"_Z2f1IJ2t7IiEEEvv"             # string offset=6765
.Linfo_string300:
	.asciz	"_STN|f1|<t7<int> >"            # string offset=6783
.Linfo_string301:
	.asciz	"ns::inl::t9"                   # string offset=6802
.Linfo_string302:
	.asciz	"_Z2f7ITtTpTyEN2ns3inl2t9EEvv"  # string offset=6814
.Linfo_string303:
	.asciz	"_STN|f7|<ns::inl::t9>"         # string offset=6843
.Linfo_string304:
	.asciz	"_Z2f1IJU7_AtomiciEEvv"         # string offset=6865
.Linfo_string305:
	.asciz	"f1<_Atomic(int)>"              # string offset=6887
.Linfo_string306:
	.asciz	"_Z2f1IJilVcEEvv"               # string offset=6904
.Linfo_string307:
	.asciz	"_STN|f1|<int, long, volatile char>" # string offset=6920
.Linfo_string308:
	.asciz	"_Z2f1IJDv2_iEEvv"              # string offset=6955
.Linfo_string309:
	.asciz	"f1<__attribute__((__vector_size__(2 * sizeof(int)))) int>" # string offset=6972
.Linfo_string310:
	.asciz	"_Z2f1IJVKPiEEvv"               # string offset=7030
.Linfo_string311:
	.asciz	"_STN|f1|<int *const volatile>" # string offset=7046
.Linfo_string312:
	.asciz	"_Z2f1IJVKvEEvv"                # string offset=7076
.Linfo_string313:
	.asciz	"_STN|f1|<const volatile void>" # string offset=7091
.Linfo_string314:
	.asciz	"t1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:192:12)>" # string offset=7121
.Linfo_string315:
	.asciz	"_Z2f1IJ2t1IJZ4mainE3$_0EEEEvv" # string offset=7231
.Linfo_string316:
	.asciz	"f1<t1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:192:12)> >" # string offset=7261
.Linfo_string317:
	.asciz	"_ZN3t10C2IvEEv"                # string offset=7376
.Linfo_string318:
	.asciz	"_Z2f1IJM3udtKFvvEEEvv"         # string offset=7391
.Linfo_string319:
	.asciz	"_STN|f1|<void (udt::*)() const>" # string offset=7413
.Linfo_string320:
	.asciz	"_Z2f1IJM3udtVFvvREEEvv"        # string offset=7445
.Linfo_string321:
	.asciz	"_STN|f1|<void (udt::*)() volatile &>" # string offset=7468
.Linfo_string322:
	.asciz	"_Z2f1IJM3udtVKFvvOEEEvv"       # string offset=7505
.Linfo_string323:
	.asciz	"_STN|f1|<void (udt::*)() const volatile &&>" # string offset=7529
.Linfo_string324:
	.asciz	"_Z2f9IiEPFvvEv"                # string offset=7573
.Linfo_string325:
	.asciz	"_STN|f9|<int>"                 # string offset=7588
.Linfo_string326:
	.asciz	"_Z2f1IJKPFvvEEEvv"             # string offset=7602
.Linfo_string327:
	.asciz	"_STN|f1|<void (*const)()>"     # string offset=7620
.Linfo_string328:
	.asciz	"_Z2f1IJRA1_KcEEvv"             # string offset=7646
.Linfo_string329:
	.asciz	"_STN|f1|<const char (&)[1]>"   # string offset=7664
.Linfo_string330:
	.asciz	"_Z2f1IJKFvvREEEvv"             # string offset=7692
.Linfo_string331:
	.asciz	"_STN|f1|<void () const &>"     # string offset=7710
.Linfo_string332:
	.asciz	"_Z2f1IJVFvvOEEEvv"             # string offset=7736
.Linfo_string333:
	.asciz	"_STN|f1|<void () volatile &&>" # string offset=7754
.Linfo_string334:
	.asciz	"_Z2f1IJVKFvvEEEvv"             # string offset=7784
.Linfo_string335:
	.asciz	"_STN|f1|<void () const volatile>" # string offset=7802
.Linfo_string336:
	.asciz	"_Z2f1IJA1_KPiEEvv"             # string offset=7835
.Linfo_string337:
	.asciz	"_STN|f1|<int *const[1]>"       # string offset=7853
.Linfo_string338:
	.asciz	"_Z2f1IJRA1_KPiEEvv"            # string offset=7877
.Linfo_string339:
	.asciz	"_STN|f1|<int *const (&)[1]>"   # string offset=7896
.Linfo_string340:
	.asciz	"_Z2f1IJRKM3udtFvvEEEvv"        # string offset=7924
.Linfo_string341:
	.asciz	"_STN|f1|<void (udt::*const &)()>" # string offset=7947
.Linfo_string342:
	.asciz	"_Z2f1IJFPFvfEiEEEvv"           # string offset=7980
.Linfo_string343:
	.asciz	"_STN|f1|<void (*(int))(float)>" # string offset=8000
.Linfo_string344:
	.asciz	"_Z2f1IJA1_2t1IJiEEEEvv"        # string offset=8031
.Linfo_string345:
	.asciz	"_STN|f1|<t1<int>[1]>"          # string offset=8054
.Linfo_string346:
	.asciz	"_Z2f1IJPDoFvvEEEvv"            # string offset=8075
.Linfo_string347:
	.asciz	"f1<void (*)() noexcept>"       # string offset=8094
.Linfo_string348:
	.asciz	"_Z2f1IJFvZ4mainE3$_1EEEvv"     # string offset=8118
.Linfo_string349:
	.asciz	"f1<void ((unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:191:3))>" # string offset=8144
.Linfo_string350:
	.asciz	"_Z2f1IJFvZ4mainE2t8Z4mainE3$_1EEEvv" # string offset=8268
.Linfo_string351:
	.asciz	"f1<void (t8, (unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:191:3))>" # string offset=8304
.Linfo_string352:
	.asciz	"_Z2f1IJFvZ4mainE2t8EEEvv"      # string offset=8432
.Linfo_string353:
	.asciz	"_STN|f1|<void (t8)>"           # string offset=8457
.Linfo_string354:
	.asciz	"_Z19operator_not_reallyIiEvv"  # string offset=8477
.Linfo_string355:
	.asciz	"_STN|operator_not_really|<int>" # string offset=8506
.Linfo_string356:
	.asciz	"_BitInt(3)"                    # string offset=8537
.Linfo_string357:
	.asciz	"V"                             # string offset=8548
.Linfo_string358:
	.asciz	"_Z3f11IDB3_TnT_LS0_2EEvv"      # string offset=8550
.Linfo_string359:
	.asciz	"_STN|f11|<_BitInt(3), (_BitInt(3))2>" # string offset=8575
.Linfo_string360:
	.asciz	"unsigned _BitInt(5)"           # string offset=8612
.Linfo_string361:
	.asciz	"_Z3f11IKDU5_TnT_LS0_2EEvv"     # string offset=8632
.Linfo_string362:
	.asciz	"_STN|f11|<const unsigned _BitInt(5), (unsigned _BitInt(5))2>" # string offset=8658
.Linfo_string363:
	.asciz	"_BitInt(65)"                   # string offset=8719
.Linfo_string364:
	.asciz	"_Z3f11IDB65_TnT_LS0_2EEvv"     # string offset=8731
.Linfo_string365:
	.asciz	"f11<_BitInt(65), (_BitInt(65))2>" # string offset=8757
.Linfo_string366:
	.asciz	"unsigned _BitInt(65)"          # string offset=8790
.Linfo_string367:
	.asciz	"_Z3f11IKDU65_TnT_LS0_2EEvv"    # string offset=8811
.Linfo_string368:
	.asciz	"f11<const unsigned _BitInt(65), (unsigned _BitInt(65))2>" # string offset=8838
.Linfo_string369:
	.asciz	"_STN|t1|<>"                    # string offset=8895
.Linfo_string370:
	.asciz	"_Z2f1IJFv2t1IJEES1_EEEvv"      # string offset=8906
.Linfo_string371:
	.asciz	"_STN|f1|<void (t1<>, t1<>)>"   # string offset=8931
.Linfo_string372:
	.asciz	"_Z2f1IJM2t1IJEEiEEvv"          # string offset=8959
.Linfo_string373:
	.asciz	"_STN|f1|<int t1<>::*>"         # string offset=8980
.Linfo_string374:
	.asciz	"_Z2f1IJU9swiftcallFvvEEEvv"    # string offset=9002
.Linfo_string375:
	.asciz	"_STN|f1|<void () __attribute__((swiftcall))>" # string offset=9029
.Linfo_string376:
	.asciz	"_Z2f1IJFivEEEvv"               # string offset=9074
.Linfo_string377:
	.asciz	"f1<int () __attribute__((noreturn))>" # string offset=9090
.Linfo_string378:
	.asciz	"_Z3f10ILN2ns3$_0E0EEvv"        # string offset=9127
.Linfo_string379:
	.asciz	"f10<(ns::(unnamed enum at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:31:1))0>" # string offset=9150
.Linfo_string380:
	.asciz	"_Z2f1IJZN2t83memEvE2t7EEvv"    # string offset=9272
.Linfo_string381:
	.asciz	"_Z2f1IJM2t8FvvEEEvv"           # string offset=9299
.Linfo_string382:
	.asciz	"_STN|f1|<void (t8::*)()>"      # string offset=9319
.Linfo_string383:
	.asciz	"_ZN18complex_type_units2f1Ev"  # string offset=9344
.Linfo_string384:
	.asciz	"f1"                            # string offset=9373
.Linfo_string385:
	.asciz	"L"                             # string offset=9376
.Linfo_string386:
	.asciz	"v2"                            # string offset=9378
.Linfo_string387:
	.asciz	"N"                             # string offset=9381
.Linfo_string388:
	.asciz	"_STN|t4|<3U>"                  # string offset=9383
.Linfo_string389:
	.asciz	"v1"                            # string offset=9396
.Linfo_string390:
	.asciz	"v6"                            # string offset=9399
.Linfo_string391:
	.asciz	"x"                             # string offset=9402
.Linfo_string392:
	.asciz	"t7i"                           # string offset=9404
.Linfo_string393:
	.asciz	"v3"                            # string offset=9408
.Linfo_string394:
	.asciz	"v4"                            # string offset=9411
.Linfo_string395:
	.asciz	"t11<(anonymous namespace)::LocalEnum, ((anonymous namespace)::LocalEnum)0>" # string offset=9414
.Linfo_string396:
	.asciz	"t12"                           # string offset=9489
.Linfo_string397:
	.asciz	"_STN|t2|<int>"                 # string offset=9493
.Linfo_string398:
	.asciz	"_STN|t2|<float>"               # string offset=9507
.Linfo_string399:
	.asciz	"_STN|t1|<bool>"                # string offset=9523
.Linfo_string400:
	.asciz	"_STN|t2|<bool>"                # string offset=9538
.Linfo_string401:
	.asciz	"_STN|t1|<double>"              # string offset=9553
.Linfo_string402:
	.asciz	"_STN|t2|<double>"              # string offset=9570
.Linfo_string403:
	.asciz	"_STN|t1|<long>"                # string offset=9587
.Linfo_string404:
	.asciz	"_STN|t2|<long>"                # string offset=9602
.Linfo_string405:
	.asciz	"_STN|t1|<short>"               # string offset=9617
.Linfo_string406:
	.asciz	"_STN|t2|<short>"               # string offset=9633
.Linfo_string407:
	.asciz	"_STN|t1|<unsigned int>"        # string offset=9649
.Linfo_string408:
	.asciz	"_STN|t2|<unsigned int>"        # string offset=9672
.Linfo_string409:
	.asciz	"_STN|t1|<unsigned long long>"  # string offset=9695
.Linfo_string410:
	.asciz	"_STN|t2|<unsigned long long>"  # string offset=9724
.Linfo_string411:
	.asciz	"_STN|t1|<long long>"           # string offset=9753
.Linfo_string412:
	.asciz	"_STN|t2|<long long>"           # string offset=9773
.Linfo_string413:
	.asciz	"_STN|t1|<udt>"                 # string offset=9793
.Linfo_string414:
	.asciz	"_STN|t2|<udt>"                 # string offset=9807
.Linfo_string415:
	.asciz	"_STN|t1|<ns::udt>"             # string offset=9821
.Linfo_string416:
	.asciz	"_STN|t2|<ns::udt>"             # string offset=9839
.Linfo_string417:
	.asciz	"_STN|t1|<ns::udt *>"           # string offset=9857
.Linfo_string418:
	.asciz	"_STN|t2|<ns::udt *>"           # string offset=9877
.Linfo_string419:
	.asciz	"_STN|t1|<ns::inner::udt>"      # string offset=9897
.Linfo_string420:
	.asciz	"_STN|t2|<ns::inner::udt>"      # string offset=9922
.Linfo_string421:
	.asciz	"_STN|t1|<t1<int> >"            # string offset=9947
.Linfo_string422:
	.asciz	"_STN|t2|<t1<int> >"            # string offset=9966
.Linfo_string423:
	.asciz	"_STN|t1|<int, float>"          # string offset=9985
.Linfo_string424:
	.asciz	"_STN|t2|<int, float>"          # string offset=10006
.Linfo_string425:
	.asciz	"_STN|t2|<int *>"               # string offset=10027
.Linfo_string426:
	.asciz	"_STN|t1|<int &>"               # string offset=10043
.Linfo_string427:
	.asciz	"_STN|t2|<int &>"               # string offset=10059
.Linfo_string428:
	.asciz	"_STN|t1|<int &&>"              # string offset=10075
.Linfo_string429:
	.asciz	"_STN|t2|<int &&>"              # string offset=10092
.Linfo_string430:
	.asciz	"_STN|t1|<const int>"           # string offset=10109
.Linfo_string431:
	.asciz	"_STN|t2|<const int>"           # string offset=10129
.Linfo_string432:
	.asciz	"_STN|t1|<int[3]>"              # string offset=10149
.Linfo_string433:
	.asciz	"_STN|t2|<int[3]>"              # string offset=10166
.Linfo_string434:
	.asciz	"_STN|t1|<void>"                # string offset=10183
.Linfo_string435:
	.asciz	"_STN|t2|<void>"                # string offset=10198
.Linfo_string436:
	.asciz	"_STN|t1|<outer_class::inner_class>" # string offset=10213
.Linfo_string437:
	.asciz	"_STN|t2|<outer_class::inner_class>" # string offset=10248
.Linfo_string438:
	.asciz	"_STN|t1|<unsigned long>"       # string offset=10283
.Linfo_string439:
	.asciz	"_STN|t2|<unsigned long>"       # string offset=10307
.Linfo_string440:
	.asciz	"_STN|t1|<t3<int, false> >"     # string offset=10331
.Linfo_string441:
	.asciz	"_STN|t2|<t3<int, false> >"     # string offset=10357
.Linfo_string442:
	.asciz	"_STN|t1|<t3<t3<int, false>, false> >" # string offset=10383
.Linfo_string443:
	.asciz	"_STN|t2|<t3<t3<int, false>, false> >" # string offset=10420
.Linfo_string444:
	.asciz	"t2<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:192:12)>" # string offset=10457
.Linfo_string445:
	.asciz	"t1<t3<t3<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:192:12), false>, false> >" # string offset=10567
.Linfo_string446:
	.asciz	"t2<t3<t3<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:192:12), false>, false> >" # string offset=10700
.Linfo_string447:
	.asciz	"_STN|t1|<int (float)>"         # string offset=10833
.Linfo_string448:
	.asciz	"_STN|t2|<int (float)>"         # string offset=10855
.Linfo_string449:
	.asciz	"_STN|t1|<void (...)>"          # string offset=10877
.Linfo_string450:
	.asciz	"_STN|t2|<void (...)>"          # string offset=10898
.Linfo_string451:
	.asciz	"_STN|t1|<void (int, ...)>"     # string offset=10919
.Linfo_string452:
	.asciz	"_STN|t2|<void (int, ...)>"     # string offset=10945
.Linfo_string453:
	.asciz	"_STN|t1|<const int &>"         # string offset=10971
.Linfo_string454:
	.asciz	"_STN|t2|<const int &>"         # string offset=10993
.Linfo_string455:
	.asciz	"_STN|t1|<const int *&>"        # string offset=11015
.Linfo_string456:
	.asciz	"_STN|t2|<const int *&>"        # string offset=11038
.Linfo_string457:
	.asciz	"_STN|t1|<(anonymous namespace)::t5>" # string offset=11061
.Linfo_string458:
	.asciz	"_STN|t2|<(anonymous namespace)::t5>" # string offset=11097
.Linfo_string459:
	.asciz	"_STN|t1|<std::nullptr_t>"      # string offset=11133
.Linfo_string460:
	.asciz	"_STN|t2|<std::nullptr_t>"      # string offset=11158
.Linfo_string461:
	.asciz	"_STN|t1|<long *, long *>"      # string offset=11183
.Linfo_string462:
	.asciz	"_STN|t2|<long *, long *>"      # string offset=11208
.Linfo_string463:
	.asciz	"_STN|t1|<long *, udt *>"       # string offset=11233
.Linfo_string464:
	.asciz	"_STN|t2|<long *, udt *>"       # string offset=11257
.Linfo_string465:
	.asciz	"_STN|t1|<void *const>"         # string offset=11281
.Linfo_string466:
	.asciz	"_STN|t2|<void *const>"         # string offset=11303
.Linfo_string467:
	.asciz	"_STN|t1|<const void *const *>" # string offset=11325
.Linfo_string468:
	.asciz	"_STN|t2|<const void *const *>" # string offset=11355
.Linfo_string469:
	.asciz	"_STN|t1|<void ()>"             # string offset=11385
.Linfo_string470:
	.asciz	"_STN|t2|<void ()>"             # string offset=11403
.Linfo_string471:
	.asciz	"_STN|t1|<void (*)()>"          # string offset=11421
.Linfo_string472:
	.asciz	"_STN|t2|<void (*)()>"          # string offset=11442
.Linfo_string473:
	.asciz	"t1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:192:12) *>" # string offset=11463
.Linfo_string474:
	.asciz	"t2<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:192:12) *>" # string offset=11575
.Linfo_string475:
	.asciz	"t1<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:191:3)>" # string offset=11687
.Linfo_string476:
	.asciz	"t2<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:191:3)>" # string offset=11804
.Linfo_string477:
	.asciz	"t1<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:191:3) *>" # string offset=11921
.Linfo_string478:
	.asciz	"t2<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:191:3) *>" # string offset=12040
.Linfo_string479:
	.asciz	"_STN|t2|<>"                    # string offset=12159
.Linfo_string480:
	.asciz	"_STN|t1|<const void *, const void *>" # string offset=12170
.Linfo_string481:
	.asciz	"_STN|t2|<const void *, const void *>" # string offset=12207
.Linfo_string482:
	.asciz	"_STN|t1|<t1<int *> *>"         # string offset=12244
.Linfo_string483:
	.asciz	"_STN|t2|<t1<int *> *>"         # string offset=12266
.Linfo_string484:
	.asciz	"_STN|t1|<int *[]>"             # string offset=12288
.Linfo_string485:
	.asciz	"_STN|t2|<int *[]>"             # string offset=12306
.Linfo_string486:
	.asciz	"this"                          # string offset=12324
.Linfo_string487:
	.asciz	"_STN|t1|<t7>"                  # string offset=12329
.Linfo_string488:
	.asciz	"_STN|t2|<t7>"                  # string offset=12342
.Linfo_string489:
	.asciz	"_STN|t1|<int (&)[3]>"          # string offset=12355
.Linfo_string490:
	.asciz	"_STN|t2|<int (&)[3]>"          # string offset=12376
.Linfo_string491:
	.asciz	"_STN|t1|<int (*)[3]>"          # string offset=12397
.Linfo_string492:
	.asciz	"_STN|t2|<int (*)[3]>"          # string offset=12418
.Linfo_string493:
	.asciz	"_STN|t1|<int *, std::nullptr_t *>" # string offset=12439
.Linfo_string494:
	.asciz	"_STN|t2|<int *, std::nullptr_t *>" # string offset=12473
.Linfo_string495:
	.asciz	"_STN|t1|<t7<int> >"            # string offset=12507
.Linfo_string496:
	.asciz	"_STN|t2|<t7<int> >"            # string offset=12526
.Linfo_string497:
	.asciz	"t1<_Atomic(int)>"              # string offset=12545
.Linfo_string498:
	.asciz	"t2<_Atomic(int)>"              # string offset=12562
.Linfo_string499:
	.asciz	"_STN|t1|<int, long, volatile char>" # string offset=12579
.Linfo_string500:
	.asciz	"_STN|t2|<int, long, volatile char>" # string offset=12614
.Linfo_string501:
	.asciz	"t1<__attribute__((__vector_size__(2 * sizeof(int)))) int>" # string offset=12649
.Linfo_string502:
	.asciz	"t2<__attribute__((__vector_size__(2 * sizeof(int)))) int>" # string offset=12707
.Linfo_string503:
	.asciz	"_STN|t1|<int *const volatile>" # string offset=12765
.Linfo_string504:
	.asciz	"_STN|t2|<int *const volatile>" # string offset=12795
.Linfo_string505:
	.asciz	"_STN|t1|<const volatile void>" # string offset=12825
.Linfo_string506:
	.asciz	"_STN|t2|<const volatile void>" # string offset=12855
.Linfo_string507:
	.asciz	"t1<t1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:192:12)> >" # string offset=12885
.Linfo_string508:
	.asciz	"t2<t1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:192:12)> >" # string offset=13000
.Linfo_string509:
	.asciz	"_STN|t1|<void (udt::*)() const>" # string offset=13115
.Linfo_string510:
	.asciz	"_STN|t2|<void (udt::*)() const>" # string offset=13147
.Linfo_string511:
	.asciz	"_STN|t1|<void (udt::*)() volatile &>" # string offset=13179
.Linfo_string512:
	.asciz	"_STN|t2|<void (udt::*)() volatile &>" # string offset=13216
.Linfo_string513:
	.asciz	"_STN|t1|<void (udt::*)() const volatile &&>" # string offset=13253
.Linfo_string514:
	.asciz	"_STN|t2|<void (udt::*)() const volatile &&>" # string offset=13297
.Linfo_string515:
	.asciz	"_STN|t1|<void (*const)()>"     # string offset=13341
.Linfo_string516:
	.asciz	"_STN|t2|<void (*const)()>"     # string offset=13367
.Linfo_string517:
	.asciz	"_STN|t1|<const char (&)[1]>"   # string offset=13393
.Linfo_string518:
	.asciz	"_STN|t2|<const char (&)[1]>"   # string offset=13421
.Linfo_string519:
	.asciz	"_STN|t1|<void () const &>"     # string offset=13449
.Linfo_string520:
	.asciz	"_STN|t2|<void () const &>"     # string offset=13475
.Linfo_string521:
	.asciz	"_STN|t1|<void () volatile &&>" # string offset=13501
.Linfo_string522:
	.asciz	"_STN|t2|<void () volatile &&>" # string offset=13531
.Linfo_string523:
	.asciz	"_STN|t1|<void () const volatile>" # string offset=13561
.Linfo_string524:
	.asciz	"_STN|t2|<void () const volatile>" # string offset=13594
.Linfo_string525:
	.asciz	"_STN|t1|<int *const[1]>"       # string offset=13627
.Linfo_string526:
	.asciz	"_STN|t2|<int *const[1]>"       # string offset=13651
.Linfo_string527:
	.asciz	"_STN|t1|<int *const (&)[1]>"   # string offset=13675
.Linfo_string528:
	.asciz	"_STN|t2|<int *const (&)[1]>"   # string offset=13703
.Linfo_string529:
	.asciz	"_STN|t1|<void (udt::*const &)()>" # string offset=13731
.Linfo_string530:
	.asciz	"_STN|t2|<void (udt::*const &)()>" # string offset=13764
.Linfo_string531:
	.asciz	"_STN|t1|<void (*(int))(float)>" # string offset=13797
.Linfo_string532:
	.asciz	"_STN|t2|<void (*(int))(float)>" # string offset=13828
.Linfo_string533:
	.asciz	"_STN|t1|<t1<int>[1]>"          # string offset=13859
.Linfo_string534:
	.asciz	"_STN|t2|<t1<int>[1]>"          # string offset=13880
.Linfo_string535:
	.asciz	"t1<void (*)() noexcept>"       # string offset=13901
.Linfo_string536:
	.asciz	"t2<void (*)() noexcept>"       # string offset=13925
.Linfo_string537:
	.asciz	"t1<void ((unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:191:3))>" # string offset=13949
.Linfo_string538:
	.asciz	"t2<void ((unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:191:3))>" # string offset=14073
.Linfo_string539:
	.asciz	"t1<void (t8, (unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:191:3))>" # string offset=14197
.Linfo_string540:
	.asciz	"t2<void (t8, (unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:191:3))>" # string offset=14325
.Linfo_string541:
	.asciz	"_STN|t1|<void (t8)>"           # string offset=14453
.Linfo_string542:
	.asciz	"_STN|t2|<void (t8)>"           # string offset=14473
.Linfo_string543:
	.asciz	"_STN|t1|<void (t1<>, t1<>)>"   # string offset=14493
.Linfo_string544:
	.asciz	"_STN|t2|<void (t1<>, t1<>)>"   # string offset=14521
.Linfo_string545:
	.asciz	"_STN|t1|<int t1<>::*>"         # string offset=14549
.Linfo_string546:
	.asciz	"_STN|t2|<int t1<>::*>"         # string offset=14571
.Linfo_string547:
	.asciz	"_STN|t1|<void () __attribute__((swiftcall))>" # string offset=14593
.Linfo_string548:
	.asciz	"_STN|t2|<void () __attribute__((swiftcall))>" # string offset=14638
.Linfo_string549:
	.asciz	"t1<int () __attribute__((noreturn))>" # string offset=14683
.Linfo_string550:
	.asciz	"t2<int () __attribute__((noreturn))>" # string offset=14720
.Linfo_string551:
	.asciz	"_STN|t1|<void (t8::*)()>"      # string offset=14757
.Linfo_string552:
	.asciz	"_STN|t2|<void (t8::*)()>"      # string offset=14782
.Linfo_string553:
	.asciz	"internal_type"                 # string offset=14807
.Linfo_string554:
	.asciz	"t2<&complex_type_units::external_function>" # string offset=14821
.Linfo_string555:
	.asciz	"_STN|t3|<complex_type_units::t2<&complex_type_units::external_function> >" # string offset=14864
.Linfo_string556:
	.asciz	"_STN|t4|<complex_type_units::(anonymous namespace)::internal_type, complex_type_units::t3<complex_type_units::t2<&complex_type_units::external_function> > >" # string offset=14938
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
	.quad	_ZN18complex_type_units17external_functionEv
.Ldebug_addr_end0:
	.ident	"clang version 22.0.0git (git@github.com:Michael137/llvm-project.git f45bb984e6f21e702b4d65f1eeea1429f43c800e)"
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
	.section	.debug_line,"",@progbits
.Lline_table_start0:
