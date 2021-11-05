# RUN: llvm-mc < %s -filetype obj -triple x86_64 -o - \
# RUN:   | llvm-dwarfdump --verify - | FileCheck %s

# Checking the LLVM side of cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp
# Compile that file with `-g -Xclang -gsimple-template-names=mangled -Xclang -debug-forward-template-params -S`
# to (re)generate this assembly file - while it might be slightly overkill in
# some ways, it seems small/simple enough to keep this as an exact match for
# that end to end test.

# CHECK: No errors.
	.text
	.file	"simplified_template_names.cpp"
	.file	1 "./" "cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp"
	.file	2 "/usr/include/x86_64-linux-gnu/bits" "types.h"
	.file	3 "/usr/include/x86_64-linux-gnu/bits" "stdint-intn.h"
	.file	4 "/usr/local/google/home/blaikie/install/bin/../lib/gcc/x86_64-pc-linux-gnu/10.0.0/../../../../include/c++/10.0.0" "cstdint"
	.file	5 "/usr/include" "stdint.h"
	.file	6 "/usr/include/x86_64-linux-gnu/bits" "stdint-uintn.h"
	.globl	_Zli5_suffy                     # -- Begin function _Zli5_suffy
	.p2align	4, 0x90
	.type	_Zli5_suffy,@function
_Zli5_suffy:                            # @_Zli5_suffy
.Lfunc_begin0:
	.loc	1 134 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:134:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp0:
	.loc	1 134 44 prologue_end           # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:134:44
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
	.loc	1 166 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:166:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
.Ltmp2:
	.loc	1 168 8 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:168:8
	movb	.L__const.main.L, %al
	movb	%al, -16(%rbp)
	.loc	1 169 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:169:3
	callq	_Z2f1IJiEEvv
	.loc	1 170 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:170:3
	callq	_Z2f1IJfEEvv
	.loc	1 171 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:171:3
	callq	_Z2f1IJbEEvv
	.loc	1 172 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:172:3
	callq	_Z2f1IJdEEvv
	.loc	1 173 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:173:3
	callq	_Z2f1IJlEEvv
	.loc	1 174 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:174:3
	callq	_Z2f1IJsEEvv
	.loc	1 175 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:175:3
	callq	_Z2f1IJjEEvv
	.loc	1 176 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:176:3
	callq	_Z2f1IJyEEvv
	.loc	1 177 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:177:3
	callq	_Z2f1IJxEEvv
	.loc	1 178 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:178:3
	callq	_Z2f1IJ3udtEEvv
	.loc	1 179 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:179:3
	callq	_Z2f1IJN2ns3udtEEEvv
	.loc	1 180 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:180:3
	callq	_Z2f1IJPN2ns3udtEEEvv
	.loc	1 181 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:181:3
	callq	_Z2f1IJN2ns5inner3udtEEEvv
	.loc	1 182 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:182:3
	callq	_Z2f1IJ2t1IJiEEEEvv
	.loc	1 183 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:183:3
	callq	_Z2f1IJifEEvv
	.loc	1 184 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:184:3
	callq	_Z2f1IJPiEEvv
	.loc	1 185 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:185:3
	callq	_Z2f1IJRiEEvv
	.loc	1 186 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:186:3
	callq	_Z2f1IJOiEEvv
	.loc	1 187 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:187:3
	callq	_Z2f1IJKiEEvv
	.loc	1 189 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:189:3
	callq	_Z2f1IJvEEvv
	.loc	1 190 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:190:3
	callq	_Z2f1IJN11outer_class11inner_classEEEvv
	.loc	1 191 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:191:3
	callq	_Z2f1IJmEEvv
	.loc	1 192 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:192:3
	callq	_Z2f2ILb1ELi3EEvv
	.loc	1 193 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:193:3
	callq	_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv
	.loc	1 194 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:194:3
	callq	_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv
	.loc	1 195 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:195:3
	callq	_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv
	.loc	1 196 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:196:3
	callq	_Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv
	.loc	1 197 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:197:3
	callq	_Z2f3IPiJXadL_Z1iEEEEvv
	.loc	1 198 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:198:3
	callq	_Z2f3IPiJLS0_0EEEvv
	.loc	1 200 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:200:3
	callq	_Z2f3ImJLm1EEEvv
	.loc	1 201 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:201:3
	callq	_Z2f3IyJLy1EEEvv
	.loc	1 202 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:202:3
	callq	_Z2f3IlJLl1EEEvv
	.loc	1 203 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:203:3
	callq	_Z2f3IjJLj1EEEvv
	.loc	1 204 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:204:3
	callq	_Z2f3IsJLs1EEEvv
	.loc	1 205 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:205:3
	callq	_Z2f3IhJLh0EEEvv
	.loc	1 206 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:206:3
	callq	_Z2f3IaJLa0EEEvv
	.loc	1 207 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:207:3
	callq	_Z2f3ItJLt1ELt2EEEvv
	.loc	1 208 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:208:3
	callq	_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv
	.loc	1 209 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:209:3
	callq	_Z2f3InJLn18446744073709551614EEEvv
	.loc	1 210 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:210:3
	callq	_Z2f4IjLj3EEvv
	.loc	1 211 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:211:3
	callq	_Z2f1IJ2t3IiLb0EEEEvv
	.loc	1 212 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:212:3
	callq	_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv
	.loc	1 213 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:213:3
	callq	_Z2f1IJZ4mainE3$_1EEvv
	.loc	1 215 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:215:3
	callq	_Z2f1IJFifEEEvv
	.loc	1 216 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:216:3
	callq	_Z2f1IJRKiEEvv
	.loc	1 217 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:217:3
	callq	_Z2f1IJRPKiEEvv
	.loc	1 218 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:218:3
	callq	_Z2f1IJN12_GLOBAL__N_12t5EEEvv
	.loc	1 219 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:219:3
	callq	_Z2f1IJDnEEvv
	.loc	1 220 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:220:3
	callq	_Z2f1IJPlS0_EEvv
	.loc	1 221 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:221:3
	callq	_Z2f1IJPlP3udtEEvv
	.loc	1 222 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:222:3
	callq	_Z2f1IJKPvEEvv
	.loc	1 223 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:223:3
	callq	_Z2f1IJPKPKvEEvv
	.loc	1 224 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:224:3
	callq	_Z2f1IJFvvEEEvv
	.loc	1 225 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:225:3
	callq	_Z2f1IJPFvvEEEvv
	.loc	1 226 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:226:3
	callq	_Z2f1IJPZ4mainE3$_1EEvv
	.loc	1 227 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:227:3
	callq	_Z2f1IJZ4mainE3$_2EEvv
	.loc	1 228 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:228:3
	callq	_Z2f1IJPZ4mainE3$_2EEvv
	.loc	1 229 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:229:3
	callq	_Z2f5IJ2t1IJiEEEiEvv
	.loc	1 230 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:230:3
	callq	_Z2f5IJEiEvv
	.loc	1 231 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:231:3
	callq	_Z2f6I2t1IJiEEJEEvv
	.loc	1 232 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:232:3
	callq	_Z2f1IJEEvv
	.loc	1 233 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:233:3
	callq	_Z2f1IJPKvS1_EEvv
	.loc	1 234 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:234:3
	callq	_Z2f1IJP2t1IJPiEEEEvv
	.loc	1 235 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:235:3
	callq	_Z2f1IJA_PiEEvv
	.loc	1 237 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:237:6
	leaq	-40(%rbp), %rdi
	movl	$1, %esi
	callq	_ZN2t6lsIiEEvi
	.loc	1 238 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:238:6
	leaq	-40(%rbp), %rdi
	movl	$1, %esi
	callq	_ZN2t6ltIiEEvi
	.loc	1 239 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:239:6
	leaq	-40(%rbp), %rdi
	movl	$1, %esi
	callq	_ZN2t6leIiEEvi
	.loc	1 240 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:240:6
	leaq	-40(%rbp), %rdi
	callq	_ZN2t6cvP2t1IJfEEIiEEv
	.loc	1 241 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:241:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6miIiEEvi
	.loc	1 242 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:242:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6mlIiEEvi
	.loc	1 243 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:243:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6dvIiEEvi
	.loc	1 244 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:244:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6rmIiEEvi
	.loc	1 245 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:245:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6eoIiEEvi
	.loc	1 246 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:246:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6anIiEEvi
	.loc	1 247 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:247:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6orIiEEvi
	.loc	1 248 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:248:6
	leaq	-40(%rbp), %rdi
	callq	_ZN2t6coIiEEvv
	.loc	1 249 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:249:6
	leaq	-40(%rbp), %rdi
	callq	_ZN2t6ntIiEEvv
	.loc	1 250 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:250:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6aSIiEEvi
	.loc	1 251 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:251:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6gtIiEEvi
	.loc	1 252 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:252:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6cmIiEEvi
	.loc	1 253 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:253:6
	leaq	-40(%rbp), %rdi
	callq	_ZN2t6clIiEEvv
	.loc	1 254 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:254:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6ixIiEEvi
	.loc	1 255 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:255:6
	leaq	-40(%rbp), %rdi
	movl	$3, %esi
	callq	_ZN2t6ssIiEEvi
	.loc	1 256 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:256:3
	xorl	%eax, %eax
	movl	%eax, %edi
	xorl	%esi, %esi
	callq	_ZN2t6nwIiEEPvmT_
	.loc	1 257 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:257:3
	xorl	%eax, %eax
	movl	%eax, %edi
	xorl	%esi, %esi
	callq	_ZN2t6naIiEEPvmT_
	.loc	1 258 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:258:3
	xorl	%eax, %eax
	movl	%eax, %edi
	xorl	%esi, %esi
	callq	_ZN2t6dlIiEEvPvT_
	.loc	1 259 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:259:3
	xorl	%eax, %eax
	movl	%eax, %edi
	xorl	%esi, %esi
	callq	_ZN2t6daIiEEvPvT_
	.loc	1 260 6                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:260:6
	leaq	-40(%rbp), %rdi
	callq	_ZN2t6awIiEEiv
	.loc	1 261 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:261:3
	movl	$42, %edi
	callq	_Zli5_suffy
	.loc	1 263 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:263:3
	callq	_Z2f1IJZ4mainE2t7EEvv
	.loc	1 264 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:264:3
	callq	_Z2f1IJRA3_iEEvv
	.loc	1 265 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:265:3
	callq	_Z2f1IJPA3_iEEvv
	.loc	1 266 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:266:3
	callq	_Z2f7I2t1Evv
	.loc	1 267 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:267:3
	callq	_Z2f8I2t1iEvv
	.loc	1 269 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:269:3
	callq	_ZN2ns8ttp_userINS_5inner3ttpEEEvv
	.loc	1 270 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:270:3
	callq	_Z2f1IJPiPDnEEvv
	.loc	1 272 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:272:3
	callq	_Z2f1IJ2t7IiEEEvv
	.loc	1 273 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:273:3
	callq	_Z2f7IN2ns3inl2t9EEvv
	.loc	1 274 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:274:3
	callq	_Z2f1IJU7_AtomiciEEvv
	.loc	1 275 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:275:3
	callq	_Z2f1IJilVcEEvv
	.loc	1 276 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:276:3
	callq	_Z2f1IJDv2_iEEvv
	.loc	1 277 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:277:3
	callq	_Z2f1IJVKPiEEvv
	.loc	1 278 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:278:3
	callq	_Z2f1IJVKvEEvv
	.loc	1 279 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:279:3
	callq	_Z2f1IJ2t1IJZ4mainE3$_1EEEEvv
	.loc	1 280 7                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:280:7
	leaq	-56(%rbp), %rdi
	callq	_ZN3t10C2IvEEv
	.loc	1 281 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:281:3
	callq	_Z2f1IJM3udtKFvvEEEvv
	.loc	1 282 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:282:3
	callq	_Z2f1IJM3udtVFvvREEEvv
	.loc	1 283 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:283:3
	callq	_Z2f1IJM3udtVKFvvOEEEvv
	.loc	1 284 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:284:3
	callq	_Z2f9IiEPFvvEv
	.loc	1 285 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:285:3
	callq	_Z2f1IJKPFvvEEEvv
	.loc	1 286 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:286:3
	callq	_Z2f1IJRA1_KcEEvv
	.loc	1 287 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:287:3
	callq	_Z2f1IJKFvvREEEvv
	.loc	1 288 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:288:3
	callq	_Z2f1IJVFvvOEEEvv
	.loc	1 289 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:289:3
	callq	_Z2f1IJVKFvvEEEvv
	.loc	1 290 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:290:3
	callq	_Z2f1IJA1_KPiEEvv
	.loc	1 291 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:291:3
	callq	_Z2f1IJRA1_KPiEEvv
	.loc	1 292 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:292:3
	callq	_Z2f1IJRKM3udtFvvEEEvv
	.loc	1 293 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:293:3
	callq	_Z2f1IJFPFvfEiEEEvv
	.loc	1 295 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:295:3
	callq	_Z2f1IJPDoFvvEEEvv
	.loc	1 296 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:296:3
	callq	_Z2f1IJFvZ4mainE3$_2EEEvv
	.loc	1 298 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:298:3
	callq	_Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv
	.loc	1 299 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:299:3
	callq	_Z2f1IJFvZ4mainE2t8EEEvv
	.loc	1 300 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:300:3
	callq	_Z19operator_not_reallyIiEvv
	.loc	1 301 1                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:301:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp4:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp6:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp8:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp10:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp12:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp14:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp16:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp18:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp20:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp22:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp24:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp26:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp28:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp30:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp32:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp34:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp36:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp38:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
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
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp40:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp41:
.Lfunc_end20:
	.size	_Z2f1IJKiEEvv, .Lfunc_end20-_Z2f1IJKiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJvEEvv,"axG",@progbits,_Z2f1IJvEEvv,comdat
	.weak	_Z2f1IJvEEvv                    # -- Begin function _Z2f1IJvEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJvEEvv,@function
_Z2f1IJvEEvv:                           # @_Z2f1IJvEEvv
.Lfunc_begin21:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp42:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp43:
.Lfunc_end21:
	.size	_Z2f1IJvEEvv, .Lfunc_end21-_Z2f1IJvEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJN11outer_class11inner_classEEEvv,"axG",@progbits,_Z2f1IJN11outer_class11inner_classEEEvv,comdat
	.weak	_Z2f1IJN11outer_class11inner_classEEEvv # -- Begin function _Z2f1IJN11outer_class11inner_classEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJN11outer_class11inner_classEEEvv,@function
_Z2f1IJN11outer_class11inner_classEEEvv: # @_Z2f1IJN11outer_class11inner_classEEEvv
.Lfunc_begin22:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp44:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp45:
.Lfunc_end22:
	.size	_Z2f1IJN11outer_class11inner_classEEEvv, .Lfunc_end22-_Z2f1IJN11outer_class11inner_classEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJmEEvv,"axG",@progbits,_Z2f1IJmEEvv,comdat
	.weak	_Z2f1IJmEEvv                    # -- Begin function _Z2f1IJmEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJmEEvv,@function
_Z2f1IJmEEvv:                           # @_Z2f1IJmEEvv
.Lfunc_begin23:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp46:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp47:
.Lfunc_end23:
	.size	_Z2f1IJmEEvv, .Lfunc_end23-_Z2f1IJmEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f2ILb1ELi3EEvv,"axG",@progbits,_Z2f2ILb1ELi3EEvv,comdat
	.weak	_Z2f2ILb1ELi3EEvv               # -- Begin function _Z2f2ILb1ELi3EEvv
	.p2align	4, 0x90
	.type	_Z2f2ILb1ELi3EEvv,@function
_Z2f2ILb1ELi3EEvv:                      # @_Z2f2ILb1ELi3EEvv
.Lfunc_begin24:
	.loc	1 31 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:31:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp48:
	.loc	1 32 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:32:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp49:
.Lfunc_end24:
	.size	_Z2f2ILb1ELi3EEvv, .Lfunc_end24-_Z2f2ILb1ELi3EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv,"axG",@progbits,_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv,comdat
	.weak	_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv # -- Begin function _Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv,@function
_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv: # @_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv
.Lfunc_begin25:
	.loc	1 34 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:34:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp50:
	.loc	1 35 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp51:
.Lfunc_end25:
	.size	_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv, .Lfunc_end25-_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv,"axG",@progbits,_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv,comdat
	.weak	_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv # -- Begin function _Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv,@function
_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv: # @_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv
.Lfunc_begin26:
	.loc	1 34 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:34:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp52:
	.loc	1 35 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp53:
.Lfunc_end26:
	.size	_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv, .Lfunc_end26-_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv,"axG",@progbits,_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv,comdat
	.weak	_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv # -- Begin function _Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv,@function
_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv: # @_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv
.Lfunc_begin27:
	.loc	1 34 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:34:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp54:
	.loc	1 35 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp55:
.Lfunc_end27:
	.size	_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv, .Lfunc_end27-_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function _Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv
	.type	_Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv,@function
_Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv:       # @"_Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv"
.Lfunc_begin28:
	.loc	1 34 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:34:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp56:
	.loc	1 35 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp57:
.Lfunc_end28:
	.size	_Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv, .Lfunc_end28-_Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IPiJXadL_Z1iEEEEvv,"axG",@progbits,_Z2f3IPiJXadL_Z1iEEEEvv,comdat
	.weak	_Z2f3IPiJXadL_Z1iEEEEvv         # -- Begin function _Z2f3IPiJXadL_Z1iEEEEvv
	.p2align	4, 0x90
	.type	_Z2f3IPiJXadL_Z1iEEEEvv,@function
_Z2f3IPiJXadL_Z1iEEEEvv:                # @_Z2f3IPiJXadL_Z1iEEEEvv
.Lfunc_begin29:
	.loc	1 34 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:34:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp58:
	.loc	1 35 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp59:
.Lfunc_end29:
	.size	_Z2f3IPiJXadL_Z1iEEEEvv, .Lfunc_end29-_Z2f3IPiJXadL_Z1iEEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IPiJLS0_0EEEvv,"axG",@progbits,_Z2f3IPiJLS0_0EEEvv,comdat
	.weak	_Z2f3IPiJLS0_0EEEvv             # -- Begin function _Z2f3IPiJLS0_0EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IPiJLS0_0EEEvv,@function
_Z2f3IPiJLS0_0EEEvv:                    # @_Z2f3IPiJLS0_0EEEvv
.Lfunc_begin30:
	.loc	1 34 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:34:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp60:
	.loc	1 35 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp61:
.Lfunc_end30:
	.size	_Z2f3IPiJLS0_0EEEvv, .Lfunc_end30-_Z2f3IPiJLS0_0EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3ImJLm1EEEvv,"axG",@progbits,_Z2f3ImJLm1EEEvv,comdat
	.weak	_Z2f3ImJLm1EEEvv                # -- Begin function _Z2f3ImJLm1EEEvv
	.p2align	4, 0x90
	.type	_Z2f3ImJLm1EEEvv,@function
_Z2f3ImJLm1EEEvv:                       # @_Z2f3ImJLm1EEEvv
.Lfunc_begin31:
	.loc	1 34 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:34:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp62:
	.loc	1 35 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp63:
.Lfunc_end31:
	.size	_Z2f3ImJLm1EEEvv, .Lfunc_end31-_Z2f3ImJLm1EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IyJLy1EEEvv,"axG",@progbits,_Z2f3IyJLy1EEEvv,comdat
	.weak	_Z2f3IyJLy1EEEvv                # -- Begin function _Z2f3IyJLy1EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IyJLy1EEEvv,@function
_Z2f3IyJLy1EEEvv:                       # @_Z2f3IyJLy1EEEvv
.Lfunc_begin32:
	.loc	1 34 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:34:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp64:
	.loc	1 35 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp65:
.Lfunc_end32:
	.size	_Z2f3IyJLy1EEEvv, .Lfunc_end32-_Z2f3IyJLy1EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IlJLl1EEEvv,"axG",@progbits,_Z2f3IlJLl1EEEvv,comdat
	.weak	_Z2f3IlJLl1EEEvv                # -- Begin function _Z2f3IlJLl1EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IlJLl1EEEvv,@function
_Z2f3IlJLl1EEEvv:                       # @_Z2f3IlJLl1EEEvv
.Lfunc_begin33:
	.loc	1 34 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:34:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp66:
	.loc	1 35 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp67:
.Lfunc_end33:
	.size	_Z2f3IlJLl1EEEvv, .Lfunc_end33-_Z2f3IlJLl1EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IjJLj1EEEvv,"axG",@progbits,_Z2f3IjJLj1EEEvv,comdat
	.weak	_Z2f3IjJLj1EEEvv                # -- Begin function _Z2f3IjJLj1EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IjJLj1EEEvv,@function
_Z2f3IjJLj1EEEvv:                       # @_Z2f3IjJLj1EEEvv
.Lfunc_begin34:
	.loc	1 34 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:34:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp68:
	.loc	1 35 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp69:
.Lfunc_end34:
	.size	_Z2f3IjJLj1EEEvv, .Lfunc_end34-_Z2f3IjJLj1EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IsJLs1EEEvv,"axG",@progbits,_Z2f3IsJLs1EEEvv,comdat
	.weak	_Z2f3IsJLs1EEEvv                # -- Begin function _Z2f3IsJLs1EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IsJLs1EEEvv,@function
_Z2f3IsJLs1EEEvv:                       # @_Z2f3IsJLs1EEEvv
.Lfunc_begin35:
	.loc	1 34 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:34:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp70:
	.loc	1 35 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp71:
.Lfunc_end35:
	.size	_Z2f3IsJLs1EEEvv, .Lfunc_end35-_Z2f3IsJLs1EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IhJLh0EEEvv,"axG",@progbits,_Z2f3IhJLh0EEEvv,comdat
	.weak	_Z2f3IhJLh0EEEvv                # -- Begin function _Z2f3IhJLh0EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IhJLh0EEEvv,@function
_Z2f3IhJLh0EEEvv:                       # @_Z2f3IhJLh0EEEvv
.Lfunc_begin36:
	.loc	1 34 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:34:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp72:
	.loc	1 35 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp73:
.Lfunc_end36:
	.size	_Z2f3IhJLh0EEEvv, .Lfunc_end36-_Z2f3IhJLh0EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IaJLa0EEEvv,"axG",@progbits,_Z2f3IaJLa0EEEvv,comdat
	.weak	_Z2f3IaJLa0EEEvv                # -- Begin function _Z2f3IaJLa0EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IaJLa0EEEvv,@function
_Z2f3IaJLa0EEEvv:                       # @_Z2f3IaJLa0EEEvv
.Lfunc_begin37:
	.loc	1 34 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:34:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp74:
	.loc	1 35 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp75:
.Lfunc_end37:
	.size	_Z2f3IaJLa0EEEvv, .Lfunc_end37-_Z2f3IaJLa0EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3ItJLt1ELt2EEEvv,"axG",@progbits,_Z2f3ItJLt1ELt2EEEvv,comdat
	.weak	_Z2f3ItJLt1ELt2EEEvv            # -- Begin function _Z2f3ItJLt1ELt2EEEvv
	.p2align	4, 0x90
	.type	_Z2f3ItJLt1ELt2EEEvv,@function
_Z2f3ItJLt1ELt2EEEvv:                   # @_Z2f3ItJLt1ELt2EEEvv
.Lfunc_begin38:
	.loc	1 34 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:34:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp76:
	.loc	1 35 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp77:
.Lfunc_end38:
	.size	_Z2f3ItJLt1ELt2EEEvv, .Lfunc_end38-_Z2f3ItJLt1ELt2EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv,"axG",@progbits,_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv,comdat
	.weak	_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv # -- Begin function _Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv
	.p2align	4, 0x90
	.type	_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv,@function
_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv: # @_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv
.Lfunc_begin39:
	.loc	1 34 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:34:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp78:
	.loc	1 35 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp79:
.Lfunc_end39:
	.size	_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv, .Lfunc_end39-_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f3InJLn18446744073709551614EEEvv,"axG",@progbits,_Z2f3InJLn18446744073709551614EEEvv,comdat
	.weak	_Z2f3InJLn18446744073709551614EEEvv # -- Begin function _Z2f3InJLn18446744073709551614EEEvv
	.p2align	4, 0x90
	.type	_Z2f3InJLn18446744073709551614EEEvv,@function
_Z2f3InJLn18446744073709551614EEEvv:    # @_Z2f3InJLn18446744073709551614EEEvv
.Lfunc_begin40:
	.loc	1 34 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:34:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp80:
	.loc	1 35 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:35:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp81:
.Lfunc_end40:
	.size	_Z2f3InJLn18446744073709551614EEEvv, .Lfunc_end40-_Z2f3InJLn18446744073709551614EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f4IjLj3EEvv,"axG",@progbits,_Z2f4IjLj3EEvv,comdat
	.weak	_Z2f4IjLj3EEvv                  # -- Begin function _Z2f4IjLj3EEvv
	.p2align	4, 0x90
	.type	_Z2f4IjLj3EEvv,@function
_Z2f4IjLj3EEvv:                         # @_Z2f4IjLj3EEvv
.Lfunc_begin41:
	.loc	1 37 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:37:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp82:
	.loc	1 38 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:38:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp83:
.Lfunc_end41:
	.size	_Z2f4IjLj3EEvv, .Lfunc_end41-_Z2f4IjLj3EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJ2t3IiLb0EEEEvv,"axG",@progbits,_Z2f1IJ2t3IiLb0EEEEvv,comdat
	.weak	_Z2f1IJ2t3IiLb0EEEEvv           # -- Begin function _Z2f1IJ2t3IiLb0EEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJ2t3IiLb0EEEEvv,@function
_Z2f1IJ2t3IiLb0EEEEvv:                  # @_Z2f1IJ2t3IiLb0EEEEvv
.Lfunc_begin42:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp84:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp85:
.Lfunc_end42:
	.size	_Z2f1IJ2t3IiLb0EEEEvv, .Lfunc_end42-_Z2f1IJ2t3IiLb0EEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv,"axG",@progbits,_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv,comdat
	.weak	_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv  # -- Begin function _Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv,@function
_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv:         # @_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv
.Lfunc_begin43:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp86:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp87:
.Lfunc_end43:
	.size	_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv, .Lfunc_end43-_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJZ4mainE3$_1EEvv
	.type	_Z2f1IJZ4mainE3$_1EEvv,@function
_Z2f1IJZ4mainE3$_1EEvv:                 # @"_Z2f1IJZ4mainE3$_1EEvv"
.Lfunc_begin44:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp88:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp89:
.Lfunc_end44:
	.size	_Z2f1IJZ4mainE3$_1EEvv, .Lfunc_end44-_Z2f1IJZ4mainE3$_1EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJFifEEEvv,"axG",@progbits,_Z2f1IJFifEEEvv,comdat
	.weak	_Z2f1IJFifEEEvv                 # -- Begin function _Z2f1IJFifEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJFifEEEvv,@function
_Z2f1IJFifEEEvv:                        # @_Z2f1IJFifEEEvv
.Lfunc_begin45:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp90:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp91:
.Lfunc_end45:
	.size	_Z2f1IJFifEEEvv, .Lfunc_end45-_Z2f1IJFifEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJRKiEEvv,"axG",@progbits,_Z2f1IJRKiEEvv,comdat
	.weak	_Z2f1IJRKiEEvv                  # -- Begin function _Z2f1IJRKiEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJRKiEEvv,@function
_Z2f1IJRKiEEvv:                         # @_Z2f1IJRKiEEvv
.Lfunc_begin46:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp92:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp93:
.Lfunc_end46:
	.size	_Z2f1IJRKiEEvv, .Lfunc_end46-_Z2f1IJRKiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJRPKiEEvv,"axG",@progbits,_Z2f1IJRPKiEEvv,comdat
	.weak	_Z2f1IJRPKiEEvv                 # -- Begin function _Z2f1IJRPKiEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJRPKiEEvv,@function
_Z2f1IJRPKiEEvv:                        # @_Z2f1IJRPKiEEvv
.Lfunc_begin47:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp94:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp95:
.Lfunc_end47:
	.size	_Z2f1IJRPKiEEvv, .Lfunc_end47-_Z2f1IJRPKiEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJN12_GLOBAL__N_12t5EEEvv
	.type	_Z2f1IJN12_GLOBAL__N_12t5EEEvv,@function
_Z2f1IJN12_GLOBAL__N_12t5EEEvv:         # @_Z2f1IJN12_GLOBAL__N_12t5EEEvv
.Lfunc_begin48:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp96:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp97:
.Lfunc_end48:
	.size	_Z2f1IJN12_GLOBAL__N_12t5EEEvv, .Lfunc_end48-_Z2f1IJN12_GLOBAL__N_12t5EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJDnEEvv,"axG",@progbits,_Z2f1IJDnEEvv,comdat
	.weak	_Z2f1IJDnEEvv                   # -- Begin function _Z2f1IJDnEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJDnEEvv,@function
_Z2f1IJDnEEvv:                          # @_Z2f1IJDnEEvv
.Lfunc_begin49:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp98:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp99:
.Lfunc_end49:
	.size	_Z2f1IJDnEEvv, .Lfunc_end49-_Z2f1IJDnEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPlS0_EEvv,"axG",@progbits,_Z2f1IJPlS0_EEvv,comdat
	.weak	_Z2f1IJPlS0_EEvv                # -- Begin function _Z2f1IJPlS0_EEvv
	.p2align	4, 0x90
	.type	_Z2f1IJPlS0_EEvv,@function
_Z2f1IJPlS0_EEvv:                       # @_Z2f1IJPlS0_EEvv
.Lfunc_begin50:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp100:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp101:
.Lfunc_end50:
	.size	_Z2f1IJPlS0_EEvv, .Lfunc_end50-_Z2f1IJPlS0_EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPlP3udtEEvv,"axG",@progbits,_Z2f1IJPlP3udtEEvv,comdat
	.weak	_Z2f1IJPlP3udtEEvv              # -- Begin function _Z2f1IJPlP3udtEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJPlP3udtEEvv,@function
_Z2f1IJPlP3udtEEvv:                     # @_Z2f1IJPlP3udtEEvv
.Lfunc_begin51:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp102:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp103:
.Lfunc_end51:
	.size	_Z2f1IJPlP3udtEEvv, .Lfunc_end51-_Z2f1IJPlP3udtEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJKPvEEvv,"axG",@progbits,_Z2f1IJKPvEEvv,comdat
	.weak	_Z2f1IJKPvEEvv                  # -- Begin function _Z2f1IJKPvEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJKPvEEvv,@function
_Z2f1IJKPvEEvv:                         # @_Z2f1IJKPvEEvv
.Lfunc_begin52:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp104:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp105:
.Lfunc_end52:
	.size	_Z2f1IJKPvEEvv, .Lfunc_end52-_Z2f1IJKPvEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPKPKvEEvv,"axG",@progbits,_Z2f1IJPKPKvEEvv,comdat
	.weak	_Z2f1IJPKPKvEEvv                # -- Begin function _Z2f1IJPKPKvEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJPKPKvEEvv,@function
_Z2f1IJPKPKvEEvv:                       # @_Z2f1IJPKPKvEEvv
.Lfunc_begin53:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp106:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp107:
.Lfunc_end53:
	.size	_Z2f1IJPKPKvEEvv, .Lfunc_end53-_Z2f1IJPKPKvEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJFvvEEEvv,"axG",@progbits,_Z2f1IJFvvEEEvv,comdat
	.weak	_Z2f1IJFvvEEEvv                 # -- Begin function _Z2f1IJFvvEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJFvvEEEvv,@function
_Z2f1IJFvvEEEvv:                        # @_Z2f1IJFvvEEEvv
.Lfunc_begin54:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp108:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp109:
.Lfunc_end54:
	.size	_Z2f1IJFvvEEEvv, .Lfunc_end54-_Z2f1IJFvvEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPFvvEEEvv,"axG",@progbits,_Z2f1IJPFvvEEEvv,comdat
	.weak	_Z2f1IJPFvvEEEvv                # -- Begin function _Z2f1IJPFvvEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJPFvvEEEvv,@function
_Z2f1IJPFvvEEEvv:                       # @_Z2f1IJPFvvEEEvv
.Lfunc_begin55:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp110:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp111:
.Lfunc_end55:
	.size	_Z2f1IJPFvvEEEvv, .Lfunc_end55-_Z2f1IJPFvvEEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJPZ4mainE3$_1EEvv
	.type	_Z2f1IJPZ4mainE3$_1EEvv,@function
_Z2f1IJPZ4mainE3$_1EEvv:                # @"_Z2f1IJPZ4mainE3$_1EEvv"
.Lfunc_begin56:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp112:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp113:
.Lfunc_end56:
	.size	_Z2f1IJPZ4mainE3$_1EEvv, .Lfunc_end56-_Z2f1IJPZ4mainE3$_1EEvv
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJZ4mainE3$_2EEvv
	.type	_Z2f1IJZ4mainE3$_2EEvv,@function
_Z2f1IJZ4mainE3$_2EEvv:                 # @"_Z2f1IJZ4mainE3$_2EEvv"
.Lfunc_begin57:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp114:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp115:
.Lfunc_end57:
	.size	_Z2f1IJZ4mainE3$_2EEvv, .Lfunc_end57-_Z2f1IJZ4mainE3$_2EEvv
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJPZ4mainE3$_2EEvv
	.type	_Z2f1IJPZ4mainE3$_2EEvv,@function
_Z2f1IJPZ4mainE3$_2EEvv:                # @"_Z2f1IJPZ4mainE3$_2EEvv"
.Lfunc_begin58:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp116:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp117:
.Lfunc_end58:
	.size	_Z2f1IJPZ4mainE3$_2EEvv, .Lfunc_end58-_Z2f1IJPZ4mainE3$_2EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f5IJ2t1IJiEEEiEvv,"axG",@progbits,_Z2f5IJ2t1IJiEEEiEvv,comdat
	.weak	_Z2f5IJ2t1IJiEEEiEvv            # -- Begin function _Z2f5IJ2t1IJiEEEiEvv
	.p2align	4, 0x90
	.type	_Z2f5IJ2t1IJiEEEiEvv,@function
_Z2f5IJ2t1IJiEEEiEvv:                   # @_Z2f5IJ2t1IJiEEEiEvv
.Lfunc_begin59:
	.loc	1 54 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:54:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp118:
	.loc	1 54 13 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:54:13
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp119:
.Lfunc_end59:
	.size	_Z2f5IJ2t1IJiEEEiEvv, .Lfunc_end59-_Z2f5IJ2t1IJiEEEiEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f5IJEiEvv,"axG",@progbits,_Z2f5IJEiEvv,comdat
	.weak	_Z2f5IJEiEvv                    # -- Begin function _Z2f5IJEiEvv
	.p2align	4, 0x90
	.type	_Z2f5IJEiEvv,@function
_Z2f5IJEiEvv:                           # @_Z2f5IJEiEvv
.Lfunc_begin60:
	.loc	1 54 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:54:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp120:
	.loc	1 54 13 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:54:13
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp121:
.Lfunc_end60:
	.size	_Z2f5IJEiEvv, .Lfunc_end60-_Z2f5IJEiEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f6I2t1IJiEEJEEvv,"axG",@progbits,_Z2f6I2t1IJiEEJEEvv,comdat
	.weak	_Z2f6I2t1IJiEEJEEvv             # -- Begin function _Z2f6I2t1IJiEEJEEvv
	.p2align	4, 0x90
	.type	_Z2f6I2t1IJiEEJEEvv,@function
_Z2f6I2t1IJiEEJEEvv:                    # @_Z2f6I2t1IJiEEJEEvv
.Lfunc_begin61:
	.loc	1 56 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:56:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp122:
	.loc	1 56 13 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:56:13
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp123:
.Lfunc_end61:
	.size	_Z2f6I2t1IJiEEJEEvv, .Lfunc_end61-_Z2f6I2t1IJiEEJEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJEEvv,"axG",@progbits,_Z2f1IJEEvv,comdat
	.weak	_Z2f1IJEEvv                     # -- Begin function _Z2f1IJEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJEEvv,@function
_Z2f1IJEEvv:                            # @_Z2f1IJEEvv
.Lfunc_begin62:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp124:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp125:
.Lfunc_end62:
	.size	_Z2f1IJEEvv, .Lfunc_end62-_Z2f1IJEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPKvS1_EEvv,"axG",@progbits,_Z2f1IJPKvS1_EEvv,comdat
	.weak	_Z2f1IJPKvS1_EEvv               # -- Begin function _Z2f1IJPKvS1_EEvv
	.p2align	4, 0x90
	.type	_Z2f1IJPKvS1_EEvv,@function
_Z2f1IJPKvS1_EEvv:                      # @_Z2f1IJPKvS1_EEvv
.Lfunc_begin63:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp126:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp127:
.Lfunc_end63:
	.size	_Z2f1IJPKvS1_EEvv, .Lfunc_end63-_Z2f1IJPKvS1_EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJP2t1IJPiEEEEvv,"axG",@progbits,_Z2f1IJP2t1IJPiEEEEvv,comdat
	.weak	_Z2f1IJP2t1IJPiEEEEvv           # -- Begin function _Z2f1IJP2t1IJPiEEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJP2t1IJPiEEEEvv,@function
_Z2f1IJP2t1IJPiEEEEvv:                  # @_Z2f1IJP2t1IJPiEEEEvv
.Lfunc_begin64:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp128:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp129:
.Lfunc_end64:
	.size	_Z2f1IJP2t1IJPiEEEEvv, .Lfunc_end64-_Z2f1IJP2t1IJPiEEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJA_PiEEvv,"axG",@progbits,_Z2f1IJA_PiEEvv,comdat
	.weak	_Z2f1IJA_PiEEvv                 # -- Begin function _Z2f1IJA_PiEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJA_PiEEvv,@function
_Z2f1IJA_PiEEvv:                        # @_Z2f1IJA_PiEEvv
.Lfunc_begin65:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp130:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp131:
.Lfunc_end65:
	.size	_Z2f1IJA_PiEEvv, .Lfunc_end65-_Z2f1IJA_PiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6lsIiEEvi,"axG",@progbits,_ZN2t6lsIiEEvi,comdat
	.weak	_ZN2t6lsIiEEvi                  # -- Begin function _ZN2t6lsIiEEvi
	.p2align	4, 0x90
	.type	_ZN2t6lsIiEEvi,@function
_ZN2t6lsIiEEvi:                         # @_ZN2t6lsIiEEvi
.Lfunc_begin66:
	.loc	1 59 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:59:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp132:
	.loc	1 60 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:60:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp133:
.Lfunc_end66:
	.size	_ZN2t6lsIiEEvi, .Lfunc_end66-_ZN2t6lsIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6ltIiEEvi,"axG",@progbits,_ZN2t6ltIiEEvi,comdat
	.weak	_ZN2t6ltIiEEvi                  # -- Begin function _ZN2t6ltIiEEvi
	.p2align	4, 0x90
	.type	_ZN2t6ltIiEEvi,@function
_ZN2t6ltIiEEvi:                         # @_ZN2t6ltIiEEvi
.Lfunc_begin67:
	.loc	1 62 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:62:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp134:
	.loc	1 63 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:63:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp135:
.Lfunc_end67:
	.size	_ZN2t6ltIiEEvi, .Lfunc_end67-_ZN2t6ltIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6leIiEEvi,"axG",@progbits,_ZN2t6leIiEEvi,comdat
	.weak	_ZN2t6leIiEEvi                  # -- Begin function _ZN2t6leIiEEvi
	.p2align	4, 0x90
	.type	_ZN2t6leIiEEvi,@function
_ZN2t6leIiEEvi:                         # @_ZN2t6leIiEEvi
.Lfunc_begin68:
	.loc	1 65 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:65:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp136:
	.loc	1 66 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:66:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp137:
.Lfunc_end68:
	.size	_ZN2t6leIiEEvi, .Lfunc_end68-_ZN2t6leIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6cvP2t1IJfEEIiEEv,"axG",@progbits,_ZN2t6cvP2t1IJfEEIiEEv,comdat
	.weak	_ZN2t6cvP2t1IJfEEIiEEv          # -- Begin function _ZN2t6cvP2t1IJfEEIiEEv
	.p2align	4, 0x90
	.type	_ZN2t6cvP2t1IJfEEIiEEv,@function
_ZN2t6cvP2t1IJfEEIiEEv:                 # @_ZN2t6cvP2t1IJfEEIiEEv
.Lfunc_begin69:
	.loc	1 68 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:68:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp138:
	.loc	1 69 5 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:69:5
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp139:
.Lfunc_end69:
	.size	_ZN2t6cvP2t1IJfEEIiEEv, .Lfunc_end69-_ZN2t6cvP2t1IJfEEIiEEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6miIiEEvi,"axG",@progbits,_ZN2t6miIiEEvi,comdat
	.weak	_ZN2t6miIiEEvi                  # -- Begin function _ZN2t6miIiEEvi
	.p2align	4, 0x90
	.type	_ZN2t6miIiEEvi,@function
_ZN2t6miIiEEvi:                         # @_ZN2t6miIiEEvi
.Lfunc_begin70:
	.loc	1 72 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:72:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp140:
	.loc	1 73 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:73:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp141:
.Lfunc_end70:
	.size	_ZN2t6miIiEEvi, .Lfunc_end70-_ZN2t6miIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6mlIiEEvi,"axG",@progbits,_ZN2t6mlIiEEvi,comdat
	.weak	_ZN2t6mlIiEEvi                  # -- Begin function _ZN2t6mlIiEEvi
	.p2align	4, 0x90
	.type	_ZN2t6mlIiEEvi,@function
_ZN2t6mlIiEEvi:                         # @_ZN2t6mlIiEEvi
.Lfunc_begin71:
	.loc	1 75 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:75:0
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
	.loc	1 76 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:76:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp143:
.Lfunc_end71:
	.size	_ZN2t6mlIiEEvi, .Lfunc_end71-_ZN2t6mlIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6dvIiEEvi,"axG",@progbits,_ZN2t6dvIiEEvi,comdat
	.weak	_ZN2t6dvIiEEvi                  # -- Begin function _ZN2t6dvIiEEvi
	.p2align	4, 0x90
	.type	_ZN2t6dvIiEEvi,@function
_ZN2t6dvIiEEvi:                         # @_ZN2t6dvIiEEvi
.Lfunc_begin72:
	.loc	1 78 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:78:0
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
	.loc	1 79 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:79:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp145:
.Lfunc_end72:
	.size	_ZN2t6dvIiEEvi, .Lfunc_end72-_ZN2t6dvIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6rmIiEEvi,"axG",@progbits,_ZN2t6rmIiEEvi,comdat
	.weak	_ZN2t6rmIiEEvi                  # -- Begin function _ZN2t6rmIiEEvi
	.p2align	4, 0x90
	.type	_ZN2t6rmIiEEvi,@function
_ZN2t6rmIiEEvi:                         # @_ZN2t6rmIiEEvi
.Lfunc_begin73:
	.loc	1 81 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:81:0
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
	.loc	1 82 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:82:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp147:
.Lfunc_end73:
	.size	_ZN2t6rmIiEEvi, .Lfunc_end73-_ZN2t6rmIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6eoIiEEvi,"axG",@progbits,_ZN2t6eoIiEEvi,comdat
	.weak	_ZN2t6eoIiEEvi                  # -- Begin function _ZN2t6eoIiEEvi
	.p2align	4, 0x90
	.type	_ZN2t6eoIiEEvi,@function
_ZN2t6eoIiEEvi:                         # @_ZN2t6eoIiEEvi
.Lfunc_begin74:
	.loc	1 84 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:84:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp148:
	.loc	1 85 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:85:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp149:
.Lfunc_end74:
	.size	_ZN2t6eoIiEEvi, .Lfunc_end74-_ZN2t6eoIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6anIiEEvi,"axG",@progbits,_ZN2t6anIiEEvi,comdat
	.weak	_ZN2t6anIiEEvi                  # -- Begin function _ZN2t6anIiEEvi
	.p2align	4, 0x90
	.type	_ZN2t6anIiEEvi,@function
_ZN2t6anIiEEvi:                         # @_ZN2t6anIiEEvi
.Lfunc_begin75:
	.loc	1 87 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:87:0
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
	.loc	1 88 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:88:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp151:
.Lfunc_end75:
	.size	_ZN2t6anIiEEvi, .Lfunc_end75-_ZN2t6anIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6orIiEEvi,"axG",@progbits,_ZN2t6orIiEEvi,comdat
	.weak	_ZN2t6orIiEEvi                  # -- Begin function _ZN2t6orIiEEvi
	.p2align	4, 0x90
	.type	_ZN2t6orIiEEvi,@function
_ZN2t6orIiEEvi:                         # @_ZN2t6orIiEEvi
.Lfunc_begin76:
	.loc	1 90 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:90:0
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
	.loc	1 91 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:91:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp153:
.Lfunc_end76:
	.size	_ZN2t6orIiEEvi, .Lfunc_end76-_ZN2t6orIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6coIiEEvv,"axG",@progbits,_ZN2t6coIiEEvv,comdat
	.weak	_ZN2t6coIiEEvv                  # -- Begin function _ZN2t6coIiEEvv
	.p2align	4, 0x90
	.type	_ZN2t6coIiEEvv,@function
_ZN2t6coIiEEvv:                         # @_ZN2t6coIiEEvv
.Lfunc_begin77:
	.loc	1 93 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:93:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp154:
	.loc	1 94 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:94:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp155:
.Lfunc_end77:
	.size	_ZN2t6coIiEEvv, .Lfunc_end77-_ZN2t6coIiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6ntIiEEvv,"axG",@progbits,_ZN2t6ntIiEEvv,comdat
	.weak	_ZN2t6ntIiEEvv                  # -- Begin function _ZN2t6ntIiEEvv
	.p2align	4, 0x90
	.type	_ZN2t6ntIiEEvv,@function
_ZN2t6ntIiEEvv:                         # @_ZN2t6ntIiEEvv
.Lfunc_begin78:
	.loc	1 96 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:96:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp156:
	.loc	1 97 3 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:97:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp157:
.Lfunc_end78:
	.size	_ZN2t6ntIiEEvv, .Lfunc_end78-_ZN2t6ntIiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6aSIiEEvi,"axG",@progbits,_ZN2t6aSIiEEvi,comdat
	.weak	_ZN2t6aSIiEEvi                  # -- Begin function _ZN2t6aSIiEEvi
	.p2align	4, 0x90
	.type	_ZN2t6aSIiEEvi,@function
_ZN2t6aSIiEEvi:                         # @_ZN2t6aSIiEEvi
.Lfunc_begin79:
	.loc	1 99 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:99:0
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
	.loc	1 100 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:100:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp159:
.Lfunc_end79:
	.size	_ZN2t6aSIiEEvi, .Lfunc_end79-_ZN2t6aSIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6gtIiEEvi,"axG",@progbits,_ZN2t6gtIiEEvi,comdat
	.weak	_ZN2t6gtIiEEvi                  # -- Begin function _ZN2t6gtIiEEvi
	.p2align	4, 0x90
	.type	_ZN2t6gtIiEEvi,@function
_ZN2t6gtIiEEvi:                         # @_ZN2t6gtIiEEvi
.Lfunc_begin80:
	.loc	1 102 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:102:0
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
	.loc	1 103 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:103:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp161:
.Lfunc_end80:
	.size	_ZN2t6gtIiEEvi, .Lfunc_end80-_ZN2t6gtIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6cmIiEEvi,"axG",@progbits,_ZN2t6cmIiEEvi,comdat
	.weak	_ZN2t6cmIiEEvi                  # -- Begin function _ZN2t6cmIiEEvi
	.p2align	4, 0x90
	.type	_ZN2t6cmIiEEvi,@function
_ZN2t6cmIiEEvi:                         # @_ZN2t6cmIiEEvi
.Lfunc_begin81:
	.loc	1 105 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:105:0
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
	.loc	1 106 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:106:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp163:
.Lfunc_end81:
	.size	_ZN2t6cmIiEEvi, .Lfunc_end81-_ZN2t6cmIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6clIiEEvv,"axG",@progbits,_ZN2t6clIiEEvv,comdat
	.weak	_ZN2t6clIiEEvv                  # -- Begin function _ZN2t6clIiEEvv
	.p2align	4, 0x90
	.type	_ZN2t6clIiEEvv,@function
_ZN2t6clIiEEvv:                         # @_ZN2t6clIiEEvv
.Lfunc_begin82:
	.loc	1 108 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:108:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp164:
	.loc	1 109 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:109:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp165:
.Lfunc_end82:
	.size	_ZN2t6clIiEEvv, .Lfunc_end82-_ZN2t6clIiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6ixIiEEvi,"axG",@progbits,_ZN2t6ixIiEEvi,comdat
	.weak	_ZN2t6ixIiEEvi                  # -- Begin function _ZN2t6ixIiEEvi
	.p2align	4, 0x90
	.type	_ZN2t6ixIiEEvi,@function
_ZN2t6ixIiEEvi:                         # @_ZN2t6ixIiEEvi
.Lfunc_begin83:
	.loc	1 111 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:111:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp166:
	.loc	1 112 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:112:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp167:
.Lfunc_end83:
	.size	_ZN2t6ixIiEEvi, .Lfunc_end83-_ZN2t6ixIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6ssIiEEvi,"axG",@progbits,_ZN2t6ssIiEEvi,comdat
	.weak	_ZN2t6ssIiEEvi                  # -- Begin function _ZN2t6ssIiEEvi
	.p2align	4, 0x90
	.type	_ZN2t6ssIiEEvi,@function
_ZN2t6ssIiEEvi:                         # @_ZN2t6ssIiEEvi
.Lfunc_begin84:
	.loc	1 114 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:114:0
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
	.loc	1 115 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:115:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp169:
.Lfunc_end84:
	.size	_ZN2t6ssIiEEvi, .Lfunc_end84-_ZN2t6ssIiEEvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6nwIiEEPvmT_,"axG",@progbits,_ZN2t6nwIiEEPvmT_,comdat
	.weak	_ZN2t6nwIiEEPvmT_               # -- Begin function _ZN2t6nwIiEEPvmT_
	.p2align	4, 0x90
	.type	_ZN2t6nwIiEEPvmT_,@function
_ZN2t6nwIiEEPvmT_:                      # @_ZN2t6nwIiEEPvmT_
.Lfunc_begin85:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Lfunc_end85:
	.size	_ZN2t6nwIiEEPvmT_, .Lfunc_end85-_ZN2t6nwIiEEPvmT_
	.cfi_endproc
	.file	7 "/usr/local/google/home/blaikie/install/bin/../lib/gcc/x86_64-pc-linux-gnu/10.0.0/../../../../include/c++/10.0.0/x86_64-pc-linux-gnu/bits" "c++config.h"
                                        # -- End function
	.section	.text._ZN2t6naIiEEPvmT_,"axG",@progbits,_ZN2t6naIiEEPvmT_,comdat
	.weak	_ZN2t6naIiEEPvmT_               # -- Begin function _ZN2t6naIiEEPvmT_
	.p2align	4, 0x90
	.type	_ZN2t6naIiEEPvmT_,@function
_ZN2t6naIiEEPvmT_:                      # @_ZN2t6naIiEEPvmT_
.Lfunc_begin86:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Lfunc_end86:
	.size	_ZN2t6naIiEEPvmT_, .Lfunc_end86-_ZN2t6naIiEEPvmT_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6dlIiEEvPvT_,"axG",@progbits,_ZN2t6dlIiEEvPvT_,comdat
	.weak	_ZN2t6dlIiEEvPvT_               # -- Begin function _ZN2t6dlIiEEvPvT_
	.p2align	4, 0x90
	.type	_ZN2t6dlIiEEvPvT_,@function
_ZN2t6dlIiEEvPvT_:                      # @_ZN2t6dlIiEEvPvT_
.Lfunc_begin87:
	.loc	1 121 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:121:0
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
	.loc	1 122 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:122:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp171:
.Lfunc_end87:
	.size	_ZN2t6dlIiEEvPvT_, .Lfunc_end87-_ZN2t6dlIiEEvPvT_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6daIiEEvPvT_,"axG",@progbits,_ZN2t6daIiEEvPvT_,comdat
	.weak	_ZN2t6daIiEEvPvT_               # -- Begin function _ZN2t6daIiEEvPvT_
	.p2align	4, 0x90
	.type	_ZN2t6daIiEEvPvT_,@function
_ZN2t6daIiEEvPvT_:                      # @_ZN2t6daIiEEvPvT_
.Lfunc_begin88:
	.loc	1 128 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:128:0
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
	.loc	1 129 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:129:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp173:
.Lfunc_end88:
	.size	_ZN2t6daIiEEvPvT_, .Lfunc_end88-_ZN2t6daIiEEvPvT_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2t6awIiEEiv,"axG",@progbits,_ZN2t6awIiEEiv,comdat
	.weak	_ZN2t6awIiEEiv                  # -- Begin function _ZN2t6awIiEEiv
	.p2align	4, 0x90
	.type	_ZN2t6awIiEEiv,@function
_ZN2t6awIiEEiv:                         # @_ZN2t6awIiEEiv
.Lfunc_begin89:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Lfunc_end89:
	.size	_ZN2t6awIiEEiv, .Lfunc_end89-_ZN2t6awIiEEiv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJZ4mainE2t7EEvv
	.type	_Z2f1IJZ4mainE2t7EEvv,@function
_Z2f1IJZ4mainE2t7EEvv:                  # @_Z2f1IJZ4mainE2t7EEvv
.Lfunc_begin90:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp174:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp175:
.Lfunc_end90:
	.size	_Z2f1IJZ4mainE2t7EEvv, .Lfunc_end90-_Z2f1IJZ4mainE2t7EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJRA3_iEEvv,"axG",@progbits,_Z2f1IJRA3_iEEvv,comdat
	.weak	_Z2f1IJRA3_iEEvv                # -- Begin function _Z2f1IJRA3_iEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJRA3_iEEvv,@function
_Z2f1IJRA3_iEEvv:                       # @_Z2f1IJRA3_iEEvv
.Lfunc_begin91:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp176:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp177:
.Lfunc_end91:
	.size	_Z2f1IJRA3_iEEvv, .Lfunc_end91-_Z2f1IJRA3_iEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPA3_iEEvv,"axG",@progbits,_Z2f1IJPA3_iEEvv,comdat
	.weak	_Z2f1IJPA3_iEEvv                # -- Begin function _Z2f1IJPA3_iEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJPA3_iEEvv,@function
_Z2f1IJPA3_iEEvv:                       # @_Z2f1IJPA3_iEEvv
.Lfunc_begin92:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp178:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp179:
.Lfunc_end92:
	.size	_Z2f1IJPA3_iEEvv, .Lfunc_end92-_Z2f1IJPA3_iEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f7I2t1Evv,"axG",@progbits,_Z2f7I2t1Evv,comdat
	.weak	_Z2f7I2t1Evv                    # -- Begin function _Z2f7I2t1Evv
	.p2align	4, 0x90
	.type	_Z2f7I2t1Evv,@function
_Z2f7I2t1Evv:                           # @_Z2f7I2t1Evv
.Lfunc_begin93:
	.loc	1 135 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:135:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp180:
	.loc	1 135 53 prologue_end           # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:135:53
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp181:
.Lfunc_end93:
	.size	_Z2f7I2t1Evv, .Lfunc_end93-_Z2f7I2t1Evv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f8I2t1iEvv,"axG",@progbits,_Z2f8I2t1iEvv,comdat
	.weak	_Z2f8I2t1iEvv                   # -- Begin function _Z2f8I2t1iEvv
	.p2align	4, 0x90
	.type	_Z2f8I2t1iEvv,@function
_Z2f8I2t1iEvv:                          # @_Z2f8I2t1iEvv
.Lfunc_begin94:
	.loc	1 136 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:136:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp182:
	.loc	1 136 66 prologue_end           # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:136:66
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp183:
.Lfunc_end94:
	.size	_Z2f8I2t1iEvv, .Lfunc_end94-_Z2f8I2t1iEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2ns8ttp_userINS_5inner3ttpEEEvv,"axG",@progbits,_ZN2ns8ttp_userINS_5inner3ttpEEEvv,comdat
	.weak	_ZN2ns8ttp_userINS_5inner3ttpEEEvv # -- Begin function _ZN2ns8ttp_userINS_5inner3ttpEEEvv
	.p2align	4, 0x90
	.type	_ZN2ns8ttp_userINS_5inner3ttpEEEvv,@function
_ZN2ns8ttp_userINS_5inner3ttpEEEvv:     # @_ZN2ns8ttp_userINS_5inner3ttpEEEvv
.Lfunc_begin95:
	.loc	1 19 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:19:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp184:
	.loc	1 19 19 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:19:19
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp185:
.Lfunc_end95:
	.size	_ZN2ns8ttp_userINS_5inner3ttpEEEvv, .Lfunc_end95-_ZN2ns8ttp_userINS_5inner3ttpEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPiPDnEEvv,"axG",@progbits,_Z2f1IJPiPDnEEvv,comdat
	.weak	_Z2f1IJPiPDnEEvv                # -- Begin function _Z2f1IJPiPDnEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJPiPDnEEvv,@function
_Z2f1IJPiPDnEEvv:                       # @_Z2f1IJPiPDnEEvv
.Lfunc_begin96:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp186:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp187:
.Lfunc_end96:
	.size	_Z2f1IJPiPDnEEvv, .Lfunc_end96-_Z2f1IJPiPDnEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJ2t7IiEEEvv,"axG",@progbits,_Z2f1IJ2t7IiEEEvv,comdat
	.weak	_Z2f1IJ2t7IiEEEvv               # -- Begin function _Z2f1IJ2t7IiEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJ2t7IiEEEvv,@function
_Z2f1IJ2t7IiEEEvv:                      # @_Z2f1IJ2t7IiEEEvv
.Lfunc_begin97:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp188:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp189:
.Lfunc_end97:
	.size	_Z2f1IJ2t7IiEEEvv, .Lfunc_end97-_Z2f1IJ2t7IiEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f7IN2ns3inl2t9EEvv,"axG",@progbits,_Z2f7IN2ns3inl2t9EEvv,comdat
	.weak	_Z2f7IN2ns3inl2t9EEvv           # -- Begin function _Z2f7IN2ns3inl2t9EEvv
	.p2align	4, 0x90
	.type	_Z2f7IN2ns3inl2t9EEvv,@function
_Z2f7IN2ns3inl2t9EEvv:                  # @_Z2f7IN2ns3inl2t9EEvv
.Lfunc_begin98:
	.loc	1 135 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:135:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp190:
	.loc	1 135 53 prologue_end           # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:135:53
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp191:
.Lfunc_end98:
	.size	_Z2f7IN2ns3inl2t9EEvv, .Lfunc_end98-_Z2f7IN2ns3inl2t9EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJU7_AtomiciEEvv,"axG",@progbits,_Z2f1IJU7_AtomiciEEvv,comdat
	.weak	_Z2f1IJU7_AtomiciEEvv           # -- Begin function _Z2f1IJU7_AtomiciEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJU7_AtomiciEEvv,@function
_Z2f1IJU7_AtomiciEEvv:                  # @_Z2f1IJU7_AtomiciEEvv
.Lfunc_begin99:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp192:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp193:
.Lfunc_end99:
	.size	_Z2f1IJU7_AtomiciEEvv, .Lfunc_end99-_Z2f1IJU7_AtomiciEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJilVcEEvv,"axG",@progbits,_Z2f1IJilVcEEvv,comdat
	.weak	_Z2f1IJilVcEEvv                 # -- Begin function _Z2f1IJilVcEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJilVcEEvv,@function
_Z2f1IJilVcEEvv:                        # @_Z2f1IJilVcEEvv
.Lfunc_begin100:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp194:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp195:
.Lfunc_end100:
	.size	_Z2f1IJilVcEEvv, .Lfunc_end100-_Z2f1IJilVcEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJDv2_iEEvv,"axG",@progbits,_Z2f1IJDv2_iEEvv,comdat
	.weak	_Z2f1IJDv2_iEEvv                # -- Begin function _Z2f1IJDv2_iEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJDv2_iEEvv,@function
_Z2f1IJDv2_iEEvv:                       # @_Z2f1IJDv2_iEEvv
.Lfunc_begin101:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp196:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp197:
.Lfunc_end101:
	.size	_Z2f1IJDv2_iEEvv, .Lfunc_end101-_Z2f1IJDv2_iEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJVKPiEEvv,"axG",@progbits,_Z2f1IJVKPiEEvv,comdat
	.weak	_Z2f1IJVKPiEEvv                 # -- Begin function _Z2f1IJVKPiEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJVKPiEEvv,@function
_Z2f1IJVKPiEEvv:                        # @_Z2f1IJVKPiEEvv
.Lfunc_begin102:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp198:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp199:
.Lfunc_end102:
	.size	_Z2f1IJVKPiEEvv, .Lfunc_end102-_Z2f1IJVKPiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJVKvEEvv,"axG",@progbits,_Z2f1IJVKvEEvv,comdat
	.weak	_Z2f1IJVKvEEvv                  # -- Begin function _Z2f1IJVKvEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJVKvEEvv,@function
_Z2f1IJVKvEEvv:                         # @_Z2f1IJVKvEEvv
.Lfunc_begin103:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp200:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp201:
.Lfunc_end103:
	.size	_Z2f1IJVKvEEvv, .Lfunc_end103-_Z2f1IJVKvEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJ2t1IJZ4mainE3$_1EEEEvv
	.type	_Z2f1IJ2t1IJZ4mainE3$_1EEEEvv,@function
_Z2f1IJ2t1IJZ4mainE3$_1EEEEvv:          # @"_Z2f1IJ2t1IJZ4mainE3$_1EEEEvv"
.Lfunc_begin104:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp202:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp203:
.Lfunc_end104:
	.size	_Z2f1IJ2t1IJZ4mainE3$_1EEEEvv, .Lfunc_end104-_Z2f1IJ2t1IJZ4mainE3$_1EEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3t10C2IvEEv,"axG",@progbits,_ZN3t10C2IvEEv,comdat
	.weak	_ZN3t10C2IvEEv                  # -- Begin function _ZN3t10C2IvEEv
	.p2align	4, 0x90
	.type	_ZN3t10C2IvEEv,@function
_ZN3t10C2IvEEv:                         # @_ZN3t10C2IvEEv
.Lfunc_begin105:
	.loc	1 159 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:159:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
.Ltmp204:
	.loc	1 159 11 prologue_end           # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:159:11
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp205:
.Lfunc_end105:
	.size	_ZN3t10C2IvEEv, .Lfunc_end105-_ZN3t10C2IvEEv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJM3udtKFvvEEEvv,"axG",@progbits,_Z2f1IJM3udtKFvvEEEvv,comdat
	.weak	_Z2f1IJM3udtKFvvEEEvv           # -- Begin function _Z2f1IJM3udtKFvvEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJM3udtKFvvEEEvv,@function
_Z2f1IJM3udtKFvvEEEvv:                  # @_Z2f1IJM3udtKFvvEEEvv
.Lfunc_begin106:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp206:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp207:
.Lfunc_end106:
	.size	_Z2f1IJM3udtKFvvEEEvv, .Lfunc_end106-_Z2f1IJM3udtKFvvEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJM3udtVFvvREEEvv,"axG",@progbits,_Z2f1IJM3udtVFvvREEEvv,comdat
	.weak	_Z2f1IJM3udtVFvvREEEvv          # -- Begin function _Z2f1IJM3udtVFvvREEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJM3udtVFvvREEEvv,@function
_Z2f1IJM3udtVFvvREEEvv:                 # @_Z2f1IJM3udtVFvvREEEvv
.Lfunc_begin107:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp208:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp209:
.Lfunc_end107:
	.size	_Z2f1IJM3udtVFvvREEEvv, .Lfunc_end107-_Z2f1IJM3udtVFvvREEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJM3udtVKFvvOEEEvv,"axG",@progbits,_Z2f1IJM3udtVKFvvOEEEvv,comdat
	.weak	_Z2f1IJM3udtVKFvvOEEEvv         # -- Begin function _Z2f1IJM3udtVKFvvOEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJM3udtVKFvvOEEEvv,@function
_Z2f1IJM3udtVKFvvOEEEvv:                # @_Z2f1IJM3udtVKFvvOEEEvv
.Lfunc_begin108:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp210:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp211:
.Lfunc_end108:
	.size	_Z2f1IJM3udtVKFvvOEEEvv, .Lfunc_end108-_Z2f1IJM3udtVKFvvOEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f9IiEPFvvEv,"axG",@progbits,_Z2f9IiEPFvvEv,comdat
	.weak	_Z2f9IiEPFvvEv                  # -- Begin function _Z2f9IiEPFvvEv
	.p2align	4, 0x90
	.type	_Z2f9IiEPFvvEv,@function
_Z2f9IiEPFvvEv:                         # @_Z2f9IiEPFvvEv
.Lfunc_begin109:
	.loc	1 154 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:154:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp212:
	.loc	1 155 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:155:3
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp213:
.Lfunc_end109:
	.size	_Z2f9IiEPFvvEv, .Lfunc_end109-_Z2f9IiEPFvvEv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJKPFvvEEEvv,"axG",@progbits,_Z2f1IJKPFvvEEEvv,comdat
	.weak	_Z2f1IJKPFvvEEEvv               # -- Begin function _Z2f1IJKPFvvEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJKPFvvEEEvv,@function
_Z2f1IJKPFvvEEEvv:                      # @_Z2f1IJKPFvvEEEvv
.Lfunc_begin110:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp214:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp215:
.Lfunc_end110:
	.size	_Z2f1IJKPFvvEEEvv, .Lfunc_end110-_Z2f1IJKPFvvEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJRA1_KcEEvv,"axG",@progbits,_Z2f1IJRA1_KcEEvv,comdat
	.weak	_Z2f1IJRA1_KcEEvv               # -- Begin function _Z2f1IJRA1_KcEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJRA1_KcEEvv,@function
_Z2f1IJRA1_KcEEvv:                      # @_Z2f1IJRA1_KcEEvv
.Lfunc_begin111:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp216:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp217:
.Lfunc_end111:
	.size	_Z2f1IJRA1_KcEEvv, .Lfunc_end111-_Z2f1IJRA1_KcEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJKFvvREEEvv,"axG",@progbits,_Z2f1IJKFvvREEEvv,comdat
	.weak	_Z2f1IJKFvvREEEvv               # -- Begin function _Z2f1IJKFvvREEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJKFvvREEEvv,@function
_Z2f1IJKFvvREEEvv:                      # @_Z2f1IJKFvvREEEvv
.Lfunc_begin112:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp218:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp219:
.Lfunc_end112:
	.size	_Z2f1IJKFvvREEEvv, .Lfunc_end112-_Z2f1IJKFvvREEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJVFvvOEEEvv,"axG",@progbits,_Z2f1IJVFvvOEEEvv,comdat
	.weak	_Z2f1IJVFvvOEEEvv               # -- Begin function _Z2f1IJVFvvOEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJVFvvOEEEvv,@function
_Z2f1IJVFvvOEEEvv:                      # @_Z2f1IJVFvvOEEEvv
.Lfunc_begin113:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp220:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp221:
.Lfunc_end113:
	.size	_Z2f1IJVFvvOEEEvv, .Lfunc_end113-_Z2f1IJVFvvOEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJVKFvvEEEvv,"axG",@progbits,_Z2f1IJVKFvvEEEvv,comdat
	.weak	_Z2f1IJVKFvvEEEvv               # -- Begin function _Z2f1IJVKFvvEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJVKFvvEEEvv,@function
_Z2f1IJVKFvvEEEvv:                      # @_Z2f1IJVKFvvEEEvv
.Lfunc_begin114:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp222:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp223:
.Lfunc_end114:
	.size	_Z2f1IJVKFvvEEEvv, .Lfunc_end114-_Z2f1IJVKFvvEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJA1_KPiEEvv,"axG",@progbits,_Z2f1IJA1_KPiEEvv,comdat
	.weak	_Z2f1IJA1_KPiEEvv               # -- Begin function _Z2f1IJA1_KPiEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJA1_KPiEEvv,@function
_Z2f1IJA1_KPiEEvv:                      # @_Z2f1IJA1_KPiEEvv
.Lfunc_begin115:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp224:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp225:
.Lfunc_end115:
	.size	_Z2f1IJA1_KPiEEvv, .Lfunc_end115-_Z2f1IJA1_KPiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJRA1_KPiEEvv,"axG",@progbits,_Z2f1IJRA1_KPiEEvv,comdat
	.weak	_Z2f1IJRA1_KPiEEvv              # -- Begin function _Z2f1IJRA1_KPiEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJRA1_KPiEEvv,@function
_Z2f1IJRA1_KPiEEvv:                     # @_Z2f1IJRA1_KPiEEvv
.Lfunc_begin116:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp226:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp227:
.Lfunc_end116:
	.size	_Z2f1IJRA1_KPiEEvv, .Lfunc_end116-_Z2f1IJRA1_KPiEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJRKM3udtFvvEEEvv,"axG",@progbits,_Z2f1IJRKM3udtFvvEEEvv,comdat
	.weak	_Z2f1IJRKM3udtFvvEEEvv          # -- Begin function _Z2f1IJRKM3udtFvvEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJRKM3udtFvvEEEvv,@function
_Z2f1IJRKM3udtFvvEEEvv:                 # @_Z2f1IJRKM3udtFvvEEEvv
.Lfunc_begin117:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp228:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp229:
.Lfunc_end117:
	.size	_Z2f1IJRKM3udtFvvEEEvv, .Lfunc_end117-_Z2f1IJRKM3udtFvvEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJFPFvfEiEEEvv,"axG",@progbits,_Z2f1IJFPFvfEiEEEvv,comdat
	.weak	_Z2f1IJFPFvfEiEEEvv             # -- Begin function _Z2f1IJFPFvfEiEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJFPFvfEiEEEvv,@function
_Z2f1IJFPFvfEiEEEvv:                    # @_Z2f1IJFPFvfEiEEEvv
.Lfunc_begin118:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp230:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp231:
.Lfunc_end118:
	.size	_Z2f1IJFPFvfEiEEEvv, .Lfunc_end118-_Z2f1IJFPFvfEiEEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJPDoFvvEEEvv,"axG",@progbits,_Z2f1IJPDoFvvEEEvv,comdat
	.weak	_Z2f1IJPDoFvvEEEvv              # -- Begin function _Z2f1IJPDoFvvEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJPDoFvvEEEvv,@function
_Z2f1IJPDoFvvEEEvv:                     # @_Z2f1IJPDoFvvEEEvv
.Lfunc_begin119:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp232:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp233:
.Lfunc_end119:
	.size	_Z2f1IJPDoFvvEEEvv, .Lfunc_end119-_Z2f1IJPDoFvvEEEvv
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJFvZ4mainE3$_2EEEvv
	.type	_Z2f1IJFvZ4mainE3$_2EEEvv,@function
_Z2f1IJFvZ4mainE3$_2EEEvv:              # @"_Z2f1IJFvZ4mainE3$_2EEEvv"
.Lfunc_begin120:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp234:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp235:
.Lfunc_end120:
	.size	_Z2f1IJFvZ4mainE3$_2EEEvv, .Lfunc_end120-_Z2f1IJFvZ4mainE3$_2EEEvv
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv
	.type	_Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv,@function
_Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv:    # @"_Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv"
.Lfunc_begin121:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp236:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp237:
.Lfunc_end121:
	.size	_Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv, .Lfunc_end121-_Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJFvZ4mainE2t8EEEvv
	.type	_Z2f1IJFvZ4mainE2t8EEEvv,@function
_Z2f1IJFvZ4mainE2t8EEEvv:               # @_Z2f1IJFvZ4mainE2t8EEEvv
.Lfunc_begin122:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp238:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp239:
.Lfunc_end122:
	.size	_Z2f1IJFvZ4mainE2t8EEEvv, .Lfunc_end122-_Z2f1IJFvZ4mainE2t8EEEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z19operator_not_reallyIiEvv,"axG",@progbits,_Z19operator_not_reallyIiEvv,comdat
	.weak	_Z19operator_not_reallyIiEvv    # -- Begin function _Z19operator_not_reallyIiEvv
	.p2align	4, 0x90
	.type	_Z19operator_not_reallyIiEvv,@function
_Z19operator_not_reallyIiEvv:           # @_Z19operator_not_reallyIiEvv
.Lfunc_begin123:
	.loc	1 163 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:163:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp240:
	.loc	1 164 1 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:164:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp241:
.Lfunc_end123:
	.size	_Z19operator_not_reallyIiEvv, .Lfunc_end123-_Z19operator_not_reallyIiEvv
	.cfi_endproc
                                        # -- End function
	.text
	.globl	_ZN2t83memEv                    # -- Begin function _ZN2t83memEv
	.p2align	4, 0x90
	.type	_ZN2t83memEv,@function
_ZN2t83memEv:                           # @_ZN2t83memEv
.Lfunc_begin124:
	.loc	1 302 0                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:302:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
.Ltmp242:
	.loc	1 304 3 prologue_end            # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:304:3
	callq	_Z2f1IJZN2t83memEvE2t7EEvv
	.loc	1 305 3                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:305:3
	callq	_Z2f1IJM2t8FvvEEEvv
	.loc	1 306 1                         # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:306:1
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp243:
.Lfunc_end124:
	.size	_ZN2t83memEv, .Lfunc_end124-_ZN2t83memEv
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _Z2f1IJZN2t83memEvE2t7EEvv
	.type	_Z2f1IJZN2t83memEvE2t7EEvv,@function
_Z2f1IJZN2t83memEvE2t7EEvv:             # @_Z2f1IJZN2t83memEvE2t7EEvv
.Lfunc_begin125:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp244:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp245:
.Lfunc_end125:
	.size	_Z2f1IJZN2t83memEvE2t7EEvv, .Lfunc_end125-_Z2f1IJZN2t83memEvE2t7EEvv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f1IJM2t8FvvEEEvv,"axG",@progbits,_Z2f1IJM2t8FvvEEEvv,comdat
	.weak	_Z2f1IJM2t8FvvEEEvv             # -- Begin function _Z2f1IJM2t8FvvEEEvv
	.p2align	4, 0x90
	.type	_Z2f1IJM2t8FvvEEEvv,@function
_Z2f1IJM2t8FvvEEEvv:                    # @_Z2f1IJM2t8FvvEEEvv
.Lfunc_begin126:
	.loc	1 26 0                          # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:26:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp246:
	.loc	1 29 1 prologue_end             # cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:29:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp247:
.Lfunc_end126:
	.size	_Z2f1IJM2t8FvvEEEvv, .Lfunc_end126-_Z2f1IJM2t8FvvEEEvv
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
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
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
	.byte	14                              # DW_FORM_strp
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	57                              # DW_TAG_namespace
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	4                               # DW_TAG_enumeration_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
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
	.byte	14                              # DW_FORM_strp
	.byte	28                              # DW_AT_const_value
	.byte	15                              # DW_FORM_udata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	4                               # DW_TAG_enumeration_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	109                             # DW_AT_enum_class
	.byte	25                              # DW_FORM_flag_present
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	40                              # DW_TAG_enumerator
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	28                              # DW_AT_const_value
	.byte	13                              # DW_FORM_sdata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
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
	.byte	10                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.ascii	"\206\202\001"                  # DW_TAG_GNU_template_template_param
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.ascii	"\220B"                         # DW_AT_GNU_template_name
	.byte	14                              # DW_FORM_strp
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	12                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	13                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	14                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	15                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	28                              # DW_AT_const_value
	.byte	15                              # DW_FORM_udata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	16                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
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
	.byte	17                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	18                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	52                              # DW_AT_artificial
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	19                              # Abbreviation Code
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
	.byte	20                              # Abbreviation Code
	.byte	22                              # DW_TAG_typedef
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	5                               # DW_FORM_data2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	21                              # Abbreviation Code
	.byte	22                              # DW_TAG_typedef
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
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
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
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
	.byte	14                              # DW_FORM_strp
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
	.byte	14                              # DW_FORM_strp
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
	.ascii	"\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
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
	.byte	14                              # DW_FORM_strp
	.byte	28                              # DW_AT_const_value
	.byte	13                              # DW_FORM_sdata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	33                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	28                              # DW_AT_const_value
	.byte	15                              # DW_FORM_udata
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
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	36                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	37                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	28                              # DW_AT_const_value
	.byte	10                              # DW_FORM_block1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	38                              # Abbreviation Code
	.ascii	"\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	39                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
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
	.byte	40                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	41                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
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
	.byte	42                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	43                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
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
	.byte	44                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	52                              # DW_AT_artificial
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	45                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	46                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	71                              # DW_AT_specification
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	47                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	100                             # DW_AT_object_pointer
	.byte	19                              # DW_FORM_ref4
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	71                              # DW_AT_specification
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	48                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
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
	.byte	49                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
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
	.byte	50                              # Abbreviation Code
	.byte	16                              # DW_TAG_reference_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	51                              # Abbreviation Code
	.byte	66                              # DW_TAG_rvalue_reference_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	52                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	53                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	54                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	55                              # Abbreviation Code
	.byte	57                              # DW_TAG_namespace
	.byte	1                               # DW_CHILDREN_yes
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	56                              # Abbreviation Code
	.byte	59                              # DW_TAG_unspecified_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	57                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	58                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	59                              # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	60                              # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	61                              # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	62                              # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	55                              # DW_AT_count
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	63                              # Abbreviation Code
	.byte	53                              # DW_TAG_volatile_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	64                              # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.ascii	"\207B"                         # DW_AT_GNU_vector
	.byte	25                              # DW_FORM_flag_present
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	65                              # Abbreviation Code
	.byte	53                              # DW_TAG_volatile_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	66                              # Abbreviation Code
	.byte	31                              # DW_TAG_ptr_to_member_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	29                              # DW_AT_containing_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	67                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	68                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	119                             # DW_AT_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	69                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	120                             # DW_AT_rvalue_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	70                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	119                             # DW_AT_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	71                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	120                             # DW_AT_rvalue_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x3300 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	0                               # DW_AT_low_pc
	.long	.Ldebug_ranges0                 # DW_AT_ranges
	.byte	2                               # Abbrev [2] 0x2a:0x15 DW_TAG_variable
	.long	.Linfo_string3                  # DW_AT_name
	.long	63                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	47                              # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	i
	.byte	3                               # Abbrev [3] 0x3f:0x7 DW_TAG_base_type
	.long	.Linfo_string4                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	4                               # Abbrev [4] 0x46:0xa6 DW_TAG_namespace
	.long	.Linfo_string5                  # DW_AT_name
	.byte	5                               # Abbrev [5] 0x4b:0x1f DW_TAG_enumeration_type
	.long	236                             # DW_AT_type
	.long	.Linfo_string10                 # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.byte	6                               # Abbrev [6] 0x57:0x6 DW_TAG_enumerator
	.long	.Linfo_string7                  # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	6                               # Abbrev [6] 0x5d:0x6 DW_TAG_enumerator
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	6                               # Abbrev [6] 0x63:0x6 DW_TAG_enumerator
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	7                               # Abbrev [7] 0x6a:0x1f DW_TAG_enumeration_type
	.long	63                              # DW_AT_type
                                        # DW_AT_enum_class
	.long	.Linfo_string11                 # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	21                              # DW_AT_decl_line
	.byte	8                               # Abbrev [8] 0x76:0x6 DW_TAG_enumerator
	.long	.Linfo_string7                  # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	8                               # Abbrev [8] 0x7c:0x6 DW_TAG_enumerator
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	8                               # Abbrev [8] 0x82:0x6 DW_TAG_enumerator
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x89:0x14 DW_TAG_enumeration_type
	.long	243                             # DW_AT_type
	.long	.Linfo_string14                 # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
	.byte	6                               # Abbrev [6] 0x95:0x7 DW_TAG_enumerator
	.long	.Linfo_string13                 # DW_AT_name
	.ascii	"\377\001"                      # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0x9d:0x1b DW_TAG_enumeration_type
	.long	236                             # DW_AT_type
	.byte	4                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	22                              # DW_AT_decl_line
	.byte	6                               # Abbrev [6] 0xa5:0x6 DW_TAG_enumerator
	.long	.Linfo_string15                 # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	6                               # Abbrev [6] 0xab:0x6 DW_TAG_enumerator
	.long	.Linfo_string16                 # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	6                               # Abbrev [6] 0xb1:0x6 DW_TAG_enumerator
	.long	.Linfo_string17                 # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xb8:0x23 DW_TAG_subprogram
	.quad	.Lfunc_begin95                  # DW_AT_low_pc
	.long	.Lfunc_end95-.Lfunc_begin95     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string294                # DW_AT_linkage_name
	.long	.Linfo_string295                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	11                              # Abbrev [11] 0xd1:0x9 DW_TAG_GNU_template_template_param
	.long	.Linfo_string18                 # DW_AT_name
	.long	.Linfo_string293                # DW_AT_GNU_template_name
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xdb:0x5 DW_TAG_structure_type
	.long	.Linfo_string156                # DW_AT_name
                                        # DW_AT_declaration
	.byte	4                               # Abbrev [4] 0xe0:0xb DW_TAG_namespace
	.long	.Linfo_string163                # DW_AT_name
	.byte	12                              # Abbrev [12] 0xe5:0x5 DW_TAG_structure_type
	.long	.Linfo_string156                # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0xec:0x7 DW_TAG_base_type
	.long	.Linfo_string6                  # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0xf3:0x7 DW_TAG_base_type
	.long	.Linfo_string12                 # DW_AT_name
	.byte	8                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	13                              # Abbrev [13] 0xfa:0x1d DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string21                 # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
	.byte	14                              # Abbrev [14] 0x103:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	15                              # Abbrev [15] 0x10c:0xa DW_TAG_template_value_parameter
	.long	279                             # DW_AT_type
	.long	.Linfo_string20                 # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0x117:0x7 DW_TAG_base_type
	.long	.Linfo_string19                 # DW_AT_name
	.byte	2                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	13                              # Abbrev [13] 0x11e:0x1c DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string22                 # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	157                             # DW_AT_decl_line
	.byte	16                              # Abbrev [16] 0x127:0x12 DW_TAG_subprogram
	.long	.Linfo_string128                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	159                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	17                              # Abbrev [17] 0x12e:0x5 DW_TAG_template_type_parameter
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x133:0x5 DW_TAG_formal_parameter
	.long	7960                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x13a:0xd6 DW_TAG_namespace
	.long	.Linfo_string23                 # DW_AT_name
	.byte	19                              # Abbrev [19] 0x13f:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	47                              # DW_AT_decl_line
	.long	528                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x146:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	48                              # DW_AT_decl_line
	.long	557                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x14d:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	49                              # DW_AT_decl_line
	.long	586                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x154:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	50                              # DW_AT_decl_line
	.long	608                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x15b:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	52                              # DW_AT_decl_line
	.long	637                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x162:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	53                              # DW_AT_decl_line
	.long	648                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x169:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	54                              # DW_AT_decl_line
	.long	659                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x170:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	55                              # DW_AT_decl_line
	.long	670                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x177:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	57                              # DW_AT_decl_line
	.long	681                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x17e:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.long	703                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x185:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	59                              # DW_AT_decl_line
	.long	725                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x18c:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	60                              # DW_AT_decl_line
	.long	747                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x193:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	62                              # DW_AT_decl_line
	.long	769                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x19a:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	63                              # DW_AT_decl_line
	.long	791                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x1a1:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	65                              # DW_AT_decl_line
	.long	802                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x1a8:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	66                              # DW_AT_decl_line
	.long	824                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x1af:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	67                              # DW_AT_decl_line
	.long	853                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x1b6:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	68                              # DW_AT_decl_line
	.long	875                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x1bd:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	70                              # DW_AT_decl_line
	.long	904                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x1c4:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	71                              # DW_AT_decl_line
	.long	915                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x1cb:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	72                              # DW_AT_decl_line
	.long	926                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x1d2:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
	.long	937                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x1d9:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	75                              # DW_AT_decl_line
	.long	948                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x1e0:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	76                              # DW_AT_decl_line
	.long	970                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x1e7:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	77                              # DW_AT_decl_line
	.long	992                             # DW_AT_import
	.byte	19                              # Abbrev [19] 0x1ee:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	78                              # DW_AT_decl_line
	.long	1014                            # DW_AT_import
	.byte	19                              # Abbrev [19] 0x1f5:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	80                              # DW_AT_decl_line
	.long	1036                            # DW_AT_import
	.byte	19                              # Abbrev [19] 0x1fc:0x7 DW_TAG_imported_declaration
	.byte	4                               # DW_AT_decl_file
	.byte	81                              # DW_AT_decl_line
	.long	1058                            # DW_AT_import
	.byte	20                              # Abbrev [20] 0x203:0xc DW_TAG_typedef
	.long	897                             # DW_AT_type
	.long	.Linfo_string119                # DW_AT_name
	.byte	7                               # DW_AT_decl_file
	.short	260                             # DW_AT_decl_line
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x210:0xb DW_TAG_typedef
	.long	539                             # DW_AT_type
	.long	.Linfo_string26                 # DW_AT_name
	.byte	3                               # DW_AT_decl_file
	.byte	24                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x21b:0xb DW_TAG_typedef
	.long	550                             # DW_AT_type
	.long	.Linfo_string25                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x226:0x7 DW_TAG_base_type
	.long	.Linfo_string24                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	21                              # Abbrev [21] 0x22d:0xb DW_TAG_typedef
	.long	568                             # DW_AT_type
	.long	.Linfo_string29                 # DW_AT_name
	.byte	3                               # DW_AT_decl_file
	.byte	25                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x238:0xb DW_TAG_typedef
	.long	579                             # DW_AT_type
	.long	.Linfo_string28                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	39                              # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x243:0x7 DW_TAG_base_type
	.long	.Linfo_string27                 # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	2                               # DW_AT_byte_size
	.byte	21                              # Abbrev [21] 0x24a:0xb DW_TAG_typedef
	.long	597                             # DW_AT_type
	.long	.Linfo_string31                 # DW_AT_name
	.byte	3                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x255:0xb DW_TAG_typedef
	.long	63                              # DW_AT_type
	.long	.Linfo_string30                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	41                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x260:0xb DW_TAG_typedef
	.long	619                             # DW_AT_type
	.long	.Linfo_string34                 # DW_AT_name
	.byte	3                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x26b:0xb DW_TAG_typedef
	.long	630                             # DW_AT_type
	.long	.Linfo_string33                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	44                              # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x276:0x7 DW_TAG_base_type
	.long	.Linfo_string32                 # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	21                              # Abbrev [21] 0x27d:0xb DW_TAG_typedef
	.long	550                             # DW_AT_type
	.long	.Linfo_string35                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x288:0xb DW_TAG_typedef
	.long	630                             # DW_AT_type
	.long	.Linfo_string36                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	60                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x293:0xb DW_TAG_typedef
	.long	630                             # DW_AT_type
	.long	.Linfo_string37                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	61                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x29e:0xb DW_TAG_typedef
	.long	630                             # DW_AT_type
	.long	.Linfo_string38                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	62                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x2a9:0xb DW_TAG_typedef
	.long	692                             # DW_AT_type
	.long	.Linfo_string40                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x2b4:0xb DW_TAG_typedef
	.long	539                             # DW_AT_type
	.long	.Linfo_string39                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	52                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x2bf:0xb DW_TAG_typedef
	.long	714                             # DW_AT_type
	.long	.Linfo_string42                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	44                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x2ca:0xb DW_TAG_typedef
	.long	568                             # DW_AT_type
	.long	.Linfo_string41                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	54                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x2d5:0xb DW_TAG_typedef
	.long	736                             # DW_AT_type
	.long	.Linfo_string44                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	45                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x2e0:0xb DW_TAG_typedef
	.long	597                             # DW_AT_type
	.long	.Linfo_string43                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	56                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x2eb:0xb DW_TAG_typedef
	.long	758                             # DW_AT_type
	.long	.Linfo_string46                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	46                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x2f6:0xb DW_TAG_typedef
	.long	619                             # DW_AT_type
	.long	.Linfo_string45                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x301:0xb DW_TAG_typedef
	.long	780                             # DW_AT_type
	.long	.Linfo_string48                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	101                             # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x30c:0xb DW_TAG_typedef
	.long	630                             # DW_AT_type
	.long	.Linfo_string47                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	72                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x317:0xb DW_TAG_typedef
	.long	630                             # DW_AT_type
	.long	.Linfo_string49                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	87                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x322:0xb DW_TAG_typedef
	.long	813                             # DW_AT_type
	.long	.Linfo_string51                 # DW_AT_name
	.byte	6                               # DW_AT_decl_file
	.byte	24                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x32d:0xb DW_TAG_typedef
	.long	243                             # DW_AT_type
	.long	.Linfo_string50                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	38                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x338:0xb DW_TAG_typedef
	.long	835                             # DW_AT_type
	.long	.Linfo_string54                 # DW_AT_name
	.byte	6                               # DW_AT_decl_file
	.byte	25                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x343:0xb DW_TAG_typedef
	.long	846                             # DW_AT_type
	.long	.Linfo_string53                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	40                              # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x34e:0x7 DW_TAG_base_type
	.long	.Linfo_string52                 # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	2                               # DW_AT_byte_size
	.byte	21                              # Abbrev [21] 0x355:0xb DW_TAG_typedef
	.long	864                             # DW_AT_type
	.long	.Linfo_string56                 # DW_AT_name
	.byte	6                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x360:0xb DW_TAG_typedef
	.long	236                             # DW_AT_type
	.long	.Linfo_string55                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	42                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x36b:0xb DW_TAG_typedef
	.long	886                             # DW_AT_type
	.long	.Linfo_string59                 # DW_AT_name
	.byte	6                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x376:0xb DW_TAG_typedef
	.long	897                             # DW_AT_type
	.long	.Linfo_string58                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	45                              # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x381:0x7 DW_TAG_base_type
	.long	.Linfo_string57                 # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	21                              # Abbrev [21] 0x388:0xb DW_TAG_typedef
	.long	243                             # DW_AT_type
	.long	.Linfo_string60                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	71                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x393:0xb DW_TAG_typedef
	.long	897                             # DW_AT_type
	.long	.Linfo_string61                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x39e:0xb DW_TAG_typedef
	.long	897                             # DW_AT_type
	.long	.Linfo_string62                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	74                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x3a9:0xb DW_TAG_typedef
	.long	897                             # DW_AT_type
	.long	.Linfo_string63                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	75                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x3b4:0xb DW_TAG_typedef
	.long	959                             # DW_AT_type
	.long	.Linfo_string65                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	49                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x3bf:0xb DW_TAG_typedef
	.long	813                             # DW_AT_type
	.long	.Linfo_string64                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	53                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x3ca:0xb DW_TAG_typedef
	.long	981                             # DW_AT_type
	.long	.Linfo_string67                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	50                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x3d5:0xb DW_TAG_typedef
	.long	835                             # DW_AT_type
	.long	.Linfo_string66                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	55                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x3e0:0xb DW_TAG_typedef
	.long	1003                            # DW_AT_type
	.long	.Linfo_string69                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	51                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x3eb:0xb DW_TAG_typedef
	.long	864                             # DW_AT_type
	.long	.Linfo_string68                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	57                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x3f6:0xb DW_TAG_typedef
	.long	1025                            # DW_AT_type
	.long	.Linfo_string71                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	52                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x401:0xb DW_TAG_typedef
	.long	886                             # DW_AT_type
	.long	.Linfo_string70                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	59                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x40c:0xb DW_TAG_typedef
	.long	1047                            # DW_AT_type
	.long	.Linfo_string73                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	102                             # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x417:0xb DW_TAG_typedef
	.long	897                             # DW_AT_type
	.long	.Linfo_string72                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x422:0xb DW_TAG_typedef
	.long	897                             # DW_AT_type
	.long	.Linfo_string74                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	90                              # DW_AT_decl_line
	.byte	10                              # Abbrev [10] 0x42d:0x24 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string132                # DW_AT_linkage_name
	.long	.Linfo_string133                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	134                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	22                              # Abbrev [22] 0x446:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	1                               # DW_AT_decl_file
	.byte	134                             # DW_AT_decl_line
	.long	9336                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x451:0x9a DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string134                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	166                             # DW_AT_decl_line
	.long	63                              # DW_AT_type
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x46a:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string189                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	167                             # DW_AT_decl_line
	.long	1243                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x478:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string357                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	168                             # DW_AT_decl_line
	.long	1238                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x486:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	104
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	199                             # DW_AT_decl_line
	.long	9929                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x494:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	96
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	214                             # DW_AT_decl_line
	.long	9949                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x4a2:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	88
	.long	.Linfo_string363                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	236                             # DW_AT_decl_line
	.long	5142                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x4b0:0xf DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	80
	.long	.Linfo_string364                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	271                             # DW_AT_decl_line
	.long	9978                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x4bf:0xf DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	72
	.long	.Linfo_string366                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	280                             # DW_AT_decl_line
	.long	286                             # DW_AT_type
	.byte	26                              # Abbrev [26] 0x4ce:0x8 DW_TAG_imported_module
	.byte	1                               # DW_AT_decl_file
	.short	268                             # DW_AT_decl_line
	.long	70                              # DW_AT_import
	.byte	27                              # Abbrev [27] 0x4d6:0x5 DW_TAG_class_type
	.byte	5                               # DW_AT_calling_convention
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	168                             # DW_AT_decl_line
	.byte	28                              # Abbrev [28] 0x4db:0x5 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	167                             # DW_AT_decl_line
	.byte	12                              # Abbrev [12] 0x4e0:0x5 DW_TAG_structure_type
	.long	.Linfo_string281                # DW_AT_name
                                        # DW_AT_declaration
	.byte	12                              # Abbrev [12] 0x4e5:0x5 DW_TAG_structure_type
	.long	.Linfo_string131                # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x4eb:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin2                   # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string135                # DW_AT_linkage_name
	.long	.Linfo_string136                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x504:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	9360                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x512:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	9989                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x520:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x525:0x5 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x52c:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin3                   # DW_AT_low_pc
	.long	.Lfunc_end3-.Lfunc_begin3       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string137                # DW_AT_linkage_name
	.long	.Linfo_string138                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x545:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	6062                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x553:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10011                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x561:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x566:0x5 DW_TAG_template_type_parameter
	.long	6083                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x56d:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin4                   # DW_AT_low_pc
	.long	.Lfunc_end4-.Lfunc_begin4       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string139                # DW_AT_linkage_name
	.long	.Linfo_string140                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x586:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10033                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x594:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10054                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x5a2:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x5a7:0x5 DW_TAG_template_type_parameter
	.long	279                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x5ae:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin5                   # DW_AT_low_pc
	.long	.Lfunc_end5-.Lfunc_begin5       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string142                # DW_AT_linkage_name
	.long	.Linfo_string143                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x5c7:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10076                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x5d5:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10097                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x5e3:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x5e8:0x5 DW_TAG_template_type_parameter
	.long	9329                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x5ef:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin6                   # DW_AT_low_pc
	.long	.Lfunc_end6-.Lfunc_begin6       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string144                # DW_AT_linkage_name
	.long	.Linfo_string145                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x608:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10119                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x616:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10140                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x624:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x629:0x5 DW_TAG_template_type_parameter
	.long	630                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x630:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin7                   # DW_AT_low_pc
	.long	.Lfunc_end7-.Lfunc_begin7       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string146                # DW_AT_linkage_name
	.long	.Linfo_string147                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x649:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10162                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x657:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10183                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x665:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x66a:0x5 DW_TAG_template_type_parameter
	.long	579                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x671:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin8                   # DW_AT_low_pc
	.long	.Lfunc_end8-.Lfunc_begin8       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string148                # DW_AT_linkage_name
	.long	.Linfo_string149                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x68a:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10205                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x698:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10226                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x6a6:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x6ab:0x5 DW_TAG_template_type_parameter
	.long	236                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x6b2:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin9                   # DW_AT_low_pc
	.long	.Lfunc_end9-.Lfunc_begin9       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string151                # DW_AT_linkage_name
	.long	.Linfo_string152                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x6cb:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10248                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x6d9:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10269                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x6e7:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x6ec:0x5 DW_TAG_template_type_parameter
	.long	9336                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x6f3:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin10                  # DW_AT_low_pc
	.long	.Lfunc_end10-.Lfunc_begin10     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string154                # DW_AT_linkage_name
	.long	.Linfo_string155                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x70c:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10291                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x71a:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10312                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x728:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x72d:0x5 DW_TAG_template_type_parameter
	.long	9343                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x734:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin11                  # DW_AT_low_pc
	.long	.Lfunc_end11-.Lfunc_begin11     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string157                # DW_AT_linkage_name
	.long	.Linfo_string158                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x74d:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10334                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x75b:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10355                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x769:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x76e:0x5 DW_TAG_template_type_parameter
	.long	9350                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x775:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin12                  # DW_AT_low_pc
	.long	.Lfunc_end12-.Lfunc_begin12     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string159                # DW_AT_linkage_name
	.long	.Linfo_string160                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x78e:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10377                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x79c:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10398                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x7aa:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x7af:0x5 DW_TAG_template_type_parameter
	.long	219                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x7b6:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin13                  # DW_AT_low_pc
	.long	.Lfunc_end13-.Lfunc_begin13     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string161                # DW_AT_linkage_name
	.long	.Linfo_string162                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x7cf:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10420                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x7dd:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10441                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x7eb:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x7f0:0x5 DW_TAG_template_type_parameter
	.long	9355                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x7f7:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin14                  # DW_AT_low_pc
	.long	.Lfunc_end14-.Lfunc_begin14     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string164                # DW_AT_linkage_name
	.long	.Linfo_string165                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x810:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10463                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x81e:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10484                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x82c:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x831:0x5 DW_TAG_template_type_parameter
	.long	229                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x838:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin15                  # DW_AT_low_pc
	.long	.Lfunc_end15-.Lfunc_begin15     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string167                # DW_AT_linkage_name
	.long	.Linfo_string168                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x851:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10506                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x85f:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10527                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x86d:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x872:0x5 DW_TAG_template_type_parameter
	.long	9360                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x879:0x46 DW_TAG_subprogram
	.quad	.Lfunc_begin16                  # DW_AT_low_pc
	.long	.Lfunc_end16-.Lfunc_begin16     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string169                # DW_AT_linkage_name
	.long	.Linfo_string170                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x892:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10549                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x8a0:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10575                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x8ae:0x10 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x8b3:0x5 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.byte	30                              # Abbrev [30] 0x8b8:0x5 DW_TAG_template_type_parameter
	.long	6083                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x8bf:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin17                  # DW_AT_low_pc
	.long	.Lfunc_end17-.Lfunc_begin17     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string171                # DW_AT_linkage_name
	.long	.Linfo_string172                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x8d8:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	9545                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x8e6:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10602                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x8f4:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x8f9:0x5 DW_TAG_template_type_parameter
	.long	9381                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x900:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin18                  # DW_AT_low_pc
	.long	.Lfunc_end18-.Lfunc_begin18     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string173                # DW_AT_linkage_name
	.long	.Linfo_string174                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x919:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10624                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x927:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10645                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x935:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x93a:0x5 DW_TAG_template_type_parameter
	.long	9386                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x941:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin19                  # DW_AT_low_pc
	.long	.Lfunc_end19-.Lfunc_begin19     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string175                # DW_AT_linkage_name
	.long	.Linfo_string176                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x95a:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10667                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x968:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10688                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x976:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x97b:0x5 DW_TAG_template_type_parameter
	.long	9391                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x982:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin20                  # DW_AT_low_pc
	.long	.Lfunc_end20-.Lfunc_begin20     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string177                # DW_AT_linkage_name
	.long	.Linfo_string178                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x99b:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10710                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x9a9:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10731                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x9b7:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x9bc:0x5 DW_TAG_template_type_parameter
	.long	9396                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x9c3:0x3d DW_TAG_subprogram
	.quad	.Lfunc_begin21                  # DW_AT_low_pc
	.long	.Lfunc_end21-.Lfunc_begin21     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string179                # DW_AT_linkage_name
	.long	.Linfo_string180                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x9dc:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10753                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x9ea:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10770                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x9f8:0x7 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	31                              # Abbrev [31] 0x9fd:0x1 DW_TAG_template_type_parameter
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xa00:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin22                  # DW_AT_low_pc
	.long	.Lfunc_end22-.Lfunc_begin22     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string183                # DW_AT_linkage_name
	.long	.Linfo_string184                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0xa19:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10788                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0xa27:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10809                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0xa35:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0xa3a:0x5 DW_TAG_template_type_parameter
	.long	9410                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xa41:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin23                  # DW_AT_low_pc
	.long	.Lfunc_end23-.Lfunc_begin23     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string185                # DW_AT_linkage_name
	.long	.Linfo_string186                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0xa5a:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10831                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0xa68:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10852                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0xa76:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0xa7b:0x5 DW_TAG_template_type_parameter
	.long	897                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xa82:0x2e DW_TAG_subprogram
	.quad	.Lfunc_begin24                  # DW_AT_low_pc
	.long	.Lfunc_end24-.Lfunc_begin24     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string187                # DW_AT_linkage_name
	.long	.Linfo_string188                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	31                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	15                              # Abbrev [15] 0xa9b:0xa DW_TAG_template_value_parameter
	.long	279                             # DW_AT_type
	.long	.Linfo_string20                 # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	32                              # Abbrev [32] 0xaa5:0xa DW_TAG_template_value_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string3                  # DW_AT_name
	.byte	3                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xab0:0x35 DW_TAG_subprogram
	.quad	.Lfunc_begin25                  # DW_AT_low_pc
	.long	.Lfunc_end25-.Lfunc_begin25     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string190                # DW_AT_linkage_name
	.long	.Linfo_string191                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xac9:0x9 DW_TAG_template_type_parameter
	.long	75                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	29                              # Abbrev [29] 0xad2:0x12 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string189                # DW_AT_name
	.byte	33                              # Abbrev [33] 0xad7:0x6 DW_TAG_template_value_parameter
	.long	75                              # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	33                              # Abbrev [33] 0xadd:0x6 DW_TAG_template_value_parameter
	.long	75                              # DW_AT_type
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xae5:0x35 DW_TAG_subprogram
	.quad	.Lfunc_begin26                  # DW_AT_low_pc
	.long	.Lfunc_end26-.Lfunc_begin26     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string192                # DW_AT_linkage_name
	.long	.Linfo_string193                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xafe:0x9 DW_TAG_template_type_parameter
	.long	106                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	29                              # Abbrev [29] 0xb07:0x12 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string189                # DW_AT_name
	.byte	34                              # Abbrev [34] 0xb0c:0x6 DW_TAG_template_value_parameter
	.long	106                             # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0xb12:0x6 DW_TAG_template_value_parameter
	.long	106                             # DW_AT_type
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xb1a:0x30 DW_TAG_subprogram
	.quad	.Lfunc_begin27                  # DW_AT_low_pc
	.long	.Lfunc_end27-.Lfunc_begin27     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string194                # DW_AT_linkage_name
	.long	.Linfo_string195                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xb33:0x9 DW_TAG_template_type_parameter
	.long	137                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	29                              # Abbrev [29] 0xb3c:0xd DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string189                # DW_AT_name
	.byte	33                              # Abbrev [33] 0xb41:0x7 DW_TAG_template_value_parameter
	.long	137                             # DW_AT_type
	.ascii	"\377\001"                      # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	35                              # Abbrev [35] 0xb4a:0x35 DW_TAG_subprogram
	.quad	.Lfunc_begin28                  # DW_AT_low_pc
	.long	.Lfunc_end28-.Lfunc_begin28     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string196                # DW_AT_linkage_name
	.long	.Linfo_string197                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
	.byte	14                              # Abbrev [14] 0xb63:0x9 DW_TAG_template_type_parameter
	.long	157                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	29                              # Abbrev [29] 0xb6c:0x12 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string189                # DW_AT_name
	.byte	33                              # Abbrev [33] 0xb71:0x6 DW_TAG_template_value_parameter
	.long	157                             # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	33                              # Abbrev [33] 0xb77:0x6 DW_TAG_template_value_parameter
	.long	157                             # DW_AT_type
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xb7f:0x39 DW_TAG_subprogram
	.quad	.Lfunc_begin29                  # DW_AT_low_pc
	.long	.Lfunc_end29-.Lfunc_begin29     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string198                # DW_AT_linkage_name
	.long	.Linfo_string199                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xb98:0x9 DW_TAG_template_type_parameter
	.long	9381                            # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	29                              # Abbrev [29] 0xba1:0x16 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string189                # DW_AT_name
	.byte	36                              # Abbrev [36] 0xba6:0x10 DW_TAG_template_value_parameter
	.long	9381                            # DW_AT_type
	.byte	10                              # DW_AT_location
	.byte	3
	.quad	i
	.byte	159
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xbb8:0x2f DW_TAG_subprogram
	.quad	.Lfunc_begin30                  # DW_AT_low_pc
	.long	.Lfunc_end30-.Lfunc_begin30     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string200                # DW_AT_linkage_name
	.long	.Linfo_string201                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xbd1:0x9 DW_TAG_template_type_parameter
	.long	9381                            # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	29                              # Abbrev [29] 0xbda:0xc DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string189                # DW_AT_name
	.byte	33                              # Abbrev [33] 0xbdf:0x6 DW_TAG_template_value_parameter
	.long	9381                            # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xbe7:0x2f DW_TAG_subprogram
	.quad	.Lfunc_begin31                  # DW_AT_low_pc
	.long	.Lfunc_end31-.Lfunc_begin31     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string202                # DW_AT_linkage_name
	.long	.Linfo_string203                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xc00:0x9 DW_TAG_template_type_parameter
	.long	897                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	29                              # Abbrev [29] 0xc09:0xc DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string189                # DW_AT_name
	.byte	33                              # Abbrev [33] 0xc0e:0x6 DW_TAG_template_value_parameter
	.long	897                             # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xc16:0x2f DW_TAG_subprogram
	.quad	.Lfunc_begin32                  # DW_AT_low_pc
	.long	.Lfunc_end32-.Lfunc_begin32     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string204                # DW_AT_linkage_name
	.long	.Linfo_string205                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xc2f:0x9 DW_TAG_template_type_parameter
	.long	9336                            # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	29                              # Abbrev [29] 0xc38:0xc DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string189                # DW_AT_name
	.byte	33                              # Abbrev [33] 0xc3d:0x6 DW_TAG_template_value_parameter
	.long	9336                            # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xc45:0x2f DW_TAG_subprogram
	.quad	.Lfunc_begin33                  # DW_AT_low_pc
	.long	.Lfunc_end33-.Lfunc_begin33     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string206                # DW_AT_linkage_name
	.long	.Linfo_string207                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xc5e:0x9 DW_TAG_template_type_parameter
	.long	630                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	29                              # Abbrev [29] 0xc67:0xc DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string189                # DW_AT_name
	.byte	34                              # Abbrev [34] 0xc6c:0x6 DW_TAG_template_value_parameter
	.long	630                             # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xc74:0x2f DW_TAG_subprogram
	.quad	.Lfunc_begin34                  # DW_AT_low_pc
	.long	.Lfunc_end34-.Lfunc_begin34     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string208                # DW_AT_linkage_name
	.long	.Linfo_string209                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xc8d:0x9 DW_TAG_template_type_parameter
	.long	236                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	29                              # Abbrev [29] 0xc96:0xc DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string189                # DW_AT_name
	.byte	33                              # Abbrev [33] 0xc9b:0x6 DW_TAG_template_value_parameter
	.long	236                             # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xca3:0x2f DW_TAG_subprogram
	.quad	.Lfunc_begin35                  # DW_AT_low_pc
	.long	.Lfunc_end35-.Lfunc_begin35     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string210                # DW_AT_linkage_name
	.long	.Linfo_string211                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xcbc:0x9 DW_TAG_template_type_parameter
	.long	579                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	29                              # Abbrev [29] 0xcc5:0xc DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string189                # DW_AT_name
	.byte	34                              # Abbrev [34] 0xcca:0x6 DW_TAG_template_value_parameter
	.long	579                             # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xcd2:0x2f DW_TAG_subprogram
	.quad	.Lfunc_begin36                  # DW_AT_low_pc
	.long	.Lfunc_end36-.Lfunc_begin36     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string212                # DW_AT_linkage_name
	.long	.Linfo_string213                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xceb:0x9 DW_TAG_template_type_parameter
	.long	243                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	29                              # Abbrev [29] 0xcf4:0xc DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string189                # DW_AT_name
	.byte	33                              # Abbrev [33] 0xcf9:0x6 DW_TAG_template_value_parameter
	.long	243                             # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xd01:0x2f DW_TAG_subprogram
	.quad	.Lfunc_begin37                  # DW_AT_low_pc
	.long	.Lfunc_end37-.Lfunc_begin37     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string214                # DW_AT_linkage_name
	.long	.Linfo_string215                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xd1a:0x9 DW_TAG_template_type_parameter
	.long	550                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	29                              # Abbrev [29] 0xd23:0xc DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string189                # DW_AT_name
	.byte	34                              # Abbrev [34] 0xd28:0x6 DW_TAG_template_value_parameter
	.long	550                             # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xd30:0x35 DW_TAG_subprogram
	.quad	.Lfunc_begin38                  # DW_AT_low_pc
	.long	.Lfunc_end38-.Lfunc_begin38     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string216                # DW_AT_linkage_name
	.long	.Linfo_string217                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xd49:0x9 DW_TAG_template_type_parameter
	.long	846                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	29                              # Abbrev [29] 0xd52:0x12 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string189                # DW_AT_name
	.byte	33                              # Abbrev [33] 0xd57:0x6 DW_TAG_template_value_parameter
	.long	846                             # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	33                              # Abbrev [33] 0xd5d:0x6 DW_TAG_template_value_parameter
	.long	846                             # DW_AT_type
	.byte	2                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xd65:0x6d DW_TAG_subprogram
	.quad	.Lfunc_begin39                  # DW_AT_low_pc
	.long	.Lfunc_end39-.Lfunc_begin39     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string219                # DW_AT_linkage_name
	.long	.Linfo_string220                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xd7e:0x9 DW_TAG_template_type_parameter
	.long	9416                            # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	29                              # Abbrev [29] 0xd87:0x4a DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string189                # DW_AT_name
	.byte	34                              # Abbrev [34] 0xd8c:0x6 DW_TAG_template_value_parameter
	.long	9416                            # DW_AT_type
	.byte	0                               # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0xd92:0x6 DW_TAG_template_value_parameter
	.long	9416                            # DW_AT_type
	.byte	1                               # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0xd98:0x6 DW_TAG_template_value_parameter
	.long	9416                            # DW_AT_type
	.byte	6                               # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0xd9e:0x6 DW_TAG_template_value_parameter
	.long	9416                            # DW_AT_type
	.byte	7                               # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0xda4:0x6 DW_TAG_template_value_parameter
	.long	9416                            # DW_AT_type
	.byte	13                              # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0xdaa:0x6 DW_TAG_template_value_parameter
	.long	9416                            # DW_AT_type
	.byte	14                              # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0xdb0:0x6 DW_TAG_template_value_parameter
	.long	9416                            # DW_AT_type
	.byte	31                              # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0xdb6:0x6 DW_TAG_template_value_parameter
	.long	9416                            # DW_AT_type
	.byte	32                              # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0xdbc:0x6 DW_TAG_template_value_parameter
	.long	9416                            # DW_AT_type
	.byte	33                              # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0xdc2:0x7 DW_TAG_template_value_parameter
	.long	9416                            # DW_AT_type
	.asciz	"\377"                          # DW_AT_const_value
	.byte	34                              # Abbrev [34] 0xdc9:0x7 DW_TAG_template_value_parameter
	.long	9416                            # DW_AT_type
	.ascii	"\200\177"                      # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xdd2:0x3f DW_TAG_subprogram
	.quad	.Lfunc_begin40                  # DW_AT_low_pc
	.long	.Lfunc_end40-.Lfunc_begin40     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string222                # DW_AT_linkage_name
	.long	.Linfo_string223                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	34                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xdeb:0x9 DW_TAG_template_type_parameter
	.long	9423                            # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	29                              # Abbrev [29] 0xdf4:0x1c DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string189                # DW_AT_name
	.byte	37                              # Abbrev [37] 0xdf9:0x16 DW_TAG_template_value_parameter
	.long	9423                            # DW_AT_type
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
	.byte	10                              # Abbrev [10] 0xe11:0x29 DW_TAG_subprogram
	.quad	.Lfunc_begin41                  # DW_AT_low_pc
	.long	.Lfunc_end41-.Lfunc_begin41     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string224                # DW_AT_linkage_name
	.long	.Linfo_string225                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	37                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xe2a:0x9 DW_TAG_template_type_parameter
	.long	236                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	33                              # Abbrev [33] 0xe33:0x6 DW_TAG_template_value_parameter
	.long	236                             # DW_AT_type
	.byte	3                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xe3a:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin42                  # DW_AT_low_pc
	.long	.Lfunc_end42-.Lfunc_begin42     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string226                # DW_AT_linkage_name
	.long	.Linfo_string227                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0xe53:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10874                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0xe61:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10895                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0xe6f:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0xe74:0x5 DW_TAG_template_type_parameter
	.long	250                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xe7b:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin43                  # DW_AT_low_pc
	.long	.Lfunc_end43-.Lfunc_begin43     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string229                # DW_AT_linkage_name
	.long	.Linfo_string230                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0xe94:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10917                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0xea2:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10938                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0xeb0:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0xeb5:0x5 DW_TAG_template_type_parameter
	.long	9430                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	35                              # Abbrev [35] 0xebc:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin44                  # DW_AT_low_pc
	.long	.Lfunc_end44-.Lfunc_begin44     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string231                # DW_AT_linkage_name
	.long	.Linfo_string232                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0xed5:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	9663                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0xee3:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	10960                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0xef1:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0xef6:0x5 DW_TAG_template_type_parameter
	.long	1238                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xefd:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin45                  # DW_AT_low_pc
	.long	.Lfunc_end45-.Lfunc_begin45     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string233                # DW_AT_linkage_name
	.long	.Linfo_string234                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0xf16:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	10982                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0xf24:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11003                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0xf32:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0xf37:0x5 DW_TAG_template_type_parameter
	.long	9455                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xf3e:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin46                  # DW_AT_low_pc
	.long	.Lfunc_end46-.Lfunc_begin46     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string235                # DW_AT_linkage_name
	.long	.Linfo_string236                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0xf57:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11025                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0xf65:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11046                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0xf73:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0xf78:0x5 DW_TAG_template_type_parameter
	.long	9466                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xf7f:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin47                  # DW_AT_low_pc
	.long	.Lfunc_end47-.Lfunc_begin47     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string237                # DW_AT_linkage_name
	.long	.Linfo_string238                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0xf98:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11068                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0xfa6:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11089                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0xfb4:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0xfb9:0x5 DW_TAG_template_type_parameter
	.long	9471                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	35                              # Abbrev [35] 0xfc0:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin48                  # DW_AT_low_pc
	.long	.Lfunc_end48-.Lfunc_begin48     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string240                # DW_AT_linkage_name
	.long	.Linfo_string241                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0xfd9:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11111                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0xfe7:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11132                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0xff5:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0xffa:0x5 DW_TAG_template_type_parameter
	.long	9482                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1001:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin49                  # DW_AT_low_pc
	.long	.Lfunc_end49-.Lfunc_begin49     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string243                # DW_AT_linkage_name
	.long	.Linfo_string244                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x101a:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11154                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1028:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11175                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1036:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x103b:0x5 DW_TAG_template_type_parameter
	.long	9488                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1042:0x46 DW_TAG_subprogram
	.quad	.Lfunc_begin50                  # DW_AT_low_pc
	.long	.Lfunc_end50-.Lfunc_begin50     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string245                # DW_AT_linkage_name
	.long	.Linfo_string246                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x105b:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11197                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1069:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11223                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1077:0x10 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x107c:0x5 DW_TAG_template_type_parameter
	.long	9493                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1081:0x5 DW_TAG_template_type_parameter
	.long	9493                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1088:0x46 DW_TAG_subprogram
	.quad	.Lfunc_begin51                  # DW_AT_low_pc
	.long	.Lfunc_end51-.Lfunc_begin51     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string247                # DW_AT_linkage_name
	.long	.Linfo_string248                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x10a1:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11250                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x10af:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11276                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x10bd:0x10 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x10c2:0x5 DW_TAG_template_type_parameter
	.long	9493                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x10c7:0x5 DW_TAG_template_type_parameter
	.long	9498                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x10ce:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin52                  # DW_AT_low_pc
	.long	.Lfunc_end52-.Lfunc_begin52     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string249                # DW_AT_linkage_name
	.long	.Linfo_string250                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x10e7:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11303                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x10f5:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11324                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1103:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1108:0x5 DW_TAG_template_type_parameter
	.long	9503                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x110f:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin53                  # DW_AT_low_pc
	.long	.Lfunc_end53-.Lfunc_begin53     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string251                # DW_AT_linkage_name
	.long	.Linfo_string252                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1128:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11346                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1136:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11367                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1144:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1149:0x5 DW_TAG_template_type_parameter
	.long	9508                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1150:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin54                  # DW_AT_low_pc
	.long	.Lfunc_end54-.Lfunc_begin54     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string253                # DW_AT_linkage_name
	.long	.Linfo_string254                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1169:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11389                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1177:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11410                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1185:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x118a:0x5 DW_TAG_template_type_parameter
	.long	9524                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1191:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin55                  # DW_AT_low_pc
	.long	.Lfunc_end55-.Lfunc_begin55     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string255                # DW_AT_linkage_name
	.long	.Linfo_string256                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x11aa:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11432                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x11b8:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11453                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x11c6:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x11cb:0x5 DW_TAG_template_type_parameter
	.long	9525                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	35                              # Abbrev [35] 0x11d2:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin56                  # DW_AT_low_pc
	.long	.Lfunc_end56-.Lfunc_begin56     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string257                # DW_AT_linkage_name
	.long	.Linfo_string258                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x11eb:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11475                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x11f9:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11496                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1207:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x120c:0x5 DW_TAG_template_type_parameter
	.long	9530                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	35                              # Abbrev [35] 0x1213:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin57                  # DW_AT_low_pc
	.long	.Lfunc_end57-.Lfunc_begin57     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string259                # DW_AT_linkage_name
	.long	.Linfo_string260                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x122c:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11518                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x123a:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11539                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1248:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x124d:0x5 DW_TAG_template_type_parameter
	.long	1243                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	35                              # Abbrev [35] 0x1254:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin58                  # DW_AT_low_pc
	.long	.Lfunc_end58-.Lfunc_begin58     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string261                # DW_AT_linkage_name
	.long	.Linfo_string262                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x126d:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11561                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x127b:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11582                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1289:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x128e:0x5 DW_TAG_template_type_parameter
	.long	9535                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1295:0x2e DW_TAG_subprogram
	.quad	.Lfunc_begin59                  # DW_AT_low_pc
	.long	.Lfunc_end59-.Lfunc_begin59     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string265                # DW_AT_linkage_name
	.long	.Linfo_string266                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	54                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	29                              # Abbrev [29] 0x12ae:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string263                # DW_AT_name
	.byte	30                              # Abbrev [30] 0x12b3:0x5 DW_TAG_template_type_parameter
	.long	9360                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	14                              # Abbrev [14] 0x12b9:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string264                # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x12c3:0x28 DW_TAG_subprogram
	.quad	.Lfunc_begin60                  # DW_AT_low_pc
	.long	.Lfunc_end60-.Lfunc_begin60     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string267                # DW_AT_linkage_name
	.long	.Linfo_string268                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	54                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	38                              # Abbrev [38] 0x12dc:0x5 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string263                # DW_AT_name
	.byte	14                              # Abbrev [14] 0x12e1:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string264                # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x12eb:0x28 DW_TAG_subprogram
	.quad	.Lfunc_begin61                  # DW_AT_low_pc
	.long	.Lfunc_end61-.Lfunc_begin61     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string269                # DW_AT_linkage_name
	.long	.Linfo_string270                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	56                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x1304:0x9 DW_TAG_template_type_parameter
	.long	9360                            # DW_AT_type
	.long	.Linfo_string263                # DW_AT_name
	.byte	38                              # Abbrev [38] 0x130d:0x5 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string264                # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1313:0x3b DW_TAG_subprogram
	.quad	.Lfunc_begin62                  # DW_AT_low_pc
	.long	.Lfunc_end62-.Lfunc_begin62     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string271                # DW_AT_linkage_name
	.long	.Linfo_string272                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x132c:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11604                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x133a:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11619                           # DW_AT_type
	.byte	38                              # Abbrev [38] 0x1348:0x5 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x134e:0x46 DW_TAG_subprogram
	.quad	.Lfunc_begin63                  # DW_AT_low_pc
	.long	.Lfunc_end63-.Lfunc_begin63     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string273                # DW_AT_linkage_name
	.long	.Linfo_string274                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1367:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11635                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1375:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11661                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1383:0x10 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1388:0x5 DW_TAG_template_type_parameter
	.long	9518                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x138d:0x5 DW_TAG_template_type_parameter
	.long	9518                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1394:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin64                  # DW_AT_low_pc
	.long	.Lfunc_end64-.Lfunc_begin64     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string276                # DW_AT_linkage_name
	.long	.Linfo_string277                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x13ad:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11688                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x13bb:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11709                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x13c9:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x13ce:0x5 DW_TAG_template_type_parameter
	.long	9540                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x13d5:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin65                  # DW_AT_low_pc
	.long	.Lfunc_end65-.Lfunc_begin65     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string279                # DW_AT_linkage_name
	.long	.Linfo_string280                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x13ee:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11731                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x13fc:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11752                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x140a:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x140f:0x5 DW_TAG_template_type_parameter
	.long	9566                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x1416:0x2e9 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string75                 # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	57                              # DW_AT_decl_line
	.byte	39                              # Abbrev [39] 0x141f:0x1f DW_TAG_subprogram
	.long	.Linfo_string76                 # DW_AT_linkage_name
	.long	.Linfo_string77                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	59                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x142a:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x1433:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	40                              # Abbrev [40] 0x1438:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x143e:0x1f DW_TAG_subprogram
	.long	.Linfo_string78                 # DW_AT_linkage_name
	.long	.Linfo_string79                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	62                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x1449:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x1452:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	40                              # Abbrev [40] 0x1457:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x145d:0x1f DW_TAG_subprogram
	.long	.Linfo_string80                 # DW_AT_linkage_name
	.long	.Linfo_string81                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	65                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x1468:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x1471:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	40                              # Abbrev [40] 0x1476:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	41                              # Abbrev [41] 0x147c:0x1e DW_TAG_subprogram
	.long	.Linfo_string82                 # DW_AT_linkage_name
	.long	.Linfo_string83                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	68                              # DW_AT_decl_line
	.long	6057                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x148b:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x1494:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x149a:0x1f DW_TAG_subprogram
	.long	.Linfo_string87                 # DW_AT_linkage_name
	.long	.Linfo_string88                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	72                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x14a5:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x14ae:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	40                              # Abbrev [40] 0x14b3:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x14b9:0x1f DW_TAG_subprogram
	.long	.Linfo_string89                 # DW_AT_linkage_name
	.long	.Linfo_string90                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	75                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x14c4:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x14cd:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	40                              # Abbrev [40] 0x14d2:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x14d8:0x1f DW_TAG_subprogram
	.long	.Linfo_string91                 # DW_AT_linkage_name
	.long	.Linfo_string92                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	78                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x14e3:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x14ec:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	40                              # Abbrev [40] 0x14f1:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x14f7:0x1f DW_TAG_subprogram
	.long	.Linfo_string93                 # DW_AT_linkage_name
	.long	.Linfo_string94                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	81                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x1502:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x150b:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	40                              # Abbrev [40] 0x1510:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x1516:0x1f DW_TAG_subprogram
	.long	.Linfo_string95                 # DW_AT_linkage_name
	.long	.Linfo_string96                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	84                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x1521:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x152a:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	40                              # Abbrev [40] 0x152f:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x1535:0x1f DW_TAG_subprogram
	.long	.Linfo_string97                 # DW_AT_linkage_name
	.long	.Linfo_string98                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	87                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x1540:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x1549:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	40                              # Abbrev [40] 0x154e:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x1554:0x1f DW_TAG_subprogram
	.long	.Linfo_string99                 # DW_AT_linkage_name
	.long	.Linfo_string100                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	90                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x155f:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x1568:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	40                              # Abbrev [40] 0x156d:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x1573:0x1a DW_TAG_subprogram
	.long	.Linfo_string101                # DW_AT_linkage_name
	.long	.Linfo_string102                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	93                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x157e:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x1587:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x158d:0x1a DW_TAG_subprogram
	.long	.Linfo_string103                # DW_AT_linkage_name
	.long	.Linfo_string104                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	96                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x1598:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x15a1:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x15a7:0x1f DW_TAG_subprogram
	.long	.Linfo_string105                # DW_AT_linkage_name
	.long	.Linfo_string106                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	99                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x15b2:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x15bb:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	40                              # Abbrev [40] 0x15c0:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x15c6:0x1f DW_TAG_subprogram
	.long	.Linfo_string107                # DW_AT_linkage_name
	.long	.Linfo_string108                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	102                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x15d1:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x15da:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	40                              # Abbrev [40] 0x15df:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x15e5:0x1f DW_TAG_subprogram
	.long	.Linfo_string109                # DW_AT_linkage_name
	.long	.Linfo_string110                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	105                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x15f0:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x15f9:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	40                              # Abbrev [40] 0x15fe:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x1604:0x1a DW_TAG_subprogram
	.long	.Linfo_string111                # DW_AT_linkage_name
	.long	.Linfo_string112                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	108                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x160f:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x1618:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x161e:0x1f DW_TAG_subprogram
	.long	.Linfo_string113                # DW_AT_linkage_name
	.long	.Linfo_string114                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	111                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x1629:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x1632:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	40                              # Abbrev [40] 0x1637:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x163d:0x1f DW_TAG_subprogram
	.long	.Linfo_string115                # DW_AT_linkage_name
	.long	.Linfo_string116                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	114                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x1648:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x1651:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	40                              # Abbrev [40] 0x1656:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	41                              # Abbrev [41] 0x165c:0x23 DW_TAG_subprogram
	.long	.Linfo_string117                # DW_AT_linkage_name
	.long	.Linfo_string118                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	117                             # DW_AT_decl_line
	.long	6930                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x166b:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	40                              # Abbrev [40] 0x1674:0x5 DW_TAG_formal_parameter
	.long	515                             # DW_AT_type
	.byte	40                              # Abbrev [40] 0x1679:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	41                              # Abbrev [41] 0x167f:0x23 DW_TAG_subprogram
	.long	.Linfo_string120                # DW_AT_linkage_name
	.long	.Linfo_string121                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	124                             # DW_AT_decl_line
	.long	6930                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x168e:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	40                              # Abbrev [40] 0x1697:0x5 DW_TAG_formal_parameter
	.long	515                             # DW_AT_type
	.byte	40                              # Abbrev [40] 0x169c:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x16a2:0x1f DW_TAG_subprogram
	.long	.Linfo_string122                # DW_AT_linkage_name
	.long	.Linfo_string123                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	121                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x16ad:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	40                              # Abbrev [40] 0x16b6:0x5 DW_TAG_formal_parameter
	.long	6930                            # DW_AT_type
	.byte	40                              # Abbrev [40] 0x16bb:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	39                              # Abbrev [39] 0x16c1:0x1f DW_TAG_subprogram
	.long	.Linfo_string124                # DW_AT_linkage_name
	.long	.Linfo_string125                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	128                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x16cc:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	40                              # Abbrev [40] 0x16d5:0x5 DW_TAG_formal_parameter
	.long	6930                            # DW_AT_type
	.byte	40                              # Abbrev [40] 0x16da:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	41                              # Abbrev [41] 0x16e0:0x1e DW_TAG_subprogram
	.long	.Linfo_string126                # DW_AT_linkage_name
	.long	.Linfo_string127                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	131                             # DW_AT_decl_line
	.long	63                              # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x16ef:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	18                              # Abbrev [18] 0x16f8:0x5 DW_TAG_formal_parameter
	.long	5887                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x16ff:0x5 DW_TAG_pointer_type
	.long	5142                            # DW_AT_type
	.byte	43                              # Abbrev [43] 0x1704:0x37 DW_TAG_subprogram
	.quad	.Lfunc_begin66                  # DW_AT_low_pc
	.long	.Lfunc_end66-.Lfunc_begin66     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	5915                            # DW_AT_object_pointer
	.long	5151                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x171b:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x1727:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	59                              # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0x1731:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	43                              # Abbrev [43] 0x173b:0x37 DW_TAG_subprogram
	.quad	.Lfunc_begin67                  # DW_AT_low_pc
	.long	.Lfunc_end67-.Lfunc_begin67     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	5970                            # DW_AT_object_pointer
	.long	5182                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x1752:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x175e:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	62                              # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0x1768:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	43                              # Abbrev [43] 0x1772:0x37 DW_TAG_subprogram
	.quad	.Lfunc_begin68                  # DW_AT_low_pc
	.long	.Lfunc_end68-.Lfunc_begin68     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6025                            # DW_AT_object_pointer
	.long	5213                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x1789:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x1795:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	65                              # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0x179f:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x17a9:0x5 DW_TAG_pointer_type
	.long	6062                            # DW_AT_type
	.byte	13                              # Abbrev [13] 0x17ae:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string86                 # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x17b7:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x17bc:0x5 DW_TAG_template_type_parameter
	.long	6083                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0x17c3:0x7 DW_TAG_base_type
	.long	.Linfo_string85                 # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	43                              # Abbrev [43] 0x17ca:0x2d DW_TAG_subprogram
	.quad	.Lfunc_begin69                  # DW_AT_low_pc
	.long	.Lfunc_end69-.Lfunc_begin69     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6113                            # DW_AT_object_pointer
	.long	5244                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x17e1:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	14                              # Abbrev [14] 0x17ed:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	43                              # Abbrev [43] 0x17f7:0x37 DW_TAG_subprogram
	.quad	.Lfunc_begin70                  # DW_AT_low_pc
	.long	.Lfunc_end70-.Lfunc_begin70     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6158                            # DW_AT_object_pointer
	.long	5274                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x180e:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x181a:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	72                              # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0x1824:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	43                              # Abbrev [43] 0x182e:0x37 DW_TAG_subprogram
	.quad	.Lfunc_begin71                  # DW_AT_low_pc
	.long	.Lfunc_end71-.Lfunc_begin71     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6213                            # DW_AT_object_pointer
	.long	5305                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x1845:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x1851:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	75                              # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0x185b:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	43                              # Abbrev [43] 0x1865:0x37 DW_TAG_subprogram
	.quad	.Lfunc_begin72                  # DW_AT_low_pc
	.long	.Lfunc_end72-.Lfunc_begin72     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6268                            # DW_AT_object_pointer
	.long	5336                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x187c:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x1888:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	78                              # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0x1892:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	43                              # Abbrev [43] 0x189c:0x37 DW_TAG_subprogram
	.quad	.Lfunc_begin73                  # DW_AT_low_pc
	.long	.Lfunc_end73-.Lfunc_begin73     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6323                            # DW_AT_object_pointer
	.long	5367                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x18b3:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x18bf:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	81                              # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0x18c9:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	43                              # Abbrev [43] 0x18d3:0x37 DW_TAG_subprogram
	.quad	.Lfunc_begin74                  # DW_AT_low_pc
	.long	.Lfunc_end74-.Lfunc_begin74     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6378                            # DW_AT_object_pointer
	.long	5398                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x18ea:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x18f6:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	84                              # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0x1900:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	43                              # Abbrev [43] 0x190a:0x37 DW_TAG_subprogram
	.quad	.Lfunc_begin75                  # DW_AT_low_pc
	.long	.Lfunc_end75-.Lfunc_begin75     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6433                            # DW_AT_object_pointer
	.long	5429                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x1921:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x192d:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	87                              # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0x1937:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	43                              # Abbrev [43] 0x1941:0x37 DW_TAG_subprogram
	.quad	.Lfunc_begin76                  # DW_AT_low_pc
	.long	.Lfunc_end76-.Lfunc_begin76     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6488                            # DW_AT_object_pointer
	.long	5460                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x1958:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x1964:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	90                              # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0x196e:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	43                              # Abbrev [43] 0x1978:0x2d DW_TAG_subprogram
	.quad	.Lfunc_begin77                  # DW_AT_low_pc
	.long	.Lfunc_end77-.Lfunc_begin77     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6543                            # DW_AT_object_pointer
	.long	5491                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x198f:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	14                              # Abbrev [14] 0x199b:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	43                              # Abbrev [43] 0x19a5:0x2d DW_TAG_subprogram
	.quad	.Lfunc_begin78                  # DW_AT_low_pc
	.long	.Lfunc_end78-.Lfunc_begin78     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6588                            # DW_AT_object_pointer
	.long	5517                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x19bc:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	14                              # Abbrev [14] 0x19c8:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	43                              # Abbrev [43] 0x19d2:0x37 DW_TAG_subprogram
	.quad	.Lfunc_begin79                  # DW_AT_low_pc
	.long	.Lfunc_end79-.Lfunc_begin79     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6633                            # DW_AT_object_pointer
	.long	5543                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x19e9:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x19f5:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	99                              # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0x19ff:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	43                              # Abbrev [43] 0x1a09:0x37 DW_TAG_subprogram
	.quad	.Lfunc_begin80                  # DW_AT_low_pc
	.long	.Lfunc_end80-.Lfunc_begin80     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6688                            # DW_AT_object_pointer
	.long	5574                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x1a20:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x1a2c:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	102                             # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0x1a36:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	43                              # Abbrev [43] 0x1a40:0x37 DW_TAG_subprogram
	.quad	.Lfunc_begin81                  # DW_AT_low_pc
	.long	.Lfunc_end81-.Lfunc_begin81     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6743                            # DW_AT_object_pointer
	.long	5605                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x1a57:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x1a63:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	105                             # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0x1a6d:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	43                              # Abbrev [43] 0x1a77:0x2d DW_TAG_subprogram
	.quad	.Lfunc_begin82                  # DW_AT_low_pc
	.long	.Lfunc_end82-.Lfunc_begin82     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6798                            # DW_AT_object_pointer
	.long	5636                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x1a8e:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	14                              # Abbrev [14] 0x1a9a:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	43                              # Abbrev [43] 0x1aa4:0x37 DW_TAG_subprogram
	.quad	.Lfunc_begin83                  # DW_AT_low_pc
	.long	.Lfunc_end83-.Lfunc_begin83     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6843                            # DW_AT_object_pointer
	.long	5662                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x1abb:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x1ac7:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	111                             # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0x1ad1:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	43                              # Abbrev [43] 0x1adb:0x37 DW_TAG_subprogram
	.quad	.Lfunc_begin84                  # DW_AT_low_pc
	.long	.Lfunc_end84-.Lfunc_begin84     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	6898                            # DW_AT_object_pointer
	.long	5693                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x1af2:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	11774                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	22                              # Abbrev [22] 0x1afe:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	114                             # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0x1b08:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	45                              # Abbrev [45] 0x1b12:0x1 DW_TAG_pointer_type
	.byte	46                              # Abbrev [46] 0x1b13:0x1d DW_TAG_subprogram
	.quad	.Lfunc_begin85                  # DW_AT_low_pc
	.long	.Lfunc_end85-.Lfunc_begin85     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	5724                            # DW_AT_specification
	.byte	14                              # Abbrev [14] 0x1b26:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x1b30:0x1d DW_TAG_subprogram
	.quad	.Lfunc_begin86                  # DW_AT_low_pc
	.long	.Lfunc_end86-.Lfunc_begin86     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	5759                            # DW_AT_specification
	.byte	14                              # Abbrev [14] 0x1b43:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x1b4d:0x31 DW_TAG_subprogram
	.quad	.Lfunc_begin87                  # DW_AT_low_pc
	.long	.Lfunc_end87-.Lfunc_begin87     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	5794                            # DW_AT_specification
	.byte	22                              # Abbrev [22] 0x1b60:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	1                               # DW_AT_decl_file
	.byte	121                             # DW_AT_decl_line
	.long	6930                            # DW_AT_type
	.byte	22                              # Abbrev [22] 0x1b6a:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	121                             # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0x1b74:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x1b7e:0x31 DW_TAG_subprogram
	.quad	.Lfunc_begin88                  # DW_AT_low_pc
	.long	.Lfunc_end88-.Lfunc_begin88     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	5825                            # DW_AT_specification
	.byte	22                              # Abbrev [22] 0x1b91:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	1                               # DW_AT_decl_file
	.byte	128                             # DW_AT_decl_line
	.long	6930                            # DW_AT_type
	.byte	22                              # Abbrev [22] 0x1b9b:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	1                               # DW_AT_decl_file
	.byte	128                             # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0x1ba5:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	46                              # Abbrev [46] 0x1baf:0x1d DW_TAG_subprogram
	.quad	.Lfunc_begin89                  # DW_AT_low_pc
	.long	.Lfunc_end89-.Lfunc_begin89     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	5856                            # DW_AT_specification
	.byte	14                              # Abbrev [14] 0x1bc2:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	35                              # Abbrev [35] 0x1bcc:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin90                  # DW_AT_low_pc
	.long	.Lfunc_end90-.Lfunc_begin90     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string282                # DW_AT_linkage_name
	.long	.Linfo_string283                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x1be5:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11779                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1bf3:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11800                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1c01:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1c06:0x5 DW_TAG_template_type_parameter
	.long	1248                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1c0d:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin91                  # DW_AT_low_pc
	.long	.Lfunc_end91-.Lfunc_begin91     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string284                # DW_AT_linkage_name
	.long	.Linfo_string285                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1c26:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11822                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1c34:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11843                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1c42:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1c47:0x5 DW_TAG_template_type_parameter
	.long	9584                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1c4e:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin92                  # DW_AT_low_pc
	.long	.Lfunc_end92-.Lfunc_begin92     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string286                # DW_AT_linkage_name
	.long	.Linfo_string287                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1c67:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11865                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1c75:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11886                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1c83:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1c88:0x5 DW_TAG_template_type_parameter
	.long	9601                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1c8f:0x23 DW_TAG_subprogram
	.quad	.Lfunc_begin93                  # DW_AT_low_pc
	.long	.Lfunc_end93-.Lfunc_begin93     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string289                # DW_AT_linkage_name
	.long	.Linfo_string290                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	135                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	11                              # Abbrev [11] 0x1ca8:0x9 DW_TAG_GNU_template_template_param
	.long	.Linfo_string18                 # DW_AT_name
	.long	.Linfo_string288                # DW_AT_GNU_template_name
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1cb2:0x2c DW_TAG_subprogram
	.quad	.Lfunc_begin94                  # DW_AT_low_pc
	.long	.Lfunc_end94-.Lfunc_begin94     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string291                # DW_AT_linkage_name
	.long	.Linfo_string292                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	136                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	11                              # Abbrev [11] 0x1ccb:0x9 DW_TAG_GNU_template_template_param
	.long	.Linfo_string18                 # DW_AT_name
	.long	.Linfo_string288                # DW_AT_GNU_template_name
	.byte	14                              # Abbrev [14] 0x1cd4:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string264                # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1cde:0x46 DW_TAG_subprogram
	.quad	.Lfunc_begin96                  # DW_AT_low_pc
	.long	.Lfunc_end96-.Lfunc_begin96     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string296                # DW_AT_linkage_name
	.long	.Linfo_string297                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1cf7:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11908                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1d05:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11934                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1d13:0x10 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1d18:0x5 DW_TAG_template_type_parameter
	.long	9381                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1d1d:0x5 DW_TAG_template_type_parameter
	.long	9606                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1d24:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin97                  # DW_AT_low_pc
	.long	.Lfunc_end97-.Lfunc_begin97     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string299                # DW_AT_linkage_name
	.long	.Linfo_string300                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1d3d:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	11961                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1d4b:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	11982                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1d59:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1d5e:0x5 DW_TAG_template_type_parameter
	.long	9611                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1d65:0x23 DW_TAG_subprogram
	.quad	.Lfunc_begin98                  # DW_AT_low_pc
	.long	.Lfunc_end98-.Lfunc_begin98     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string302                # DW_AT_linkage_name
	.long	.Linfo_string303                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	135                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	11                              # Abbrev [11] 0x1d7e:0x9 DW_TAG_GNU_template_template_param
	.long	.Linfo_string18                 # DW_AT_name
	.long	.Linfo_string301                # DW_AT_GNU_template_name
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1d88:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin99                  # DW_AT_low_pc
	.long	.Lfunc_end99-.Lfunc_begin99     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string304                # DW_AT_linkage_name
	.long	.Linfo_string305                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1da1:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12004                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1daf:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12025                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1dbd:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1dc2:0x5 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1dc9:0x4b DW_TAG_subprogram
	.quad	.Lfunc_begin100                 # DW_AT_low_pc
	.long	.Lfunc_end100-.Lfunc_begin100   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string306                # DW_AT_linkage_name
	.long	.Linfo_string307                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1de2:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12047                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1df0:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12078                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1dfe:0x15 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1e03:0x5 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1e08:0x5 DW_TAG_template_type_parameter
	.long	630                             # DW_AT_type
	.byte	30                              # Abbrev [30] 0x1e0d:0x5 DW_TAG_template_type_parameter
	.long	9630                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1e14:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin101                 # DW_AT_low_pc
	.long	.Lfunc_end101-.Lfunc_begin101   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string308                # DW_AT_linkage_name
	.long	.Linfo_string309                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1e2d:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12110                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1e3b:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12131                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1e49:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1e4e:0x5 DW_TAG_template_type_parameter
	.long	9635                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1e55:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin102                 # DW_AT_low_pc
	.long	.Lfunc_end102-.Lfunc_begin102   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string310                # DW_AT_linkage_name
	.long	.Linfo_string311                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1e6e:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12153                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1e7c:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12174                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1e8a:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1e8f:0x5 DW_TAG_template_type_parameter
	.long	9647                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1e96:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin103                 # DW_AT_low_pc
	.long	.Lfunc_end103-.Lfunc_begin103   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string312                # DW_AT_linkage_name
	.long	.Linfo_string313                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1eaf:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12196                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1ebd:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12217                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1ecb:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1ed0:0x5 DW_TAG_template_type_parameter
	.long	9657                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	35                              # Abbrev [35] 0x1ed7:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin104                 # DW_AT_low_pc
	.long	.Lfunc_end104-.Lfunc_begin104   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string315                # DW_AT_linkage_name
	.long	.Linfo_string316                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x1ef0:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12239                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1efe:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12260                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1f0c:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1f11:0x5 DW_TAG_template_type_parameter
	.long	9663                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x1f18:0x5 DW_TAG_pointer_type
	.long	286                             # DW_AT_type
	.byte	47                              # Abbrev [47] 0x1f1d:0x2d DW_TAG_subprogram
	.quad	.Lfunc_begin105                 # DW_AT_low_pc
	.long	.Lfunc_end105-.Lfunc_begin105   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	7992                            # DW_AT_object_pointer
	.long	.Linfo_string317                # DW_AT_linkage_name
	.long	295                             # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x1f38:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	12282                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	17                              # Abbrev [17] 0x1f44:0x5 DW_TAG_template_type_parameter
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1f4a:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin106                 # DW_AT_low_pc
	.long	.Lfunc_end106-.Lfunc_begin106   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string318                # DW_AT_linkage_name
	.long	.Linfo_string319                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1f63:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12287                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1f71:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12308                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1f7f:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1f84:0x5 DW_TAG_template_type_parameter
	.long	9684                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1f8b:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin107                 # DW_AT_low_pc
	.long	.Lfunc_end107-.Lfunc_begin107   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string320                # DW_AT_linkage_name
	.long	.Linfo_string321                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1fa4:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12330                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1fb2:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12351                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x1fc0:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x1fc5:0x5 DW_TAG_template_type_parameter
	.long	9710                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1fcc:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin108                 # DW_AT_low_pc
	.long	.Lfunc_end108-.Lfunc_begin108   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string322                # DW_AT_linkage_name
	.long	.Linfo_string323                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x1fe5:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12373                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1ff3:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12394                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x2001:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2006:0x5 DW_TAG_template_type_parameter
	.long	9736                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	48                              # Abbrev [48] 0x200d:0x27 DW_TAG_subprogram
	.quad	.Lfunc_begin109                 # DW_AT_low_pc
	.long	.Lfunc_end109-.Lfunc_begin109   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string324                # DW_AT_linkage_name
	.long	.Linfo_string325                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	154                             # DW_AT_decl_line
	.long	9525                            # DW_AT_type
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x202a:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x2034:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin110                 # DW_AT_low_pc
	.long	.Lfunc_end110-.Lfunc_begin110   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string326                # DW_AT_linkage_name
	.long	.Linfo_string327                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x204d:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12416                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x205b:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12437                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x2069:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x206e:0x5 DW_TAG_template_type_parameter
	.long	9762                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x2075:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin111                 # DW_AT_low_pc
	.long	.Lfunc_end111-.Lfunc_begin111   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string328                # DW_AT_linkage_name
	.long	.Linfo_string329                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x208e:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12459                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x209c:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12480                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x20aa:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x20af:0x5 DW_TAG_template_type_parameter
	.long	9767                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x20b6:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin112                 # DW_AT_low_pc
	.long	.Lfunc_end112-.Lfunc_begin112   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string330                # DW_AT_linkage_name
	.long	.Linfo_string331                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x20cf:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12502                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x20dd:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12523                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x20eb:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x20f0:0x5 DW_TAG_template_type_parameter
	.long	9789                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x20f7:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin113                 # DW_AT_low_pc
	.long	.Lfunc_end113-.Lfunc_begin113   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string332                # DW_AT_linkage_name
	.long	.Linfo_string333                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x2110:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12545                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x211e:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12566                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x212c:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2131:0x5 DW_TAG_template_type_parameter
	.long	9795                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x2138:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin114                 # DW_AT_low_pc
	.long	.Lfunc_end114-.Lfunc_begin114   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string334                # DW_AT_linkage_name
	.long	.Linfo_string335                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x2151:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12588                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x215f:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12609                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x216d:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2172:0x5 DW_TAG_template_type_parameter
	.long	9801                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x2179:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin115                 # DW_AT_low_pc
	.long	.Lfunc_end115-.Lfunc_begin115   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string336                # DW_AT_linkage_name
	.long	.Linfo_string337                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x2192:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12631                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x21a0:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12652                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x21ae:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x21b3:0x5 DW_TAG_template_type_parameter
	.long	9811                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x21ba:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin116                 # DW_AT_low_pc
	.long	.Lfunc_end116-.Lfunc_begin116   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string338                # DW_AT_linkage_name
	.long	.Linfo_string339                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x21d3:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12674                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x21e1:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12695                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x21ef:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x21f4:0x5 DW_TAG_template_type_parameter
	.long	9828                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x21fb:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin117                 # DW_AT_low_pc
	.long	.Lfunc_end117-.Lfunc_begin117   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string340                # DW_AT_linkage_name
	.long	.Linfo_string341                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x2214:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12717                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x2222:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12738                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x2230:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2235:0x5 DW_TAG_template_type_parameter
	.long	9833                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x223c:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin118                 # DW_AT_low_pc
	.long	.Lfunc_end118-.Lfunc_begin118   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string342                # DW_AT_linkage_name
	.long	.Linfo_string343                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x2255:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12760                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x2263:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12781                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x2271:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2276:0x5 DW_TAG_template_type_parameter
	.long	9864                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x227d:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin119                 # DW_AT_low_pc
	.long	.Lfunc_end119-.Lfunc_begin119   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string344                # DW_AT_linkage_name
	.long	.Linfo_string345                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x2296:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12803                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x22a4:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12824                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x22b2:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x22b7:0x5 DW_TAG_template_type_parameter
	.long	9525                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	35                              # Abbrev [35] 0x22be:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin120                 # DW_AT_low_pc
	.long	.Lfunc_end120-.Lfunc_begin120   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string346                # DW_AT_linkage_name
	.long	.Linfo_string347                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x22d7:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12846                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x22e5:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12867                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x22f3:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x22f8:0x5 DW_TAG_template_type_parameter
	.long	9887                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	35                              # Abbrev [35] 0x22ff:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin121                 # DW_AT_low_pc
	.long	.Lfunc_end121-.Lfunc_begin121   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string348                # DW_AT_linkage_name
	.long	.Linfo_string349                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x2318:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12889                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x2326:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12910                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x2334:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2339:0x5 DW_TAG_template_type_parameter
	.long	9894                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	35                              # Abbrev [35] 0x2340:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin122                 # DW_AT_low_pc
	.long	.Lfunc_end122-.Lfunc_begin122   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string350                # DW_AT_linkage_name
	.long	.Linfo_string351                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x2359:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12932                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x2367:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	12953                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x2375:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x237a:0x5 DW_TAG_template_type_parameter
	.long	9906                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x2381:0x23 DW_TAG_subprogram
	.quad	.Lfunc_begin123                 # DW_AT_low_pc
	.long	.Lfunc_end123-.Lfunc_begin123   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string352                # DW_AT_linkage_name
	.long	.Linfo_string353                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	163                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x239a:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x23a4:0x1b DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string131                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	145                             # DW_AT_decl_line
	.byte	39                              # Abbrev [39] 0x23ad:0x11 DW_TAG_subprogram
	.long	.Linfo_string129                # DW_AT_linkage_name
	.long	.Linfo_string130                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	146                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	18                              # Abbrev [18] 0x23b8:0x5 DW_TAG_formal_parameter
	.long	9151                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x23bf:0x5 DW_TAG_pointer_type
	.long	9124                            # DW_AT_type
	.byte	49                              # Abbrev [49] 0x23c4:0x2b DW_TAG_subprogram
	.quad	.Lfunc_begin124                 # DW_AT_low_pc
	.long	.Lfunc_end124-.Lfunc_begin124   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	9181                            # DW_AT_object_pointer
	.short	302                             # DW_AT_decl_line
	.long	9133                            # DW_AT_specification
	.byte	44                              # Abbrev [44] 0x23dd:0xc DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string449                # DW_AT_name
	.long	12975                           # DW_AT_type
                                        # DW_AT_artificial
	.byte	12                              # Abbrev [12] 0x23e9:0x5 DW_TAG_structure_type
	.long	.Linfo_string281                # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	35                              # Abbrev [35] 0x23ef:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin125                 # DW_AT_low_pc
	.long	.Lfunc_end125-.Lfunc_begin125   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string354                # DW_AT_linkage_name
	.long	.Linfo_string283                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x2408:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	12980                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x2416:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	13001                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x2424:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2429:0x5 DW_TAG_template_type_parameter
	.long	9193                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x2430:0x41 DW_TAG_subprogram
	.quad	.Lfunc_begin126                 # DW_AT_low_pc
	.long	.Lfunc_end126-.Lfunc_begin126   # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string355                # DW_AT_linkage_name
	.long	.Linfo_string356                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	24                              # Abbrev [24] 0x2449:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string361                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	13023                           # DW_AT_type
	.byte	24                              # Abbrev [24] 0x2457:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string358                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	13044                           # DW_AT_type
	.byte	29                              # Abbrev [29] 0x2465:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x246a:0x5 DW_TAG_template_type_parameter
	.long	9913                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0x2471:0x7 DW_TAG_base_type
	.long	.Linfo_string141                # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x2478:0x7 DW_TAG_base_type
	.long	.Linfo_string150                # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x247f:0x7 DW_TAG_base_type
	.long	.Linfo_string153                # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	12                              # Abbrev [12] 0x2486:0x5 DW_TAG_structure_type
	.long	.Linfo_string156                # DW_AT_name
                                        # DW_AT_declaration
	.byte	42                              # Abbrev [42] 0x248b:0x5 DW_TAG_pointer_type
	.long	219                             # DW_AT_type
	.byte	13                              # Abbrev [13] 0x2490:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string166                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2499:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x249e:0x5 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x24a5:0x5 DW_TAG_pointer_type
	.long	63                              # DW_AT_type
	.byte	50                              # Abbrev [50] 0x24aa:0x5 DW_TAG_reference_type
	.long	63                              # DW_AT_type
	.byte	51                              # Abbrev [51] 0x24af:0x5 DW_TAG_rvalue_reference_type
	.long	63                              # DW_AT_type
	.byte	52                              # Abbrev [52] 0x24b4:0x5 DW_TAG_const_type
	.long	63                              # DW_AT_type
	.byte	13                              # Abbrev [13] 0x24b9:0xf DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string181                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
	.byte	12                              # Abbrev [12] 0x24c2:0x5 DW_TAG_structure_type
	.long	.Linfo_string182                # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0x24c8:0x7 DW_TAG_base_type
	.long	.Linfo_string218                # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x24cf:0x7 DW_TAG_base_type
	.long	.Linfo_string221                # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	16                              # DW_AT_byte_size
	.byte	53                              # Abbrev [53] 0x24d6:0x19 DW_TAG_structure_type
	.long	.Linfo_string228                # DW_AT_name
                                        # DW_AT_declaration
	.byte	14                              # Abbrev [14] 0x24db:0x9 DW_TAG_template_type_parameter
	.long	250                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	15                              # Abbrev [15] 0x24e4:0xa DW_TAG_template_value_parameter
	.long	279                             # DW_AT_type
	.long	.Linfo_string20                 # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	54                              # Abbrev [54] 0x24ef:0xb DW_TAG_subroutine_type
	.long	63                              # DW_AT_type
	.byte	40                              # Abbrev [40] 0x24f4:0x5 DW_TAG_formal_parameter
	.long	6083                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	50                              # Abbrev [50] 0x24fa:0x5 DW_TAG_reference_type
	.long	9396                            # DW_AT_type
	.byte	50                              # Abbrev [50] 0x24ff:0x5 DW_TAG_reference_type
	.long	9476                            # DW_AT_type
	.byte	42                              # Abbrev [42] 0x2504:0x5 DW_TAG_pointer_type
	.long	9396                            # DW_AT_type
	.byte	55                              # Abbrev [55] 0x2509:0x7 DW_TAG_namespace
	.byte	12                              # Abbrev [12] 0x250a:0x5 DW_TAG_structure_type
	.long	.Linfo_string239                # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	56                              # Abbrev [56] 0x2510:0x5 DW_TAG_unspecified_type
	.long	.Linfo_string242                # DW_AT_name
	.byte	42                              # Abbrev [42] 0x2515:0x5 DW_TAG_pointer_type
	.long	630                             # DW_AT_type
	.byte	42                              # Abbrev [42] 0x251a:0x5 DW_TAG_pointer_type
	.long	9350                            # DW_AT_type
	.byte	52                              # Abbrev [52] 0x251f:0x5 DW_TAG_const_type
	.long	6930                            # DW_AT_type
	.byte	42                              # Abbrev [42] 0x2524:0x5 DW_TAG_pointer_type
	.long	9513                            # DW_AT_type
	.byte	52                              # Abbrev [52] 0x2529:0x5 DW_TAG_const_type
	.long	9518                            # DW_AT_type
	.byte	42                              # Abbrev [42] 0x252e:0x5 DW_TAG_pointer_type
	.long	9523                            # DW_AT_type
	.byte	57                              # Abbrev [57] 0x2533:0x1 DW_TAG_const_type
	.byte	58                              # Abbrev [58] 0x2534:0x1 DW_TAG_subroutine_type
	.byte	42                              # Abbrev [42] 0x2535:0x5 DW_TAG_pointer_type
	.long	9524                            # DW_AT_type
	.byte	42                              # Abbrev [42] 0x253a:0x5 DW_TAG_pointer_type
	.long	1238                            # DW_AT_type
	.byte	42                              # Abbrev [42] 0x253f:0x5 DW_TAG_pointer_type
	.long	1243                            # DW_AT_type
	.byte	42                              # Abbrev [42] 0x2544:0x5 DW_TAG_pointer_type
	.long	9545                            # DW_AT_type
	.byte	13                              # Abbrev [13] 0x2549:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string275                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2552:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2557:0x5 DW_TAG_template_type_parameter
	.long	9381                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	59                              # Abbrev [59] 0x255e:0xb DW_TAG_array_type
	.long	9381                            # DW_AT_type
	.byte	60                              # Abbrev [60] 0x2563:0x5 DW_TAG_subrange_type
	.long	9577                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	61                              # Abbrev [61] 0x2569:0x7 DW_TAG_base_type
	.long	.Linfo_string278                # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	7                               # DW_AT_encoding
	.byte	50                              # Abbrev [50] 0x2570:0x5 DW_TAG_reference_type
	.long	9589                            # DW_AT_type
	.byte	59                              # Abbrev [59] 0x2575:0xc DW_TAG_array_type
	.long	63                              # DW_AT_type
	.byte	62                              # Abbrev [62] 0x257a:0x6 DW_TAG_subrange_type
	.long	9577                            # DW_AT_type
	.byte	3                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2581:0x5 DW_TAG_pointer_type
	.long	9589                            # DW_AT_type
	.byte	42                              # Abbrev [42] 0x2586:0x5 DW_TAG_pointer_type
	.long	9488                            # DW_AT_type
	.byte	13                              # Abbrev [13] 0x258b:0x13 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string298                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	143                             # DW_AT_decl_line
	.byte	14                              # Abbrev [14] 0x2594:0x9 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	63                              # Abbrev [63] 0x259e:0x5 DW_TAG_volatile_type
	.long	9416                            # DW_AT_type
	.byte	64                              # Abbrev [64] 0x25a3:0xc DW_TAG_array_type
                                        # DW_AT_GNU_vector
	.long	63                              # DW_AT_type
	.byte	62                              # Abbrev [62] 0x25a8:0x6 DW_TAG_subrange_type
	.long	9577                            # DW_AT_type
	.byte	2                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	52                              # Abbrev [52] 0x25af:0x5 DW_TAG_const_type
	.long	9652                            # DW_AT_type
	.byte	63                              # Abbrev [63] 0x25b4:0x5 DW_TAG_volatile_type
	.long	9381                            # DW_AT_type
	.byte	52                              # Abbrev [52] 0x25b9:0x5 DW_TAG_const_type
	.long	9662                            # DW_AT_type
	.byte	65                              # Abbrev [65] 0x25be:0x1 DW_TAG_volatile_type
	.byte	13                              # Abbrev [13] 0x25bf:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string314                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x25c8:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x25cd:0x5 DW_TAG_template_type_parameter
	.long	1238                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	66                              # Abbrev [66] 0x25d4:0x9 DW_TAG_ptr_to_member_type
	.long	9693                            # DW_AT_type
	.long	9350                            # DW_AT_containing_type
	.byte	67                              # Abbrev [67] 0x25dd:0x7 DW_TAG_subroutine_type
	.byte	18                              # Abbrev [18] 0x25de:0x5 DW_TAG_formal_parameter
	.long	9700                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x25e4:0x5 DW_TAG_pointer_type
	.long	9705                            # DW_AT_type
	.byte	52                              # Abbrev [52] 0x25e9:0x5 DW_TAG_const_type
	.long	9350                            # DW_AT_type
	.byte	66                              # Abbrev [66] 0x25ee:0x9 DW_TAG_ptr_to_member_type
	.long	9719                            # DW_AT_type
	.long	9350                            # DW_AT_containing_type
	.byte	68                              # Abbrev [68] 0x25f7:0x7 DW_TAG_subroutine_type
                                        # DW_AT_reference
	.byte	18                              # Abbrev [18] 0x25f8:0x5 DW_TAG_formal_parameter
	.long	9726                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x25fe:0x5 DW_TAG_pointer_type
	.long	9731                            # DW_AT_type
	.byte	63                              # Abbrev [63] 0x2603:0x5 DW_TAG_volatile_type
	.long	9350                            # DW_AT_type
	.byte	66                              # Abbrev [66] 0x2608:0x9 DW_TAG_ptr_to_member_type
	.long	9745                            # DW_AT_type
	.long	9350                            # DW_AT_containing_type
	.byte	69                              # Abbrev [69] 0x2611:0x7 DW_TAG_subroutine_type
                                        # DW_AT_rvalue_reference
	.byte	18                              # Abbrev [18] 0x2612:0x5 DW_TAG_formal_parameter
	.long	9752                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2618:0x5 DW_TAG_pointer_type
	.long	9757                            # DW_AT_type
	.byte	52                              # Abbrev [52] 0x261d:0x5 DW_TAG_const_type
	.long	9731                            # DW_AT_type
	.byte	52                              # Abbrev [52] 0x2622:0x5 DW_TAG_const_type
	.long	9525                            # DW_AT_type
	.byte	50                              # Abbrev [50] 0x2627:0x5 DW_TAG_reference_type
	.long	9772                            # DW_AT_type
	.byte	52                              # Abbrev [52] 0x262c:0x5 DW_TAG_const_type
	.long	9777                            # DW_AT_type
	.byte	59                              # Abbrev [59] 0x2631:0xc DW_TAG_array_type
	.long	9416                            # DW_AT_type
	.byte	62                              # Abbrev [62] 0x2636:0x6 DW_TAG_subrange_type
	.long	9577                            # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	52                              # Abbrev [52] 0x263d:0x5 DW_TAG_const_type
	.long	9794                            # DW_AT_type
	.byte	70                              # Abbrev [70] 0x2642:0x1 DW_TAG_subroutine_type
                                        # DW_AT_reference
	.byte	63                              # Abbrev [63] 0x2643:0x5 DW_TAG_volatile_type
	.long	9800                            # DW_AT_type
	.byte	71                              # Abbrev [71] 0x2648:0x1 DW_TAG_subroutine_type
                                        # DW_AT_rvalue_reference
	.byte	52                              # Abbrev [52] 0x2649:0x5 DW_TAG_const_type
	.long	9806                            # DW_AT_type
	.byte	63                              # Abbrev [63] 0x264e:0x5 DW_TAG_volatile_type
	.long	9524                            # DW_AT_type
	.byte	52                              # Abbrev [52] 0x2653:0x5 DW_TAG_const_type
	.long	9816                            # DW_AT_type
	.byte	59                              # Abbrev [59] 0x2658:0xc DW_TAG_array_type
	.long	9381                            # DW_AT_type
	.byte	62                              # Abbrev [62] 0x265d:0x6 DW_TAG_subrange_type
	.long	9577                            # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	50                              # Abbrev [50] 0x2664:0x5 DW_TAG_reference_type
	.long	9811                            # DW_AT_type
	.byte	50                              # Abbrev [50] 0x2669:0x5 DW_TAG_reference_type
	.long	9838                            # DW_AT_type
	.byte	52                              # Abbrev [52] 0x266e:0x5 DW_TAG_const_type
	.long	9843                            # DW_AT_type
	.byte	66                              # Abbrev [66] 0x2673:0x9 DW_TAG_ptr_to_member_type
	.long	9852                            # DW_AT_type
	.long	9350                            # DW_AT_containing_type
	.byte	67                              # Abbrev [67] 0x267c:0x7 DW_TAG_subroutine_type
	.byte	18                              # Abbrev [18] 0x267d:0x5 DW_TAG_formal_parameter
	.long	9859                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2683:0x5 DW_TAG_pointer_type
	.long	9350                            # DW_AT_type
	.byte	54                              # Abbrev [54] 0x2688:0xb DW_TAG_subroutine_type
	.long	9875                            # DW_AT_type
	.byte	40                              # Abbrev [40] 0x268d:0x5 DW_TAG_formal_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2693:0x5 DW_TAG_pointer_type
	.long	9880                            # DW_AT_type
	.byte	67                              # Abbrev [67] 0x2698:0x7 DW_TAG_subroutine_type
	.byte	40                              # Abbrev [40] 0x2699:0x5 DW_TAG_formal_parameter
	.long	6083                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	67                              # Abbrev [67] 0x269f:0x7 DW_TAG_subroutine_type
	.byte	40                              # Abbrev [40] 0x26a0:0x5 DW_TAG_formal_parameter
	.long	1243                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	67                              # Abbrev [67] 0x26a6:0xc DW_TAG_subroutine_type
	.byte	40                              # Abbrev [40] 0x26a7:0x5 DW_TAG_formal_parameter
	.long	1253                            # DW_AT_type
	.byte	40                              # Abbrev [40] 0x26ac:0x5 DW_TAG_formal_parameter
	.long	1243                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	67                              # Abbrev [67] 0x26b2:0x7 DW_TAG_subroutine_type
	.byte	40                              # Abbrev [40] 0x26b3:0x5 DW_TAG_formal_parameter
	.long	1253                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	66                              # Abbrev [66] 0x26b9:0x9 DW_TAG_ptr_to_member_type
	.long	9922                            # DW_AT_type
	.long	9124                            # DW_AT_containing_type
	.byte	67                              # Abbrev [67] 0x26c2:0x7 DW_TAG_subroutine_type
	.byte	18                              # Abbrev [18] 0x26c3:0x5 DW_TAG_formal_parameter
	.long	9151                            # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x26c9:0x14 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string360                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	49                              # DW_AT_decl_line
	.byte	15                              # Abbrev [15] 0x26d2:0xa DW_TAG_template_value_parameter
	.long	236                             # DW_AT_type
	.long	.Linfo_string359                # DW_AT_name
	.byte	3                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x26dd:0x1d DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string362                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	40                              # DW_AT_decl_line
	.byte	14                              # Abbrev [14] 0x26e6:0x9 DW_TAG_template_type_parameter
	.long	1238                            # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	15                              # Abbrev [15] 0x26ef:0xa DW_TAG_template_value_parameter
	.long	279                             # DW_AT_type
	.long	.Linfo_string20                 # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x26fa:0xb DW_TAG_typedef
	.long	9611                            # DW_AT_type
	.long	.Linfo_string365                # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	139                             # DW_AT_decl_line
	.byte	42                              # Abbrev [42] 0x2705:0x5 DW_TAG_pointer_type
	.long	9994                            # DW_AT_type
	.byte	53                              # Abbrev [53] 0x270a:0x11 DW_TAG_structure_type
	.long	.Linfo_string367                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x270f:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2714:0x5 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x271b:0x5 DW_TAG_pointer_type
	.long	10016                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2720:0x11 DW_TAG_structure_type
	.long	.Linfo_string368                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2725:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x272a:0x5 DW_TAG_template_type_parameter
	.long	6083                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2731:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string369                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x273a:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x273f:0x5 DW_TAG_template_type_parameter
	.long	279                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2746:0x5 DW_TAG_pointer_type
	.long	10059                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x274b:0x11 DW_TAG_structure_type
	.long	.Linfo_string370                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2750:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2755:0x5 DW_TAG_template_type_parameter
	.long	279                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x275c:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string371                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2765:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x276a:0x5 DW_TAG_template_type_parameter
	.long	9329                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2771:0x5 DW_TAG_pointer_type
	.long	10102                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2776:0x11 DW_TAG_structure_type
	.long	.Linfo_string372                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x277b:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2780:0x5 DW_TAG_template_type_parameter
	.long	9329                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2787:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string373                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2790:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2795:0x5 DW_TAG_template_type_parameter
	.long	630                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x279c:0x5 DW_TAG_pointer_type
	.long	10145                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x27a1:0x11 DW_TAG_structure_type
	.long	.Linfo_string374                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x27a6:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x27ab:0x5 DW_TAG_template_type_parameter
	.long	630                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x27b2:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string375                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x27bb:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x27c0:0x5 DW_TAG_template_type_parameter
	.long	579                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x27c7:0x5 DW_TAG_pointer_type
	.long	10188                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x27cc:0x11 DW_TAG_structure_type
	.long	.Linfo_string376                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x27d1:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x27d6:0x5 DW_TAG_template_type_parameter
	.long	579                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x27dd:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string377                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x27e6:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x27eb:0x5 DW_TAG_template_type_parameter
	.long	236                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x27f2:0x5 DW_TAG_pointer_type
	.long	10231                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x27f7:0x11 DW_TAG_structure_type
	.long	.Linfo_string378                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x27fc:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2801:0x5 DW_TAG_template_type_parameter
	.long	236                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2808:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string379                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2811:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2816:0x5 DW_TAG_template_type_parameter
	.long	9336                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x281d:0x5 DW_TAG_pointer_type
	.long	10274                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2822:0x11 DW_TAG_structure_type
	.long	.Linfo_string380                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2827:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x282c:0x5 DW_TAG_template_type_parameter
	.long	9336                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2833:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string381                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x283c:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2841:0x5 DW_TAG_template_type_parameter
	.long	9343                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2848:0x5 DW_TAG_pointer_type
	.long	10317                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x284d:0x11 DW_TAG_structure_type
	.long	.Linfo_string382                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2852:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2857:0x5 DW_TAG_template_type_parameter
	.long	9343                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x285e:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string383                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2867:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x286c:0x5 DW_TAG_template_type_parameter
	.long	9350                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2873:0x5 DW_TAG_pointer_type
	.long	10360                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2878:0x11 DW_TAG_structure_type
	.long	.Linfo_string384                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x287d:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2882:0x5 DW_TAG_template_type_parameter
	.long	9350                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2889:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string385                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2892:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2897:0x5 DW_TAG_template_type_parameter
	.long	219                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x289e:0x5 DW_TAG_pointer_type
	.long	10403                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x28a3:0x11 DW_TAG_structure_type
	.long	.Linfo_string386                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x28a8:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x28ad:0x5 DW_TAG_template_type_parameter
	.long	219                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x28b4:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string387                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x28bd:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x28c2:0x5 DW_TAG_template_type_parameter
	.long	9355                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x28c9:0x5 DW_TAG_pointer_type
	.long	10446                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x28ce:0x11 DW_TAG_structure_type
	.long	.Linfo_string388                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x28d3:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x28d8:0x5 DW_TAG_template_type_parameter
	.long	9355                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x28df:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string389                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x28e8:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x28ed:0x5 DW_TAG_template_type_parameter
	.long	229                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x28f4:0x5 DW_TAG_pointer_type
	.long	10489                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x28f9:0x11 DW_TAG_structure_type
	.long	.Linfo_string390                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x28fe:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2903:0x5 DW_TAG_template_type_parameter
	.long	229                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x290a:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string391                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2913:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2918:0x5 DW_TAG_template_type_parameter
	.long	9360                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x291f:0x5 DW_TAG_pointer_type
	.long	10532                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2924:0x11 DW_TAG_structure_type
	.long	.Linfo_string392                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2929:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x292e:0x5 DW_TAG_template_type_parameter
	.long	9360                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2935:0x1a DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string393                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x293e:0x10 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2943:0x5 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2948:0x5 DW_TAG_template_type_parameter
	.long	6083                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x294f:0x5 DW_TAG_pointer_type
	.long	10580                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2954:0x16 DW_TAG_structure_type
	.long	.Linfo_string394                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2959:0x10 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x295e:0x5 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2963:0x5 DW_TAG_template_type_parameter
	.long	6083                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x296a:0x5 DW_TAG_pointer_type
	.long	10607                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x296f:0x11 DW_TAG_structure_type
	.long	.Linfo_string395                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2974:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2979:0x5 DW_TAG_template_type_parameter
	.long	9381                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2980:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string396                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2989:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x298e:0x5 DW_TAG_template_type_parameter
	.long	9386                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2995:0x5 DW_TAG_pointer_type
	.long	10650                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x299a:0x11 DW_TAG_structure_type
	.long	.Linfo_string397                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x299f:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x29a4:0x5 DW_TAG_template_type_parameter
	.long	9386                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x29ab:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string398                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x29b4:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x29b9:0x5 DW_TAG_template_type_parameter
	.long	9391                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x29c0:0x5 DW_TAG_pointer_type
	.long	10693                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x29c5:0x11 DW_TAG_structure_type
	.long	.Linfo_string399                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x29ca:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x29cf:0x5 DW_TAG_template_type_parameter
	.long	9391                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x29d6:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string400                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x29df:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x29e4:0x5 DW_TAG_template_type_parameter
	.long	9396                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x29eb:0x5 DW_TAG_pointer_type
	.long	10736                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x29f0:0x11 DW_TAG_structure_type
	.long	.Linfo_string401                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x29f5:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x29fa:0x5 DW_TAG_template_type_parameter
	.long	9396                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2a01:0x11 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string402                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2a0a:0x7 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2a0f:0x1 DW_TAG_template_type_parameter
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2a12:0x5 DW_TAG_pointer_type
	.long	10775                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2a17:0xd DW_TAG_structure_type
	.long	.Linfo_string403                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2a1c:0x7 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	31                              # Abbrev [31] 0x2a21:0x1 DW_TAG_template_type_parameter
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2a24:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string404                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2a2d:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2a32:0x5 DW_TAG_template_type_parameter
	.long	9410                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2a39:0x5 DW_TAG_pointer_type
	.long	10814                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2a3e:0x11 DW_TAG_structure_type
	.long	.Linfo_string405                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2a43:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2a48:0x5 DW_TAG_template_type_parameter
	.long	9410                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2a4f:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string406                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2a58:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2a5d:0x5 DW_TAG_template_type_parameter
	.long	897                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2a64:0x5 DW_TAG_pointer_type
	.long	10857                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2a69:0x11 DW_TAG_structure_type
	.long	.Linfo_string407                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2a6e:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2a73:0x5 DW_TAG_template_type_parameter
	.long	897                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2a7a:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string408                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2a83:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2a88:0x5 DW_TAG_template_type_parameter
	.long	250                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2a8f:0x5 DW_TAG_pointer_type
	.long	10900                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2a94:0x11 DW_TAG_structure_type
	.long	.Linfo_string409                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2a99:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2a9e:0x5 DW_TAG_template_type_parameter
	.long	250                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2aa5:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string410                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2aae:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2ab3:0x5 DW_TAG_template_type_parameter
	.long	9430                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2aba:0x5 DW_TAG_pointer_type
	.long	10943                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2abf:0x11 DW_TAG_structure_type
	.long	.Linfo_string411                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2ac4:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2ac9:0x5 DW_TAG_template_type_parameter
	.long	9430                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2ad0:0x5 DW_TAG_pointer_type
	.long	10965                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2ad5:0x11 DW_TAG_structure_type
	.long	.Linfo_string412                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2ada:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2adf:0x5 DW_TAG_template_type_parameter
	.long	1238                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2ae6:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string413                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2aef:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2af4:0x5 DW_TAG_template_type_parameter
	.long	9455                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2afb:0x5 DW_TAG_pointer_type
	.long	11008                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2b00:0x11 DW_TAG_structure_type
	.long	.Linfo_string414                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2b05:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2b0a:0x5 DW_TAG_template_type_parameter
	.long	9455                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2b11:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string415                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2b1a:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2b1f:0x5 DW_TAG_template_type_parameter
	.long	9466                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2b26:0x5 DW_TAG_pointer_type
	.long	11051                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2b2b:0x11 DW_TAG_structure_type
	.long	.Linfo_string416                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2b30:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2b35:0x5 DW_TAG_template_type_parameter
	.long	9466                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2b3c:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string417                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2b45:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2b4a:0x5 DW_TAG_template_type_parameter
	.long	9471                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2b51:0x5 DW_TAG_pointer_type
	.long	11094                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2b56:0x11 DW_TAG_structure_type
	.long	.Linfo_string418                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2b5b:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2b60:0x5 DW_TAG_template_type_parameter
	.long	9471                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2b67:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string419                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2b70:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2b75:0x5 DW_TAG_template_type_parameter
	.long	9482                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2b7c:0x5 DW_TAG_pointer_type
	.long	11137                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2b81:0x11 DW_TAG_structure_type
	.long	.Linfo_string420                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2b86:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2b8b:0x5 DW_TAG_template_type_parameter
	.long	9482                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2b92:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string421                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2b9b:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2ba0:0x5 DW_TAG_template_type_parameter
	.long	9488                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2ba7:0x5 DW_TAG_pointer_type
	.long	11180                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2bac:0x11 DW_TAG_structure_type
	.long	.Linfo_string422                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2bb1:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2bb6:0x5 DW_TAG_template_type_parameter
	.long	9488                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2bbd:0x1a DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string423                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2bc6:0x10 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2bcb:0x5 DW_TAG_template_type_parameter
	.long	9493                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2bd0:0x5 DW_TAG_template_type_parameter
	.long	9493                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2bd7:0x5 DW_TAG_pointer_type
	.long	11228                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2bdc:0x16 DW_TAG_structure_type
	.long	.Linfo_string424                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2be1:0x10 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2be6:0x5 DW_TAG_template_type_parameter
	.long	9493                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2beb:0x5 DW_TAG_template_type_parameter
	.long	9493                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2bf2:0x1a DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string425                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2bfb:0x10 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2c00:0x5 DW_TAG_template_type_parameter
	.long	9493                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2c05:0x5 DW_TAG_template_type_parameter
	.long	9498                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2c0c:0x5 DW_TAG_pointer_type
	.long	11281                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2c11:0x16 DW_TAG_structure_type
	.long	.Linfo_string426                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2c16:0x10 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2c1b:0x5 DW_TAG_template_type_parameter
	.long	9493                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2c20:0x5 DW_TAG_template_type_parameter
	.long	9498                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2c27:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string427                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2c30:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2c35:0x5 DW_TAG_template_type_parameter
	.long	9503                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2c3c:0x5 DW_TAG_pointer_type
	.long	11329                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2c41:0x11 DW_TAG_structure_type
	.long	.Linfo_string428                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2c46:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2c4b:0x5 DW_TAG_template_type_parameter
	.long	9503                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2c52:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string429                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2c5b:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2c60:0x5 DW_TAG_template_type_parameter
	.long	9508                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2c67:0x5 DW_TAG_pointer_type
	.long	11372                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2c6c:0x11 DW_TAG_structure_type
	.long	.Linfo_string430                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2c71:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2c76:0x5 DW_TAG_template_type_parameter
	.long	9508                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2c7d:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string431                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2c86:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2c8b:0x5 DW_TAG_template_type_parameter
	.long	9524                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2c92:0x5 DW_TAG_pointer_type
	.long	11415                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2c97:0x11 DW_TAG_structure_type
	.long	.Linfo_string432                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2c9c:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2ca1:0x5 DW_TAG_template_type_parameter
	.long	9524                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2ca8:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string433                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2cb1:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2cb6:0x5 DW_TAG_template_type_parameter
	.long	9525                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2cbd:0x5 DW_TAG_pointer_type
	.long	11458                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2cc2:0x11 DW_TAG_structure_type
	.long	.Linfo_string434                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2cc7:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2ccc:0x5 DW_TAG_template_type_parameter
	.long	9525                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2cd3:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string435                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2cdc:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2ce1:0x5 DW_TAG_template_type_parameter
	.long	9530                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2ce8:0x5 DW_TAG_pointer_type
	.long	11501                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2ced:0x11 DW_TAG_structure_type
	.long	.Linfo_string436                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2cf2:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2cf7:0x5 DW_TAG_template_type_parameter
	.long	9530                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2cfe:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string437                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2d07:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2d0c:0x5 DW_TAG_template_type_parameter
	.long	1243                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2d13:0x5 DW_TAG_pointer_type
	.long	11544                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2d18:0x11 DW_TAG_structure_type
	.long	.Linfo_string438                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2d1d:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2d22:0x5 DW_TAG_template_type_parameter
	.long	1243                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2d29:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string439                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2d32:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2d37:0x5 DW_TAG_template_type_parameter
	.long	9535                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2d3e:0x5 DW_TAG_pointer_type
	.long	11587                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2d43:0x11 DW_TAG_structure_type
	.long	.Linfo_string440                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2d48:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2d4d:0x5 DW_TAG_template_type_parameter
	.long	9535                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2d54:0xf DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string441                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	38                              # Abbrev [38] 0x2d5d:0x5 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2d63:0x5 DW_TAG_pointer_type
	.long	11624                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2d68:0xb DW_TAG_structure_type
	.long	.Linfo_string442                # DW_AT_name
                                        # DW_AT_declaration
	.byte	38                              # Abbrev [38] 0x2d6d:0x5 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2d73:0x1a DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string443                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2d7c:0x10 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2d81:0x5 DW_TAG_template_type_parameter
	.long	9518                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2d86:0x5 DW_TAG_template_type_parameter
	.long	9518                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2d8d:0x5 DW_TAG_pointer_type
	.long	11666                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2d92:0x16 DW_TAG_structure_type
	.long	.Linfo_string444                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2d97:0x10 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2d9c:0x5 DW_TAG_template_type_parameter
	.long	9518                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2da1:0x5 DW_TAG_template_type_parameter
	.long	9518                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2da8:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string445                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2db1:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2db6:0x5 DW_TAG_template_type_parameter
	.long	9540                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2dbd:0x5 DW_TAG_pointer_type
	.long	11714                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2dc2:0x11 DW_TAG_structure_type
	.long	.Linfo_string446                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2dc7:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2dcc:0x5 DW_TAG_template_type_parameter
	.long	9540                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2dd3:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string447                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2ddc:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2de1:0x5 DW_TAG_template_type_parameter
	.long	9566                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2de8:0x5 DW_TAG_pointer_type
	.long	11757                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2ded:0x11 DW_TAG_structure_type
	.long	.Linfo_string448                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2df2:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2df7:0x5 DW_TAG_template_type_parameter
	.long	9566                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2dfe:0x5 DW_TAG_pointer_type
	.long	5142                            # DW_AT_type
	.byte	13                              # Abbrev [13] 0x2e03:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string450                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2e0c:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2e11:0x5 DW_TAG_template_type_parameter
	.long	1248                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2e18:0x5 DW_TAG_pointer_type
	.long	11805                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2e1d:0x11 DW_TAG_structure_type
	.long	.Linfo_string451                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2e22:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2e27:0x5 DW_TAG_template_type_parameter
	.long	1248                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2e2e:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string452                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2e37:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2e3c:0x5 DW_TAG_template_type_parameter
	.long	9584                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2e43:0x5 DW_TAG_pointer_type
	.long	11848                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2e48:0x11 DW_TAG_structure_type
	.long	.Linfo_string453                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2e4d:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2e52:0x5 DW_TAG_template_type_parameter
	.long	9584                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2e59:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string454                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2e62:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2e67:0x5 DW_TAG_template_type_parameter
	.long	9601                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2e6e:0x5 DW_TAG_pointer_type
	.long	11891                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2e73:0x11 DW_TAG_structure_type
	.long	.Linfo_string455                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2e78:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2e7d:0x5 DW_TAG_template_type_parameter
	.long	9601                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2e84:0x1a DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string456                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2e8d:0x10 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2e92:0x5 DW_TAG_template_type_parameter
	.long	9381                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2e97:0x5 DW_TAG_template_type_parameter
	.long	9606                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2e9e:0x5 DW_TAG_pointer_type
	.long	11939                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2ea3:0x16 DW_TAG_structure_type
	.long	.Linfo_string457                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2ea8:0x10 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2ead:0x5 DW_TAG_template_type_parameter
	.long	9381                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2eb2:0x5 DW_TAG_template_type_parameter
	.long	9606                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2eb9:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string458                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2ec2:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2ec7:0x5 DW_TAG_template_type_parameter
	.long	9611                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2ece:0x5 DW_TAG_pointer_type
	.long	11987                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2ed3:0x11 DW_TAG_structure_type
	.long	.Linfo_string459                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2ed8:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2edd:0x5 DW_TAG_template_type_parameter
	.long	9611                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2ee4:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string460                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2eed:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2ef2:0x5 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2ef9:0x5 DW_TAG_pointer_type
	.long	12030                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2efe:0x11 DW_TAG_structure_type
	.long	.Linfo_string461                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2f03:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2f08:0x5 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2f0f:0x1f DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string462                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2f18:0x15 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2f1d:0x5 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2f22:0x5 DW_TAG_template_type_parameter
	.long	630                             # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2f27:0x5 DW_TAG_template_type_parameter
	.long	9630                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2f2e:0x5 DW_TAG_pointer_type
	.long	12083                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2f33:0x1b DW_TAG_structure_type
	.long	.Linfo_string463                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2f38:0x15 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2f3d:0x5 DW_TAG_template_type_parameter
	.long	63                              # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2f42:0x5 DW_TAG_template_type_parameter
	.long	630                             # DW_AT_type
	.byte	30                              # Abbrev [30] 0x2f47:0x5 DW_TAG_template_type_parameter
	.long	9630                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2f4e:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string464                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2f57:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2f5c:0x5 DW_TAG_template_type_parameter
	.long	9635                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2f63:0x5 DW_TAG_pointer_type
	.long	12136                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2f68:0x11 DW_TAG_structure_type
	.long	.Linfo_string465                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2f6d:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2f72:0x5 DW_TAG_template_type_parameter
	.long	9635                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2f79:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string466                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2f82:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2f87:0x5 DW_TAG_template_type_parameter
	.long	9647                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2f8e:0x5 DW_TAG_pointer_type
	.long	12179                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2f93:0x11 DW_TAG_structure_type
	.long	.Linfo_string467                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2f98:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2f9d:0x5 DW_TAG_template_type_parameter
	.long	9647                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2fa4:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string468                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2fad:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2fb2:0x5 DW_TAG_template_type_parameter
	.long	9657                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2fb9:0x5 DW_TAG_pointer_type
	.long	12222                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2fbe:0x11 DW_TAG_structure_type
	.long	.Linfo_string469                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2fc3:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2fc8:0x5 DW_TAG_template_type_parameter
	.long	9657                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x2fcf:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string470                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x2fd8:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2fdd:0x5 DW_TAG_template_type_parameter
	.long	9663                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2fe4:0x5 DW_TAG_pointer_type
	.long	12265                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x2fe9:0x11 DW_TAG_structure_type
	.long	.Linfo_string471                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x2fee:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x2ff3:0x5 DW_TAG_template_type_parameter
	.long	9663                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x2ffa:0x5 DW_TAG_pointer_type
	.long	286                             # DW_AT_type
	.byte	13                              # Abbrev [13] 0x2fff:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string472                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x3008:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x300d:0x5 DW_TAG_template_type_parameter
	.long	9684                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x3014:0x5 DW_TAG_pointer_type
	.long	12313                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x3019:0x11 DW_TAG_structure_type
	.long	.Linfo_string473                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x301e:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x3023:0x5 DW_TAG_template_type_parameter
	.long	9684                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x302a:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string474                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x3033:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x3038:0x5 DW_TAG_template_type_parameter
	.long	9710                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x303f:0x5 DW_TAG_pointer_type
	.long	12356                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x3044:0x11 DW_TAG_structure_type
	.long	.Linfo_string475                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x3049:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x304e:0x5 DW_TAG_template_type_parameter
	.long	9710                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x3055:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string476                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x305e:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x3063:0x5 DW_TAG_template_type_parameter
	.long	9736                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x306a:0x5 DW_TAG_pointer_type
	.long	12399                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x306f:0x11 DW_TAG_structure_type
	.long	.Linfo_string477                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x3074:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x3079:0x5 DW_TAG_template_type_parameter
	.long	9736                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x3080:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string478                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x3089:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x308e:0x5 DW_TAG_template_type_parameter
	.long	9762                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x3095:0x5 DW_TAG_pointer_type
	.long	12442                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x309a:0x11 DW_TAG_structure_type
	.long	.Linfo_string479                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x309f:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x30a4:0x5 DW_TAG_template_type_parameter
	.long	9762                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x30ab:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string480                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x30b4:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x30b9:0x5 DW_TAG_template_type_parameter
	.long	9767                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x30c0:0x5 DW_TAG_pointer_type
	.long	12485                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x30c5:0x11 DW_TAG_structure_type
	.long	.Linfo_string481                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x30ca:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x30cf:0x5 DW_TAG_template_type_parameter
	.long	9767                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x30d6:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string482                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x30df:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x30e4:0x5 DW_TAG_template_type_parameter
	.long	9789                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x30eb:0x5 DW_TAG_pointer_type
	.long	12528                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x30f0:0x11 DW_TAG_structure_type
	.long	.Linfo_string483                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x30f5:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x30fa:0x5 DW_TAG_template_type_parameter
	.long	9789                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x3101:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string484                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x310a:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x310f:0x5 DW_TAG_template_type_parameter
	.long	9795                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x3116:0x5 DW_TAG_pointer_type
	.long	12571                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x311b:0x11 DW_TAG_structure_type
	.long	.Linfo_string485                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x3120:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x3125:0x5 DW_TAG_template_type_parameter
	.long	9795                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x312c:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string486                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x3135:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x313a:0x5 DW_TAG_template_type_parameter
	.long	9801                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x3141:0x5 DW_TAG_pointer_type
	.long	12614                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x3146:0x11 DW_TAG_structure_type
	.long	.Linfo_string487                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x314b:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x3150:0x5 DW_TAG_template_type_parameter
	.long	9801                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x3157:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string488                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x3160:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x3165:0x5 DW_TAG_template_type_parameter
	.long	9811                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x316c:0x5 DW_TAG_pointer_type
	.long	12657                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x3171:0x11 DW_TAG_structure_type
	.long	.Linfo_string489                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x3176:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x317b:0x5 DW_TAG_template_type_parameter
	.long	9811                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x3182:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string490                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x318b:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x3190:0x5 DW_TAG_template_type_parameter
	.long	9828                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x3197:0x5 DW_TAG_pointer_type
	.long	12700                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x319c:0x11 DW_TAG_structure_type
	.long	.Linfo_string491                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x31a1:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x31a6:0x5 DW_TAG_template_type_parameter
	.long	9828                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x31ad:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string492                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x31b6:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x31bb:0x5 DW_TAG_template_type_parameter
	.long	9833                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x31c2:0x5 DW_TAG_pointer_type
	.long	12743                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x31c7:0x11 DW_TAG_structure_type
	.long	.Linfo_string493                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x31cc:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x31d1:0x5 DW_TAG_template_type_parameter
	.long	9833                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x31d8:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string494                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x31e1:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x31e6:0x5 DW_TAG_template_type_parameter
	.long	9864                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x31ed:0x5 DW_TAG_pointer_type
	.long	12786                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x31f2:0x11 DW_TAG_structure_type
	.long	.Linfo_string495                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x31f7:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x31fc:0x5 DW_TAG_template_type_parameter
	.long	9864                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x3203:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string496                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x320c:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x3211:0x5 DW_TAG_template_type_parameter
	.long	9525                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x3218:0x5 DW_TAG_pointer_type
	.long	12829                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x321d:0x11 DW_TAG_structure_type
	.long	.Linfo_string497                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x3222:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x3227:0x5 DW_TAG_template_type_parameter
	.long	9525                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x322e:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string498                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x3237:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x323c:0x5 DW_TAG_template_type_parameter
	.long	9887                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x3243:0x5 DW_TAG_pointer_type
	.long	12872                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x3248:0x11 DW_TAG_structure_type
	.long	.Linfo_string499                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x324d:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x3252:0x5 DW_TAG_template_type_parameter
	.long	9887                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x3259:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string500                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x3262:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x3267:0x5 DW_TAG_template_type_parameter
	.long	9894                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x326e:0x5 DW_TAG_pointer_type
	.long	12915                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x3273:0x11 DW_TAG_structure_type
	.long	.Linfo_string501                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x3278:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x327d:0x5 DW_TAG_template_type_parameter
	.long	9894                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x3284:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string502                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x328d:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x3292:0x5 DW_TAG_template_type_parameter
	.long	9906                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x3299:0x5 DW_TAG_pointer_type
	.long	12958                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x329e:0x11 DW_TAG_structure_type
	.long	.Linfo_string503                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x32a3:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x32a8:0x5 DW_TAG_template_type_parameter
	.long	9906                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x32af:0x5 DW_TAG_pointer_type
	.long	9124                            # DW_AT_type
	.byte	13                              # Abbrev [13] 0x32b4:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string450                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x32bd:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x32c2:0x5 DW_TAG_template_type_parameter
	.long	9193                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x32c9:0x5 DW_TAG_pointer_type
	.long	13006                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x32ce:0x11 DW_TAG_structure_type
	.long	.Linfo_string451                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x32d3:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x32d8:0x5 DW_TAG_template_type_parameter
	.long	9193                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x32df:0x15 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string504                # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x32e8:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x32ed:0x5 DW_TAG_template_type_parameter
	.long	9913                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	42                              # Abbrev [42] 0x32f4:0x5 DW_TAG_pointer_type
	.long	13049                           # DW_AT_type
	.byte	53                              # Abbrev [53] 0x32f9:0x11 DW_TAG_structure_type
	.long	.Linfo_string505                # DW_AT_name
                                        # DW_AT_declaration
	.byte	29                              # Abbrev [29] 0x32fe:0xb DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string84                 # DW_AT_name
	.byte	30                              # Abbrev [30] 0x3303:0x5 DW_TAG_template_type_parameter
	.long	9913                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_end1
	.quad	.Lfunc_begin28
	.quad	.Lfunc_end28
	.quad	.Lfunc_begin44
	.quad	.Lfunc_end44
	.quad	.Lfunc_begin48
	.quad	.Lfunc_end48
	.quad	.Lfunc_begin56
	.quad	.Lfunc_end58
	.quad	.Lfunc_begin90
	.quad	.Lfunc_end90
	.quad	.Lfunc_begin104
	.quad	.Lfunc_end104
	.quad	.Lfunc_begin120
	.quad	.Lfunc_end122
	.quad	.Lfunc_begin124
	.quad	.Lfunc_end125
	.quad	.Lfunc_begin2
	.quad	.Lfunc_end2
	.quad	.Lfunc_begin3
	.quad	.Lfunc_end3
	.quad	.Lfunc_begin4
	.quad	.Lfunc_end4
	.quad	.Lfunc_begin5
	.quad	.Lfunc_end5
	.quad	.Lfunc_begin6
	.quad	.Lfunc_end6
	.quad	.Lfunc_begin7
	.quad	.Lfunc_end7
	.quad	.Lfunc_begin8
	.quad	.Lfunc_end8
	.quad	.Lfunc_begin9
	.quad	.Lfunc_end9
	.quad	.Lfunc_begin10
	.quad	.Lfunc_end10
	.quad	.Lfunc_begin11
	.quad	.Lfunc_end11
	.quad	.Lfunc_begin12
	.quad	.Lfunc_end12
	.quad	.Lfunc_begin13
	.quad	.Lfunc_end13
	.quad	.Lfunc_begin14
	.quad	.Lfunc_end14
	.quad	.Lfunc_begin15
	.quad	.Lfunc_end15
	.quad	.Lfunc_begin16
	.quad	.Lfunc_end16
	.quad	.Lfunc_begin17
	.quad	.Lfunc_end17
	.quad	.Lfunc_begin18
	.quad	.Lfunc_end18
	.quad	.Lfunc_begin19
	.quad	.Lfunc_end19
	.quad	.Lfunc_begin20
	.quad	.Lfunc_end20
	.quad	.Lfunc_begin21
	.quad	.Lfunc_end21
	.quad	.Lfunc_begin22
	.quad	.Lfunc_end22
	.quad	.Lfunc_begin23
	.quad	.Lfunc_end23
	.quad	.Lfunc_begin24
	.quad	.Lfunc_end24
	.quad	.Lfunc_begin25
	.quad	.Lfunc_end25
	.quad	.Lfunc_begin26
	.quad	.Lfunc_end26
	.quad	.Lfunc_begin27
	.quad	.Lfunc_end27
	.quad	.Lfunc_begin29
	.quad	.Lfunc_end29
	.quad	.Lfunc_begin30
	.quad	.Lfunc_end30
	.quad	.Lfunc_begin31
	.quad	.Lfunc_end31
	.quad	.Lfunc_begin32
	.quad	.Lfunc_end32
	.quad	.Lfunc_begin33
	.quad	.Lfunc_end33
	.quad	.Lfunc_begin34
	.quad	.Lfunc_end34
	.quad	.Lfunc_begin35
	.quad	.Lfunc_end35
	.quad	.Lfunc_begin36
	.quad	.Lfunc_end36
	.quad	.Lfunc_begin37
	.quad	.Lfunc_end37
	.quad	.Lfunc_begin38
	.quad	.Lfunc_end38
	.quad	.Lfunc_begin39
	.quad	.Lfunc_end39
	.quad	.Lfunc_begin40
	.quad	.Lfunc_end40
	.quad	.Lfunc_begin41
	.quad	.Lfunc_end41
	.quad	.Lfunc_begin42
	.quad	.Lfunc_end42
	.quad	.Lfunc_begin43
	.quad	.Lfunc_end43
	.quad	.Lfunc_begin45
	.quad	.Lfunc_end45
	.quad	.Lfunc_begin46
	.quad	.Lfunc_end46
	.quad	.Lfunc_begin47
	.quad	.Lfunc_end47
	.quad	.Lfunc_begin49
	.quad	.Lfunc_end49
	.quad	.Lfunc_begin50
	.quad	.Lfunc_end50
	.quad	.Lfunc_begin51
	.quad	.Lfunc_end51
	.quad	.Lfunc_begin52
	.quad	.Lfunc_end52
	.quad	.Lfunc_begin53
	.quad	.Lfunc_end53
	.quad	.Lfunc_begin54
	.quad	.Lfunc_end54
	.quad	.Lfunc_begin55
	.quad	.Lfunc_end55
	.quad	.Lfunc_begin59
	.quad	.Lfunc_end59
	.quad	.Lfunc_begin60
	.quad	.Lfunc_end60
	.quad	.Lfunc_begin61
	.quad	.Lfunc_end61
	.quad	.Lfunc_begin62
	.quad	.Lfunc_end62
	.quad	.Lfunc_begin63
	.quad	.Lfunc_end63
	.quad	.Lfunc_begin64
	.quad	.Lfunc_end64
	.quad	.Lfunc_begin65
	.quad	.Lfunc_end65
	.quad	.Lfunc_begin66
	.quad	.Lfunc_end66
	.quad	.Lfunc_begin67
	.quad	.Lfunc_end67
	.quad	.Lfunc_begin68
	.quad	.Lfunc_end68
	.quad	.Lfunc_begin69
	.quad	.Lfunc_end69
	.quad	.Lfunc_begin70
	.quad	.Lfunc_end70
	.quad	.Lfunc_begin71
	.quad	.Lfunc_end71
	.quad	.Lfunc_begin72
	.quad	.Lfunc_end72
	.quad	.Lfunc_begin73
	.quad	.Lfunc_end73
	.quad	.Lfunc_begin74
	.quad	.Lfunc_end74
	.quad	.Lfunc_begin75
	.quad	.Lfunc_end75
	.quad	.Lfunc_begin76
	.quad	.Lfunc_end76
	.quad	.Lfunc_begin77
	.quad	.Lfunc_end77
	.quad	.Lfunc_begin78
	.quad	.Lfunc_end78
	.quad	.Lfunc_begin79
	.quad	.Lfunc_end79
	.quad	.Lfunc_begin80
	.quad	.Lfunc_end80
	.quad	.Lfunc_begin81
	.quad	.Lfunc_end81
	.quad	.Lfunc_begin82
	.quad	.Lfunc_end82
	.quad	.Lfunc_begin83
	.quad	.Lfunc_end83
	.quad	.Lfunc_begin84
	.quad	.Lfunc_end84
	.quad	.Lfunc_begin85
	.quad	.Lfunc_end85
	.quad	.Lfunc_begin86
	.quad	.Lfunc_end86
	.quad	.Lfunc_begin87
	.quad	.Lfunc_end87
	.quad	.Lfunc_begin88
	.quad	.Lfunc_end88
	.quad	.Lfunc_begin89
	.quad	.Lfunc_end89
	.quad	.Lfunc_begin91
	.quad	.Lfunc_end91
	.quad	.Lfunc_begin92
	.quad	.Lfunc_end92
	.quad	.Lfunc_begin93
	.quad	.Lfunc_end93
	.quad	.Lfunc_begin94
	.quad	.Lfunc_end94
	.quad	.Lfunc_begin95
	.quad	.Lfunc_end95
	.quad	.Lfunc_begin96
	.quad	.Lfunc_end96
	.quad	.Lfunc_begin97
	.quad	.Lfunc_end97
	.quad	.Lfunc_begin98
	.quad	.Lfunc_end98
	.quad	.Lfunc_begin99
	.quad	.Lfunc_end99
	.quad	.Lfunc_begin100
	.quad	.Lfunc_end100
	.quad	.Lfunc_begin101
	.quad	.Lfunc_end101
	.quad	.Lfunc_begin102
	.quad	.Lfunc_end102
	.quad	.Lfunc_begin103
	.quad	.Lfunc_end103
	.quad	.Lfunc_begin105
	.quad	.Lfunc_end105
	.quad	.Lfunc_begin106
	.quad	.Lfunc_end106
	.quad	.Lfunc_begin107
	.quad	.Lfunc_end107
	.quad	.Lfunc_begin108
	.quad	.Lfunc_end108
	.quad	.Lfunc_begin109
	.quad	.Lfunc_end109
	.quad	.Lfunc_begin110
	.quad	.Lfunc_end110
	.quad	.Lfunc_begin111
	.quad	.Lfunc_end111
	.quad	.Lfunc_begin112
	.quad	.Lfunc_end112
	.quad	.Lfunc_begin113
	.quad	.Lfunc_end113
	.quad	.Lfunc_begin114
	.quad	.Lfunc_end114
	.quad	.Lfunc_begin115
	.quad	.Lfunc_end115
	.quad	.Lfunc_begin116
	.quad	.Lfunc_end116
	.quad	.Lfunc_begin117
	.quad	.Lfunc_end117
	.quad	.Lfunc_begin118
	.quad	.Lfunc_end118
	.quad	.Lfunc_begin119
	.quad	.Lfunc_end119
	.quad	.Lfunc_begin123
	.quad	.Lfunc_end123
	.quad	.Lfunc_begin126
	.quad	.Lfunc_end126
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 14.0.0 (git@github.com:llvm/llvm-project.git 6d48e2505c7a68a470e75b61ad504d51db0f8a36)" # string offset=0
.Linfo_string1:
	.asciz	"cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp" # string offset=101
.Linfo_string2:
	.asciz	"./"                            # string offset=188
.Linfo_string3:
	.asciz	"i"                             # string offset=191
.Linfo_string4:
	.asciz	"int"                           # string offset=193
.Linfo_string5:
	.asciz	"ns"                            # string offset=197
.Linfo_string6:
	.asciz	"unsigned int"                  # string offset=200
.Linfo_string7:
	.asciz	"Enumerator1"                   # string offset=213
.Linfo_string8:
	.asciz	"Enumerator2"                   # string offset=225
.Linfo_string9:
	.asciz	"Enumerator3"                   # string offset=237
.Linfo_string10:
	.asciz	"Enumeration"                   # string offset=249
.Linfo_string11:
	.asciz	"EnumerationClass"              # string offset=261
.Linfo_string12:
	.asciz	"unsigned char"                 # string offset=278
.Linfo_string13:
	.asciz	"kNeg"                          # string offset=292
.Linfo_string14:
	.asciz	"EnumerationSmall"              # string offset=297
.Linfo_string15:
	.asciz	"AnonEnum1"                     # string offset=314
.Linfo_string16:
	.asciz	"AnonEnum2"                     # string offset=324
.Linfo_string17:
	.asciz	"AnonEnum3"                     # string offset=334
.Linfo_string18:
	.asciz	"T"                             # string offset=344
.Linfo_string19:
	.asciz	"bool"                          # string offset=346
.Linfo_string20:
	.asciz	"b"                             # string offset=351
.Linfo_string21:
	.asciz	"_STNt3|<int, false>"           # string offset=353
.Linfo_string22:
	.asciz	"t10"                           # string offset=373
.Linfo_string23:
	.asciz	"std"                           # string offset=377
.Linfo_string24:
	.asciz	"signed char"                   # string offset=381
.Linfo_string25:
	.asciz	"__int8_t"                      # string offset=393
.Linfo_string26:
	.asciz	"int8_t"                        # string offset=402
.Linfo_string27:
	.asciz	"short"                         # string offset=409
.Linfo_string28:
	.asciz	"__int16_t"                     # string offset=415
.Linfo_string29:
	.asciz	"int16_t"                       # string offset=425
.Linfo_string30:
	.asciz	"__int32_t"                     # string offset=433
.Linfo_string31:
	.asciz	"int32_t"                       # string offset=443
.Linfo_string32:
	.asciz	"long"                          # string offset=451
.Linfo_string33:
	.asciz	"__int64_t"                     # string offset=456
.Linfo_string34:
	.asciz	"int64_t"                       # string offset=466
.Linfo_string35:
	.asciz	"int_fast8_t"                   # string offset=474
.Linfo_string36:
	.asciz	"int_fast16_t"                  # string offset=486
.Linfo_string37:
	.asciz	"int_fast32_t"                  # string offset=499
.Linfo_string38:
	.asciz	"int_fast64_t"                  # string offset=512
.Linfo_string39:
	.asciz	"__int_least8_t"                # string offset=525
.Linfo_string40:
	.asciz	"int_least8_t"                  # string offset=540
.Linfo_string41:
	.asciz	"__int_least16_t"               # string offset=553
.Linfo_string42:
	.asciz	"int_least16_t"                 # string offset=569
.Linfo_string43:
	.asciz	"__int_least32_t"               # string offset=583
.Linfo_string44:
	.asciz	"int_least32_t"                 # string offset=599
.Linfo_string45:
	.asciz	"__int_least64_t"               # string offset=613
.Linfo_string46:
	.asciz	"int_least64_t"                 # string offset=629
.Linfo_string47:
	.asciz	"__intmax_t"                    # string offset=643
.Linfo_string48:
	.asciz	"intmax_t"                      # string offset=654
.Linfo_string49:
	.asciz	"intptr_t"                      # string offset=663
.Linfo_string50:
	.asciz	"__uint8_t"                     # string offset=672
.Linfo_string51:
	.asciz	"uint8_t"                       # string offset=682
.Linfo_string52:
	.asciz	"unsigned short"                # string offset=690
.Linfo_string53:
	.asciz	"__uint16_t"                    # string offset=705
.Linfo_string54:
	.asciz	"uint16_t"                      # string offset=716
.Linfo_string55:
	.asciz	"__uint32_t"                    # string offset=725
.Linfo_string56:
	.asciz	"uint32_t"                      # string offset=736
.Linfo_string57:
	.asciz	"unsigned long"                 # string offset=745
.Linfo_string58:
	.asciz	"__uint64_t"                    # string offset=759
.Linfo_string59:
	.asciz	"uint64_t"                      # string offset=770
.Linfo_string60:
	.asciz	"uint_fast8_t"                  # string offset=779
.Linfo_string61:
	.asciz	"uint_fast16_t"                 # string offset=792
.Linfo_string62:
	.asciz	"uint_fast32_t"                 # string offset=806
.Linfo_string63:
	.asciz	"uint_fast64_t"                 # string offset=820
.Linfo_string64:
	.asciz	"__uint_least8_t"               # string offset=834
.Linfo_string65:
	.asciz	"uint_least8_t"                 # string offset=850
.Linfo_string66:
	.asciz	"__uint_least16_t"              # string offset=864
.Linfo_string67:
	.asciz	"uint_least16_t"                # string offset=881
.Linfo_string68:
	.asciz	"__uint_least32_t"              # string offset=896
.Linfo_string69:
	.asciz	"uint_least32_t"                # string offset=913
.Linfo_string70:
	.asciz	"__uint_least64_t"              # string offset=928
.Linfo_string71:
	.asciz	"uint_least64_t"                # string offset=945
.Linfo_string72:
	.asciz	"__uintmax_t"                   # string offset=960
.Linfo_string73:
	.asciz	"uintmax_t"                     # string offset=972
.Linfo_string74:
	.asciz	"uintptr_t"                     # string offset=982
.Linfo_string75:
	.asciz	"t6"                            # string offset=992
.Linfo_string76:
	.asciz	"_ZN2t6lsIiEEvi"                # string offset=995
.Linfo_string77:
	.asciz	"operator<<<int>"               # string offset=1010
.Linfo_string78:
	.asciz	"_ZN2t6ltIiEEvi"                # string offset=1026
.Linfo_string79:
	.asciz	"operator<<int>"                # string offset=1041
.Linfo_string80:
	.asciz	"_ZN2t6leIiEEvi"                # string offset=1056
.Linfo_string81:
	.asciz	"operator<=<int>"               # string offset=1071
.Linfo_string82:
	.asciz	"_ZN2t6cvP2t1IJfEEIiEEv"        # string offset=1087
.Linfo_string83:
	.asciz	"operator t1<float> *<int>"     # string offset=1110
.Linfo_string84:
	.asciz	"Ts"                            # string offset=1136
.Linfo_string85:
	.asciz	"float"                         # string offset=1139
.Linfo_string86:
	.asciz	"_STNt1|<float>"                # string offset=1145
.Linfo_string87:
	.asciz	"_ZN2t6miIiEEvi"                # string offset=1160
.Linfo_string88:
	.asciz	"operator-<int>"                # string offset=1175
.Linfo_string89:
	.asciz	"_ZN2t6mlIiEEvi"                # string offset=1190
.Linfo_string90:
	.asciz	"operator*<int>"                # string offset=1205
.Linfo_string91:
	.asciz	"_ZN2t6dvIiEEvi"                # string offset=1220
.Linfo_string92:
	.asciz	"operator/<int>"                # string offset=1235
.Linfo_string93:
	.asciz	"_ZN2t6rmIiEEvi"                # string offset=1250
.Linfo_string94:
	.asciz	"operator%<int>"                # string offset=1265
.Linfo_string95:
	.asciz	"_ZN2t6eoIiEEvi"                # string offset=1280
.Linfo_string96:
	.asciz	"operator^<int>"                # string offset=1295
.Linfo_string97:
	.asciz	"_ZN2t6anIiEEvi"                # string offset=1310
.Linfo_string98:
	.asciz	"operator&<int>"                # string offset=1325
.Linfo_string99:
	.asciz	"_ZN2t6orIiEEvi"                # string offset=1340
.Linfo_string100:
	.asciz	"operator|<int>"                # string offset=1355
.Linfo_string101:
	.asciz	"_ZN2t6coIiEEvv"                # string offset=1370
.Linfo_string102:
	.asciz	"operator~<int>"                # string offset=1385
.Linfo_string103:
	.asciz	"_ZN2t6ntIiEEvv"                # string offset=1400
.Linfo_string104:
	.asciz	"operator!<int>"                # string offset=1415
.Linfo_string105:
	.asciz	"_ZN2t6aSIiEEvi"                # string offset=1430
.Linfo_string106:
	.asciz	"operator=<int>"                # string offset=1445
.Linfo_string107:
	.asciz	"_ZN2t6gtIiEEvi"                # string offset=1460
.Linfo_string108:
	.asciz	"operator><int>"                # string offset=1475
.Linfo_string109:
	.asciz	"_ZN2t6cmIiEEvi"                # string offset=1490
.Linfo_string110:
	.asciz	"operator,<int>"                # string offset=1505
.Linfo_string111:
	.asciz	"_ZN2t6clIiEEvv"                # string offset=1520
.Linfo_string112:
	.asciz	"operator()<int>"               # string offset=1535
.Linfo_string113:
	.asciz	"_ZN2t6ixIiEEvi"                # string offset=1551
.Linfo_string114:
	.asciz	"operator[]<int>"               # string offset=1566
.Linfo_string115:
	.asciz	"_ZN2t6ssIiEEvi"                # string offset=1582
.Linfo_string116:
	.asciz	"operator<=><int>"              # string offset=1597
.Linfo_string117:
	.asciz	"_ZN2t6nwIiEEPvmT_"             # string offset=1614
.Linfo_string118:
	.asciz	"operator new<int>"             # string offset=1632
.Linfo_string119:
	.asciz	"size_t"                        # string offset=1650
.Linfo_string120:
	.asciz	"_ZN2t6naIiEEPvmT_"             # string offset=1657
.Linfo_string121:
	.asciz	"operator new[]<int>"           # string offset=1675
.Linfo_string122:
	.asciz	"_ZN2t6dlIiEEvPvT_"             # string offset=1695
.Linfo_string123:
	.asciz	"operator delete<int>"          # string offset=1713
.Linfo_string124:
	.asciz	"_ZN2t6daIiEEvPvT_"             # string offset=1734
.Linfo_string125:
	.asciz	"operator delete[]<int>"        # string offset=1752
.Linfo_string126:
	.asciz	"_ZN2t6awIiEEiv"                # string offset=1775
.Linfo_string127:
	.asciz	"operator co_await<int>"        # string offset=1790
.Linfo_string128:
	.asciz	"_STNt10|<void>"                # string offset=1813
.Linfo_string129:
	.asciz	"_ZN2t83memEv"                  # string offset=1828
.Linfo_string130:
	.asciz	"mem"                           # string offset=1841
.Linfo_string131:
	.asciz	"t8"                            # string offset=1845
.Linfo_string132:
	.asciz	"_Zli5_suffy"                   # string offset=1848
.Linfo_string133:
	.asciz	"operator\"\"_suff"             # string offset=1860
.Linfo_string134:
	.asciz	"main"                          # string offset=1876
.Linfo_string135:
	.asciz	"_Z2f1IJiEEvv"                  # string offset=1881
.Linfo_string136:
	.asciz	"_STNf1|<int>"                  # string offset=1894
.Linfo_string137:
	.asciz	"_Z2f1IJfEEvv"                  # string offset=1907
.Linfo_string138:
	.asciz	"_STNf1|<float>"                # string offset=1920
.Linfo_string139:
	.asciz	"_Z2f1IJbEEvv"                  # string offset=1935
.Linfo_string140:
	.asciz	"_STNf1|<bool>"                 # string offset=1948
.Linfo_string141:
	.asciz	"double"                        # string offset=1962
.Linfo_string142:
	.asciz	"_Z2f1IJdEEvv"                  # string offset=1969
.Linfo_string143:
	.asciz	"_STNf1|<double>"               # string offset=1982
.Linfo_string144:
	.asciz	"_Z2f1IJlEEvv"                  # string offset=1998
.Linfo_string145:
	.asciz	"_STNf1|<long>"                 # string offset=2011
.Linfo_string146:
	.asciz	"_Z2f1IJsEEvv"                  # string offset=2025
.Linfo_string147:
	.asciz	"_STNf1|<short>"                # string offset=2038
.Linfo_string148:
	.asciz	"_Z2f1IJjEEvv"                  # string offset=2053
.Linfo_string149:
	.asciz	"_STNf1|<unsigned int>"         # string offset=2066
.Linfo_string150:
	.asciz	"unsigned long long"            # string offset=2088
.Linfo_string151:
	.asciz	"_Z2f1IJyEEvv"                  # string offset=2107
.Linfo_string152:
	.asciz	"_STNf1|<unsigned long long>"   # string offset=2120
.Linfo_string153:
	.asciz	"long long"                     # string offset=2148
.Linfo_string154:
	.asciz	"_Z2f1IJxEEvv"                  # string offset=2158
.Linfo_string155:
	.asciz	"_STNf1|<long long>"            # string offset=2171
.Linfo_string156:
	.asciz	"udt"                           # string offset=2190
.Linfo_string157:
	.asciz	"_Z2f1IJ3udtEEvv"               # string offset=2194
.Linfo_string158:
	.asciz	"_STNf1|<udt>"                  # string offset=2210
.Linfo_string159:
	.asciz	"_Z2f1IJN2ns3udtEEEvv"          # string offset=2223
.Linfo_string160:
	.asciz	"_STNf1|<ns::udt>"              # string offset=2244
.Linfo_string161:
	.asciz	"_Z2f1IJPN2ns3udtEEEvv"         # string offset=2261
.Linfo_string162:
	.asciz	"_STNf1|<ns::udt *>"            # string offset=2283
.Linfo_string163:
	.asciz	"inner"                         # string offset=2302
.Linfo_string164:
	.asciz	"_Z2f1IJN2ns5inner3udtEEEvv"    # string offset=2308
.Linfo_string165:
	.asciz	"_STNf1|<ns::inner::udt>"       # string offset=2335
.Linfo_string166:
	.asciz	"_STNt1|<int>"                  # string offset=2359
.Linfo_string167:
	.asciz	"_Z2f1IJ2t1IJiEEEEvv"           # string offset=2372
.Linfo_string168:
	.asciz	"_STNf1|<t1<int> >"             # string offset=2392
.Linfo_string169:
	.asciz	"_Z2f1IJifEEvv"                 # string offset=2410
.Linfo_string170:
	.asciz	"_STNf1|<int, float>"           # string offset=2424
.Linfo_string171:
	.asciz	"_Z2f1IJPiEEvv"                 # string offset=2444
.Linfo_string172:
	.asciz	"_STNf1|<int *>"                # string offset=2458
.Linfo_string173:
	.asciz	"_Z2f1IJRiEEvv"                 # string offset=2473
.Linfo_string174:
	.asciz	"_STNf1|<int &>"                # string offset=2487
.Linfo_string175:
	.asciz	"_Z2f1IJOiEEvv"                 # string offset=2502
.Linfo_string176:
	.asciz	"_STNf1|<int &&>"               # string offset=2516
.Linfo_string177:
	.asciz	"_Z2f1IJKiEEvv"                 # string offset=2532
.Linfo_string178:
	.asciz	"_STNf1|<const int>"            # string offset=2546
.Linfo_string179:
	.asciz	"_Z2f1IJvEEvv"                  # string offset=2565
.Linfo_string180:
	.asciz	"_STNf1|<void>"                 # string offset=2578
.Linfo_string181:
	.asciz	"outer_class"                   # string offset=2592
.Linfo_string182:
	.asciz	"inner_class"                   # string offset=2604
.Linfo_string183:
	.asciz	"_Z2f1IJN11outer_class11inner_classEEEvv" # string offset=2616
.Linfo_string184:
	.asciz	"_STNf1|<outer_class::inner_class>" # string offset=2656
.Linfo_string185:
	.asciz	"_Z2f1IJmEEvv"                  # string offset=2690
.Linfo_string186:
	.asciz	"_STNf1|<unsigned long>"        # string offset=2703
.Linfo_string187:
	.asciz	"_Z2f2ILb1ELi3EEvv"             # string offset=2726
.Linfo_string188:
	.asciz	"_STNf2|<true, 3>"              # string offset=2744
.Linfo_string189:
	.asciz	"A"                             # string offset=2761
.Linfo_string190:
	.asciz	"_Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv" # string offset=2763
.Linfo_string191:
	.asciz	"_STNf3|<ns::Enumeration, ns::Enumerator2, (ns::Enumeration)2>" # string offset=2805
.Linfo_string192:
	.asciz	"_Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv" # string offset=2867
.Linfo_string193:
	.asciz	"_STNf3|<ns::EnumerationClass, ns::EnumerationClass::Enumerator2, (ns::EnumerationClass)2>" # string offset=2914
.Linfo_string194:
	.asciz	"_Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv" # string offset=3004
.Linfo_string195:
	.asciz	"_STNf3|<ns::EnumerationSmall, ns::kNeg>" # string offset=3047
.Linfo_string196:
	.asciz	"_Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv" # string offset=3087
.Linfo_string197:
	.asciz	"f3<ns::(unnamed enum at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:22:1), ns::AnonEnum2, (ns::(unnamed enum at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:22:1))2>" # string offset=3120
.Linfo_string198:
	.asciz	"_Z2f3IPiJXadL_Z1iEEEEvv"       # string offset=3371
.Linfo_string199:
	.asciz	"f3<int *, &i>"                 # string offset=3395
.Linfo_string200:
	.asciz	"_Z2f3IPiJLS0_0EEEvv"           # string offset=3409
.Linfo_string201:
	.asciz	"f3<int *, nullptr>"            # string offset=3429
.Linfo_string202:
	.asciz	"_Z2f3ImJLm1EEEvv"              # string offset=3448
.Linfo_string203:
	.asciz	"_STNf3|<unsigned long, 1UL>"   # string offset=3465
.Linfo_string204:
	.asciz	"_Z2f3IyJLy1EEEvv"              # string offset=3493
.Linfo_string205:
	.asciz	"_STNf3|<unsigned long long, 1ULL>" # string offset=3510
.Linfo_string206:
	.asciz	"_Z2f3IlJLl1EEEvv"              # string offset=3544
.Linfo_string207:
	.asciz	"_STNf3|<long, 1L>"             # string offset=3561
.Linfo_string208:
	.asciz	"_Z2f3IjJLj1EEEvv"              # string offset=3579
.Linfo_string209:
	.asciz	"_STNf3|<unsigned int, 1U>"     # string offset=3596
.Linfo_string210:
	.asciz	"_Z2f3IsJLs1EEEvv"              # string offset=3622
.Linfo_string211:
	.asciz	"_STNf3|<short, (short)1>"      # string offset=3639
.Linfo_string212:
	.asciz	"_Z2f3IhJLh0EEEvv"              # string offset=3664
.Linfo_string213:
	.asciz	"_STNf3|<unsigned char, (unsigned char)'\\x00'>" # string offset=3681
.Linfo_string214:
	.asciz	"_Z2f3IaJLa0EEEvv"              # string offset=3727
.Linfo_string215:
	.asciz	"_STNf3|<signed char, (signed char)'\\x00'>" # string offset=3744
.Linfo_string216:
	.asciz	"_Z2f3ItJLt1ELt2EEEvv"          # string offset=3786
.Linfo_string217:
	.asciz	"_STNf3|<unsigned short, (unsigned short)1, (unsigned short)2>" # string offset=3807
.Linfo_string218:
	.asciz	"char"                          # string offset=3869
.Linfo_string219:
	.asciz	"_Z2f3IcJLc0ELc1ELc6ELc7ELc13ELc14ELc31ELc32ELc33ELc127ELcn128EEEvv" # string offset=3874
.Linfo_string220:
	.asciz	"_STNf3|<char, '\\x00', '\\x01', '\\x06', '\\a', '\\r', '\\x0e', '\\x1f', ' ', '!', '\\x7f', '\\x80'>" # string offset=3941
.Linfo_string221:
	.asciz	"__int128"                      # string offset=4033
.Linfo_string222:
	.asciz	"_Z2f3InJLn18446744073709551614EEEvv" # string offset=4042
.Linfo_string223:
	.asciz	"f3<__int128, (__int128)18446744073709551614>" # string offset=4078
.Linfo_string224:
	.asciz	"_Z2f4IjLj3EEvv"                # string offset=4123
.Linfo_string225:
	.asciz	"_STNf4|<unsigned int, 3U>"     # string offset=4138
.Linfo_string226:
	.asciz	"_Z2f1IJ2t3IiLb0EEEEvv"         # string offset=4164
.Linfo_string227:
	.asciz	"_STNf1|<t3<int, false> >"      # string offset=4186
.Linfo_string228:
	.asciz	"_STNt3|<t3<int, false>, false>" # string offset=4211
.Linfo_string229:
	.asciz	"_Z2f1IJ2t3IS0_IiLb0EELb0EEEEvv" # string offset=4242
.Linfo_string230:
	.asciz	"_STNf1|<t3<t3<int, false>, false> >" # string offset=4273
.Linfo_string231:
	.asciz	"_Z2f1IJZ4mainE3$_1EEvv"        # string offset=4309
.Linfo_string232:
	.asciz	"f1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:168:12)>" # string offset=4332
.Linfo_string233:
	.asciz	"_Z2f1IJFifEEEvv"               # string offset=4442
.Linfo_string234:
	.asciz	"_STNf1|<int (float)>"          # string offset=4458
.Linfo_string235:
	.asciz	"_Z2f1IJRKiEEvv"                # string offset=4479
.Linfo_string236:
	.asciz	"_STNf1|<const int &>"          # string offset=4494
.Linfo_string237:
	.asciz	"_Z2f1IJRPKiEEvv"               # string offset=4515
.Linfo_string238:
	.asciz	"_STNf1|<const int *&>"         # string offset=4531
.Linfo_string239:
	.asciz	"t5"                            # string offset=4553
.Linfo_string240:
	.asciz	"_Z2f1IJN12_GLOBAL__N_12t5EEEvv" # string offset=4556
.Linfo_string241:
	.asciz	"_STNf1|<(anonymous namespace)::t5>" # string offset=4587
.Linfo_string242:
	.asciz	"decltype(nullptr)"             # string offset=4622
.Linfo_string243:
	.asciz	"_Z2f1IJDnEEvv"                 # string offset=4640
.Linfo_string244:
	.asciz	"_STNf1|<std::nullptr_t>"       # string offset=4654
.Linfo_string245:
	.asciz	"_Z2f1IJPlS0_EEvv"              # string offset=4678
.Linfo_string246:
	.asciz	"_STNf1|<long *, long *>"       # string offset=4695
.Linfo_string247:
	.asciz	"_Z2f1IJPlP3udtEEvv"            # string offset=4719
.Linfo_string248:
	.asciz	"_STNf1|<long *, udt *>"        # string offset=4738
.Linfo_string249:
	.asciz	"_Z2f1IJKPvEEvv"                # string offset=4761
.Linfo_string250:
	.asciz	"_STNf1|<void *const>"          # string offset=4776
.Linfo_string251:
	.asciz	"_Z2f1IJPKPKvEEvv"              # string offset=4797
.Linfo_string252:
	.asciz	"_STNf1|<const void *const *>"  # string offset=4814
.Linfo_string253:
	.asciz	"_Z2f1IJFvvEEEvv"               # string offset=4843
.Linfo_string254:
	.asciz	"_STNf1|<void ()>"              # string offset=4859
.Linfo_string255:
	.asciz	"_Z2f1IJPFvvEEEvv"              # string offset=4876
.Linfo_string256:
	.asciz	"_STNf1|<void (*)()>"           # string offset=4893
.Linfo_string257:
	.asciz	"_Z2f1IJPZ4mainE3$_1EEvv"       # string offset=4913
.Linfo_string258:
	.asciz	"f1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:168:12) *>" # string offset=4937
.Linfo_string259:
	.asciz	"_Z2f1IJZ4mainE3$_2EEvv"        # string offset=5049
.Linfo_string260:
	.asciz	"f1<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:167:3)>" # string offset=5072
.Linfo_string261:
	.asciz	"_Z2f1IJPZ4mainE3$_2EEvv"       # string offset=5189
.Linfo_string262:
	.asciz	"f1<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:167:3) *>" # string offset=5213
.Linfo_string263:
	.asciz	"T1"                            # string offset=5332
.Linfo_string264:
	.asciz	"T2"                            # string offset=5335
.Linfo_string265:
	.asciz	"_Z2f5IJ2t1IJiEEEiEvv"          # string offset=5338
.Linfo_string266:
	.asciz	"_STNf5|<t1<int>, int>"         # string offset=5359
.Linfo_string267:
	.asciz	"_Z2f5IJEiEvv"                  # string offset=5381
.Linfo_string268:
	.asciz	"_STNf5|<int>"                  # string offset=5394
.Linfo_string269:
	.asciz	"_Z2f6I2t1IJiEEJEEvv"           # string offset=5407
.Linfo_string270:
	.asciz	"_STNf6|<t1<int> >"             # string offset=5427
.Linfo_string271:
	.asciz	"_Z2f1IJEEvv"                   # string offset=5445
.Linfo_string272:
	.asciz	"_STNf1|<>"                     # string offset=5457
.Linfo_string273:
	.asciz	"_Z2f1IJPKvS1_EEvv"             # string offset=5467
.Linfo_string274:
	.asciz	"_STNf1|<const void *, const void *>" # string offset=5485
.Linfo_string275:
	.asciz	"_STNt1|<int *>"                # string offset=5521
.Linfo_string276:
	.asciz	"_Z2f1IJP2t1IJPiEEEEvv"         # string offset=5536
.Linfo_string277:
	.asciz	"_STNf1|<t1<int *> *>"          # string offset=5558
.Linfo_string278:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=5579
.Linfo_string279:
	.asciz	"_Z2f1IJA_PiEEvv"               # string offset=5599
.Linfo_string280:
	.asciz	"_STNf1|<int *[]>"              # string offset=5615
.Linfo_string281:
	.asciz	"t7"                            # string offset=5632
.Linfo_string282:
	.asciz	"_Z2f1IJZ4mainE2t7EEvv"         # string offset=5635
.Linfo_string283:
	.asciz	"_STNf1|<t7>"                   # string offset=5657
.Linfo_string284:
	.asciz	"_Z2f1IJRA3_iEEvv"              # string offset=5669
.Linfo_string285:
	.asciz	"_STNf1|<int (&)[3]>"           # string offset=5686
.Linfo_string286:
	.asciz	"_Z2f1IJPA3_iEEvv"              # string offset=5706
.Linfo_string287:
	.asciz	"_STNf1|<int (*)[3]>"           # string offset=5723
.Linfo_string288:
	.asciz	"t1"                            # string offset=5743
.Linfo_string289:
	.asciz	"_Z2f7I2t1Evv"                  # string offset=5746
.Linfo_string290:
	.asciz	"_STNf7|<t1>"                   # string offset=5759
.Linfo_string291:
	.asciz	"_Z2f8I2t1iEvv"                 # string offset=5771
.Linfo_string292:
	.asciz	"_STNf8|<t1, int>"              # string offset=5785
.Linfo_string293:
	.asciz	"ns::inner::ttp"                # string offset=5802
.Linfo_string294:
	.asciz	"_ZN2ns8ttp_userINS_5inner3ttpEEEvv" # string offset=5817
.Linfo_string295:
	.asciz	"_STNttp_user|<ns::inner::ttp>" # string offset=5852
.Linfo_string296:
	.asciz	"_Z2f1IJPiPDnEEvv"              # string offset=5882
.Linfo_string297:
	.asciz	"_STNf1|<int *, std::nullptr_t *>" # string offset=5899
.Linfo_string298:
	.asciz	"_STNt7|<int>"                  # string offset=5932
.Linfo_string299:
	.asciz	"_Z2f1IJ2t7IiEEEvv"             # string offset=5945
.Linfo_string300:
	.asciz	"_STNf1|<t7<int> >"             # string offset=5963
.Linfo_string301:
	.asciz	"ns::inl::t9"                   # string offset=5981
.Linfo_string302:
	.asciz	"_Z2f7IN2ns3inl2t9EEvv"         # string offset=5993
.Linfo_string303:
	.asciz	"_STNf7|<ns::inl::t9>"          # string offset=6015
.Linfo_string304:
	.asciz	"_Z2f1IJU7_AtomiciEEvv"         # string offset=6036
.Linfo_string305:
	.asciz	"f1<_Atomic(int)>"              # string offset=6058
.Linfo_string306:
	.asciz	"_Z2f1IJilVcEEvv"               # string offset=6075
.Linfo_string307:
	.asciz	"_STNf1|<int, long, volatile char>" # string offset=6091
.Linfo_string308:
	.asciz	"_Z2f1IJDv2_iEEvv"              # string offset=6125
.Linfo_string309:
	.asciz	"f1<__attribute__((__vector_size__(2 * sizeof(int)))) int>" # string offset=6142
.Linfo_string310:
	.asciz	"_Z2f1IJVKPiEEvv"               # string offset=6200
.Linfo_string311:
	.asciz	"_STNf1|<int *const volatile>"  # string offset=6216
.Linfo_string312:
	.asciz	"_Z2f1IJVKvEEvv"                # string offset=6245
.Linfo_string313:
	.asciz	"_STNf1|<const volatile void>"  # string offset=6260
.Linfo_string314:
	.asciz	"t1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:168:12)>" # string offset=6289
.Linfo_string315:
	.asciz	"_Z2f1IJ2t1IJZ4mainE3$_1EEEEvv" # string offset=6399
.Linfo_string316:
	.asciz	"_STNf1|<t1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:168:12)> >" # string offset=6429
.Linfo_string317:
	.asciz	"_ZN3t10C2IvEEv"                # string offset=6549
.Linfo_string318:
	.asciz	"_Z2f1IJM3udtKFvvEEEvv"         # string offset=6564
.Linfo_string319:
	.asciz	"_STNf1|<void (udt::*)() const>" # string offset=6586
.Linfo_string320:
	.asciz	"_Z2f1IJM3udtVFvvREEEvv"        # string offset=6617
.Linfo_string321:
	.asciz	"_STNf1|<void (udt::*)() volatile &>" # string offset=6640
.Linfo_string322:
	.asciz	"_Z2f1IJM3udtVKFvvOEEEvv"       # string offset=6676
.Linfo_string323:
	.asciz	"_STNf1|<void (udt::*)() const volatile &&>" # string offset=6700
.Linfo_string324:
	.asciz	"_Z2f9IiEPFvvEv"                # string offset=6743
.Linfo_string325:
	.asciz	"_STNf9|<int>"                  # string offset=6758
.Linfo_string326:
	.asciz	"_Z2f1IJKPFvvEEEvv"             # string offset=6771
.Linfo_string327:
	.asciz	"_STNf1|<void (*const)()>"      # string offset=6789
.Linfo_string328:
	.asciz	"_Z2f1IJRA1_KcEEvv"             # string offset=6814
.Linfo_string329:
	.asciz	"_STNf1|<const char (&)[1]>"    # string offset=6832
.Linfo_string330:
	.asciz	"_Z2f1IJKFvvREEEvv"             # string offset=6859
.Linfo_string331:
	.asciz	"_STNf1|<void () const &>"      # string offset=6877
.Linfo_string332:
	.asciz	"_Z2f1IJVFvvOEEEvv"             # string offset=6902
.Linfo_string333:
	.asciz	"_STNf1|<void () volatile &&>"  # string offset=6920
.Linfo_string334:
	.asciz	"_Z2f1IJVKFvvEEEvv"             # string offset=6949
.Linfo_string335:
	.asciz	"_STNf1|<void () const volatile>" # string offset=6967
.Linfo_string336:
	.asciz	"_Z2f1IJA1_KPiEEvv"             # string offset=6999
.Linfo_string337:
	.asciz	"_STNf1|<int *const[1]>"        # string offset=7017
.Linfo_string338:
	.asciz	"_Z2f1IJRA1_KPiEEvv"            # string offset=7040
.Linfo_string339:
	.asciz	"_STNf1|<int *const (&)[1]>"    # string offset=7059
.Linfo_string340:
	.asciz	"_Z2f1IJRKM3udtFvvEEEvv"        # string offset=7086
.Linfo_string341:
	.asciz	"_STNf1|<void (udt::*const &)()>" # string offset=7109
.Linfo_string342:
	.asciz	"_Z2f1IJFPFvfEiEEEvv"           # string offset=7141
.Linfo_string343:
	.asciz	"_STNf1|<void (*(int))(float)>" # string offset=7161
.Linfo_string344:
	.asciz	"_Z2f1IJPDoFvvEEEvv"            # string offset=7191
.Linfo_string345:
	.asciz	"f1<void (*)() noexcept>"       # string offset=7210
.Linfo_string346:
	.asciz	"_Z2f1IJFvZ4mainE3$_2EEEvv"     # string offset=7234
.Linfo_string347:
	.asciz	"f1<void ((unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:167:3))>" # string offset=7260
.Linfo_string348:
	.asciz	"_Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv" # string offset=7384
.Linfo_string349:
	.asciz	"f1<void (t8, (unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:167:3))>" # string offset=7420
.Linfo_string350:
	.asciz	"_Z2f1IJFvZ4mainE2t8EEEvv"      # string offset=7548
.Linfo_string351:
	.asciz	"_STNf1|<void (t8)>"            # string offset=7573
.Linfo_string352:
	.asciz	"_Z19operator_not_reallyIiEvv"  # string offset=7592
.Linfo_string353:
	.asciz	"_STNoperator_not_really|<int>" # string offset=7621
.Linfo_string354:
	.asciz	"_Z2f1IJZN2t83memEvE2t7EEvv"    # string offset=7651
.Linfo_string355:
	.asciz	"_Z2f1IJM2t8FvvEEEvv"           # string offset=7678
.Linfo_string356:
	.asciz	"_STNf1|<void (t8::*)()>"       # string offset=7698
.Linfo_string357:
	.asciz	"L"                             # string offset=7722
.Linfo_string358:
	.asciz	"v2"                            # string offset=7724
.Linfo_string359:
	.asciz	"N"                             # string offset=7727
.Linfo_string360:
	.asciz	"_STNt4|<3U>"                   # string offset=7729
.Linfo_string361:
	.asciz	"v1"                            # string offset=7741
.Linfo_string362:
	.asciz	"t3<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:168:12), false>" # string offset=7744
.Linfo_string363:
	.asciz	"v6"                            # string offset=7861
.Linfo_string364:
	.asciz	"x"                             # string offset=7864
.Linfo_string365:
	.asciz	"t7i"                           # string offset=7866
.Linfo_string366:
	.asciz	"v3"                            # string offset=7870
.Linfo_string367:
	.asciz	"_STNt2|<int>"                  # string offset=7873
.Linfo_string368:
	.asciz	"_STNt2|<float>"                # string offset=7886
.Linfo_string369:
	.asciz	"_STNt1|<bool>"                 # string offset=7901
.Linfo_string370:
	.asciz	"_STNt2|<bool>"                 # string offset=7915
.Linfo_string371:
	.asciz	"_STNt1|<double>"               # string offset=7929
.Linfo_string372:
	.asciz	"_STNt2|<double>"               # string offset=7945
.Linfo_string373:
	.asciz	"_STNt1|<long>"                 # string offset=7961
.Linfo_string374:
	.asciz	"_STNt2|<long>"                 # string offset=7975
.Linfo_string375:
	.asciz	"_STNt1|<short>"                # string offset=7989
.Linfo_string376:
	.asciz	"_STNt2|<short>"                # string offset=8004
.Linfo_string377:
	.asciz	"_STNt1|<unsigned int>"         # string offset=8019
.Linfo_string378:
	.asciz	"_STNt2|<unsigned int>"         # string offset=8041
.Linfo_string379:
	.asciz	"_STNt1|<unsigned long long>"   # string offset=8063
.Linfo_string380:
	.asciz	"_STNt2|<unsigned long long>"   # string offset=8091
.Linfo_string381:
	.asciz	"_STNt1|<long long>"            # string offset=8119
.Linfo_string382:
	.asciz	"_STNt2|<long long>"            # string offset=8138
.Linfo_string383:
	.asciz	"_STNt1|<udt>"                  # string offset=8157
.Linfo_string384:
	.asciz	"_STNt2|<udt>"                  # string offset=8170
.Linfo_string385:
	.asciz	"_STNt1|<ns::udt>"              # string offset=8183
.Linfo_string386:
	.asciz	"_STNt2|<ns::udt>"              # string offset=8200
.Linfo_string387:
	.asciz	"_STNt1|<ns::udt *>"            # string offset=8217
.Linfo_string388:
	.asciz	"_STNt2|<ns::udt *>"            # string offset=8236
.Linfo_string389:
	.asciz	"_STNt1|<ns::inner::udt>"       # string offset=8255
.Linfo_string390:
	.asciz	"_STNt2|<ns::inner::udt>"       # string offset=8279
.Linfo_string391:
	.asciz	"_STNt1|<t1<int> >"             # string offset=8303
.Linfo_string392:
	.asciz	"_STNt2|<t1<int> >"             # string offset=8321
.Linfo_string393:
	.asciz	"_STNt1|<int, float>"           # string offset=8339
.Linfo_string394:
	.asciz	"_STNt2|<int, float>"           # string offset=8359
.Linfo_string395:
	.asciz	"_STNt2|<int *>"                # string offset=8379
.Linfo_string396:
	.asciz	"_STNt1|<int &>"                # string offset=8394
.Linfo_string397:
	.asciz	"_STNt2|<int &>"                # string offset=8409
.Linfo_string398:
	.asciz	"_STNt1|<int &&>"               # string offset=8424
.Linfo_string399:
	.asciz	"_STNt2|<int &&>"               # string offset=8440
.Linfo_string400:
	.asciz	"_STNt1|<const int>"            # string offset=8456
.Linfo_string401:
	.asciz	"_STNt2|<const int>"            # string offset=8475
.Linfo_string402:
	.asciz	"_STNt1|<void>"                 # string offset=8494
.Linfo_string403:
	.asciz	"_STNt2|<void>"                 # string offset=8508
.Linfo_string404:
	.asciz	"_STNt1|<outer_class::inner_class>" # string offset=8522
.Linfo_string405:
	.asciz	"_STNt2|<outer_class::inner_class>" # string offset=8556
.Linfo_string406:
	.asciz	"_STNt1|<unsigned long>"        # string offset=8590
.Linfo_string407:
	.asciz	"_STNt2|<unsigned long>"        # string offset=8613
.Linfo_string408:
	.asciz	"_STNt1|<t3<int, false> >"      # string offset=8636
.Linfo_string409:
	.asciz	"_STNt2|<t3<int, false> >"      # string offset=8661
.Linfo_string410:
	.asciz	"_STNt1|<t3<t3<int, false>, false> >" # string offset=8686
.Linfo_string411:
	.asciz	"_STNt2|<t3<t3<int, false>, false> >" # string offset=8722
.Linfo_string412:
	.asciz	"t2<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:168:12)>" # string offset=8758
.Linfo_string413:
	.asciz	"_STNt1|<int (float)>"          # string offset=8868
.Linfo_string414:
	.asciz	"_STNt2|<int (float)>"          # string offset=8889
.Linfo_string415:
	.asciz	"_STNt1|<const int &>"          # string offset=8910
.Linfo_string416:
	.asciz	"_STNt2|<const int &>"          # string offset=8931
.Linfo_string417:
	.asciz	"_STNt1|<const int *&>"         # string offset=8952
.Linfo_string418:
	.asciz	"_STNt2|<const int *&>"         # string offset=8974
.Linfo_string419:
	.asciz	"_STNt1|<(anonymous namespace)::t5>" # string offset=8996
.Linfo_string420:
	.asciz	"_STNt2|<(anonymous namespace)::t5>" # string offset=9031
.Linfo_string421:
	.asciz	"_STNt1|<std::nullptr_t>"       # string offset=9066
.Linfo_string422:
	.asciz	"_STNt2|<std::nullptr_t>"       # string offset=9090
.Linfo_string423:
	.asciz	"_STNt1|<long *, long *>"       # string offset=9114
.Linfo_string424:
	.asciz	"_STNt2|<long *, long *>"       # string offset=9138
.Linfo_string425:
	.asciz	"_STNt1|<long *, udt *>"        # string offset=9162
.Linfo_string426:
	.asciz	"_STNt2|<long *, udt *>"        # string offset=9185
.Linfo_string427:
	.asciz	"_STNt1|<void *const>"          # string offset=9208
.Linfo_string428:
	.asciz	"_STNt2|<void *const>"          # string offset=9229
.Linfo_string429:
	.asciz	"_STNt1|<const void *const *>"  # string offset=9250
.Linfo_string430:
	.asciz	"_STNt2|<const void *const *>"  # string offset=9279
.Linfo_string431:
	.asciz	"_STNt1|<void ()>"              # string offset=9308
.Linfo_string432:
	.asciz	"_STNt2|<void ()>"              # string offset=9325
.Linfo_string433:
	.asciz	"_STNt1|<void (*)()>"           # string offset=9342
.Linfo_string434:
	.asciz	"_STNt2|<void (*)()>"           # string offset=9362
.Linfo_string435:
	.asciz	"t1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:168:12) *>" # string offset=9382
.Linfo_string436:
	.asciz	"t2<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:168:12) *>" # string offset=9494
.Linfo_string437:
	.asciz	"t1<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:167:3)>" # string offset=9606
.Linfo_string438:
	.asciz	"t2<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:167:3)>" # string offset=9723
.Linfo_string439:
	.asciz	"t1<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:167:3) *>" # string offset=9840
.Linfo_string440:
	.asciz	"t2<(unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:167:3) *>" # string offset=9959
.Linfo_string441:
	.asciz	"_STNt1|<>"                     # string offset=10078
.Linfo_string442:
	.asciz	"_STNt2|<>"                     # string offset=10088
.Linfo_string443:
	.asciz	"_STNt1|<const void *, const void *>" # string offset=10098
.Linfo_string444:
	.asciz	"_STNt2|<const void *, const void *>" # string offset=10134
.Linfo_string445:
	.asciz	"_STNt1|<t1<int *> *>"          # string offset=10170
.Linfo_string446:
	.asciz	"_STNt2|<t1<int *> *>"          # string offset=10191
.Linfo_string447:
	.asciz	"_STNt1|<int *[]>"              # string offset=10212
.Linfo_string448:
	.asciz	"_STNt2|<int *[]>"              # string offset=10229
.Linfo_string449:
	.asciz	"this"                          # string offset=10246
.Linfo_string450:
	.asciz	"_STNt1|<t7>"                   # string offset=10251
.Linfo_string451:
	.asciz	"_STNt2|<t7>"                   # string offset=10263
.Linfo_string452:
	.asciz	"_STNt1|<int (&)[3]>"           # string offset=10275
.Linfo_string453:
	.asciz	"_STNt2|<int (&)[3]>"           # string offset=10295
.Linfo_string454:
	.asciz	"_STNt1|<int (*)[3]>"           # string offset=10315
.Linfo_string455:
	.asciz	"_STNt2|<int (*)[3]>"           # string offset=10335
.Linfo_string456:
	.asciz	"_STNt1|<int *, std::nullptr_t *>" # string offset=10355
.Linfo_string457:
	.asciz	"_STNt2|<int *, std::nullptr_t *>" # string offset=10388
.Linfo_string458:
	.asciz	"_STNt1|<t7<int> >"             # string offset=10421
.Linfo_string459:
	.asciz	"_STNt2|<t7<int> >"             # string offset=10439
.Linfo_string460:
	.asciz	"t1<_Atomic(int)>"              # string offset=10457
.Linfo_string461:
	.asciz	"t2<_Atomic(int)>"              # string offset=10474
.Linfo_string462:
	.asciz	"_STNt1|<int, long, volatile char>" # string offset=10491
.Linfo_string463:
	.asciz	"_STNt2|<int, long, volatile char>" # string offset=10525
.Linfo_string464:
	.asciz	"t1<__attribute__((__vector_size__(2 * sizeof(int)))) int>" # string offset=10559
.Linfo_string465:
	.asciz	"t2<__attribute__((__vector_size__(2 * sizeof(int)))) int>" # string offset=10617
.Linfo_string466:
	.asciz	"_STNt1|<int *const volatile>"  # string offset=10675
.Linfo_string467:
	.asciz	"_STNt2|<int *const volatile>"  # string offset=10704
.Linfo_string468:
	.asciz	"_STNt1|<const volatile void>"  # string offset=10733
.Linfo_string469:
	.asciz	"_STNt2|<const volatile void>"  # string offset=10762
.Linfo_string470:
	.asciz	"_STNt1|<t1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:168:12)> >" # string offset=10791
.Linfo_string471:
	.asciz	"_STNt2|<t1<(lambda at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:168:12)> >" # string offset=10911
.Linfo_string472:
	.asciz	"_STNt1|<void (udt::*)() const>" # string offset=11031
.Linfo_string473:
	.asciz	"_STNt2|<void (udt::*)() const>" # string offset=11062
.Linfo_string474:
	.asciz	"_STNt1|<void (udt::*)() volatile &>" # string offset=11093
.Linfo_string475:
	.asciz	"_STNt2|<void (udt::*)() volatile &>" # string offset=11129
.Linfo_string476:
	.asciz	"_STNt1|<void (udt::*)() const volatile &&>" # string offset=11165
.Linfo_string477:
	.asciz	"_STNt2|<void (udt::*)() const volatile &&>" # string offset=11208
.Linfo_string478:
	.asciz	"_STNt1|<void (*const)()>"      # string offset=11251
.Linfo_string479:
	.asciz	"_STNt2|<void (*const)()>"      # string offset=11276
.Linfo_string480:
	.asciz	"_STNt1|<const char (&)[1]>"    # string offset=11301
.Linfo_string481:
	.asciz	"_STNt2|<const char (&)[1]>"    # string offset=11328
.Linfo_string482:
	.asciz	"_STNt1|<void () const &>"      # string offset=11355
.Linfo_string483:
	.asciz	"_STNt2|<void () const &>"      # string offset=11380
.Linfo_string484:
	.asciz	"_STNt1|<void () volatile &&>"  # string offset=11405
.Linfo_string485:
	.asciz	"_STNt2|<void () volatile &&>"  # string offset=11434
.Linfo_string486:
	.asciz	"_STNt1|<void () const volatile>" # string offset=11463
.Linfo_string487:
	.asciz	"_STNt2|<void () const volatile>" # string offset=11495
.Linfo_string488:
	.asciz	"_STNt1|<int *const[1]>"        # string offset=11527
.Linfo_string489:
	.asciz	"_STNt2|<int *const[1]>"        # string offset=11550
.Linfo_string490:
	.asciz	"_STNt1|<int *const (&)[1]>"    # string offset=11573
.Linfo_string491:
	.asciz	"_STNt2|<int *const (&)[1]>"    # string offset=11600
.Linfo_string492:
	.asciz	"_STNt1|<void (udt::*const &)()>" # string offset=11627
.Linfo_string493:
	.asciz	"_STNt2|<void (udt::*const &)()>" # string offset=11659
.Linfo_string494:
	.asciz	"_STNt1|<void (*(int))(float)>" # string offset=11691
.Linfo_string495:
	.asciz	"_STNt2|<void (*(int))(float)>" # string offset=11721
.Linfo_string496:
	.asciz	"t1<void (*)() noexcept>"       # string offset=11751
.Linfo_string497:
	.asciz	"t2<void (*)() noexcept>"       # string offset=11775
.Linfo_string498:
	.asciz	"t1<void ((unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:167:3))>" # string offset=11799
.Linfo_string499:
	.asciz	"t2<void ((unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:167:3))>" # string offset=11923
.Linfo_string500:
	.asciz	"t1<void (t8, (unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:167:3))>" # string offset=12047
.Linfo_string501:
	.asciz	"t2<void (t8, (unnamed struct at cross-project-tests/debuginfo-tests/clang_llvm_roundtrip/simplified_template_names.cpp:167:3))>" # string offset=12175
.Linfo_string502:
	.asciz	"_STNt1|<void (t8)>"            # string offset=12303
.Linfo_string503:
	.asciz	"_STNt2|<void (t8)>"            # string offset=12322
.Linfo_string504:
	.asciz	"_STNt1|<void (t8::*)()>"       # string offset=12341
.Linfo_string505:
	.asciz	"_STNt2|<void (t8::*)()>"       # string offset=12365
	.ident	"clang version 14.0.0 (git@github.com:llvm/llvm-project.git 6d48e2505c7a68a470e75b61ad504d51db0f8a36)"
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
	.addrsig_sym _Z2f1IJvEEvv
	.addrsig_sym _Z2f1IJN11outer_class11inner_classEEEvv
	.addrsig_sym _Z2f1IJmEEvv
	.addrsig_sym _Z2f2ILb1ELi3EEvv
	.addrsig_sym _Z2f3IN2ns11EnumerationEJLS1_1ELS1_2EEEvv
	.addrsig_sym _Z2f3IN2ns16EnumerationClassEJLS1_1ELS1_2EEEvv
	.addrsig_sym _Z2f3IN2ns16EnumerationSmallEJLS1_255EEEvv
	.addrsig_sym _Z2f3IN2ns3$_0EJLS1_1ELS1_2EEEvv
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
	.addrsig_sym _Z2f1IJFifEEEvv
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
	.addrsig_sym _Z2f1IJPDoFvvEEEvv
	.addrsig_sym _Z2f1IJFvZ4mainE3$_2EEEvv
	.addrsig_sym _Z2f1IJFvZ4mainE2t8Z4mainE3$_2EEEvv
	.addrsig_sym _Z2f1IJFvZ4mainE2t8EEEvv
	.addrsig_sym _Z19operator_not_reallyIiEvv
	.addrsig_sym _Z2f1IJZN2t83memEvE2t7EEvv
	.addrsig_sym _Z2f1IJM2t8FvvEEEvv
	.section	.debug_line,"",@progbits
.Lline_table_start0:
