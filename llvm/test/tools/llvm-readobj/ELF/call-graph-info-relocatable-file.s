## Tests --call-graph-info prints information from call graph section.

# REQUIRES: x86-registered-target

# RUN: llvm-mc %s -filetype=obj -triple=x86_64-pc-linux -o %t
# RUN: llvm-readelf --call-graph-info %t 2>&1 | FileCheck %s --allow-empty -DFILE=%t
# RUN: llvm-readelf --elf-output-style=LLVM --call-graph-info %t 2>&1 | FileCheck %s -DFILE=%t --check-prefix=LLVM
# RUN: llvm-readelf --elf-output-style=JSON --pretty-print --call-graph-info %t 2>&1 | FileCheck %s -DFILE=%t --check-prefix=JSON

## Assembly output was generated from the following source using this command
##      clang -fexperimental-call-graph-section -S test.cpp -o test.S
##
## void foo() {}
## 
## void bar() {}
## 
## int baz(char a) { return 0; }
## 
## int caller() {
##   void (*fp_foo)() = foo;
##   fp_foo();
## 
##   void (*fp_bar)() = bar;
##   fp_bar();
## 
##   char a;
##   int (*fp_baz)(char) = baz;
##   fp_baz(a);
##  
##   foo();
##   bar();
##   baz(a);
## 
##   return 0;
## }
##

## We do not support GNU format console output for --call-graph-info as it is an LLVM only info.
# CHECK-NOT: .

# LLVM: 		CallGraph [
# LLVM-NEXT:   Function {
# LLVM-NEXT:     Name: _Z3foov
# LLVM-NEXT:     Version: 0
# LLVM-NEXT:     IsIndirectTarget: Yes
# LLVM-NEXT:     TypeId: 0xF85C699BB8EF20A2
# LLVM-NEXT:     NumDirectCallees: 0
# LLVM-NEXT:     DirectCallees [
# LLVM-NEXT:     ]
# LLVM-NEXT:     NumIndirectTargetTypeIDs: 0
# LLVM-NEXT:     IndirectTypeIDs: []
# LLVM-NEXT:   }
# LLVM-NEXT:   Function {
# LLVM-NEXT:     Name: _Z3barv
# LLVM-NEXT:     Version: 0
# LLVM-NEXT:     IsIndirectTarget: Yes
# LLVM-NEXT:     TypeId: 0xF85C699BB8EF20A2
# LLVM-NEXT:     NumDirectCallees: 0
# LLVM-NEXT:     DirectCallees [
# LLVM-NEXT:     ]
# LLVM-NEXT:     NumIndirectTargetTypeIDs: 0
# LLVM-NEXT:     IndirectTypeIDs: []
# LLVM-NEXT:   }
# LLVM-NEXT:   Function {
# LLVM-NEXT:     Name: _Z3bazc
# LLVM-NEXT:     Version: 0
# LLVM-NEXT:     IsIndirectTarget: Yes
# LLVM-NEXT:     TypeId: 0x308E4B8159BC8654
# LLVM-NEXT:     NumDirectCallees: 0
# LLVM-NEXT:     DirectCallees [
# LLVM-NEXT:     ]
# LLVM-NEXT:     NumIndirectTargetTypeIDs: 0
# LLVM-NEXT:     IndirectTypeIDs: []
# LLVM-NEXT:   }
# LLVM-NEXT:   Function {
# LLVM-NEXT:     Name: main
# LLVM-NEXT:     Version: 0
# LLVM-NEXT:     IsIndirectTarget: Yes
# LLVM-NEXT:     TypeId: 0xA9494DEF81A01DC
# LLVM-NEXT:     NumDirectCallees: 3
# LLVM-NEXT:     DirectCallees [
# LLVM-NEXT:       {
# LLVM-NEXT:         Name: _Z3foov
# LLVM-NEXT:       }
# LLVM-NEXT:       {
# LLVM-NEXT:         Name: _Z3barv
# LLVM-NEXT:       }
# LLVM-NEXT:       {
# LLVM-NEXT:         Name: _Z3bazc
# LLVM-NEXT:       }
# LLVM-NEXT:     ]
# LLVM-NEXT:     NumIndirectTargetTypeIDs: 2
# LLVM-NEXT:     IndirectTypeIDs: [0xF85C699BB8EF20A2, 0x308E4B8159BC8654]
# LLVM-NEXT:   }
# LLVM-NEXT: ]

# JSON:     "CallGraph": [
# JSON-NEXT:      {
# JSON-NEXT:        "Function": {
# JSON-NEXT:          "Name": "_Z3foov",
# JSON-NEXT:          "Version": 0,
# JSON-NEXT:          "IsIndirectTarget": true,
# JSON-NEXT:          "TypeId": 17896295136807035042,
# JSON-NEXT:          "NumDirectCallees": 0,
# JSON-NEXT:          "DirectCallees": [],
# JSON-NEXT:          "NumIndirectTargetTypeIDs": 0,
# JSON-NEXT:          "IndirectTypeIDs": []
# JSON-NEXT:        }
# JSON-NEXT:      },
# JSON-NEXT:      {
# JSON-NEXT:        "Function": {
# JSON-NEXT:          "Name": "_Z3barv",
# JSON-NEXT:          "Version": 0,
# JSON-NEXT:          "IsIndirectTarget": true,
# JSON-NEXT:          "TypeId": 17896295136807035042,
# JSON-NEXT:          "NumDirectCallees": 0,
# JSON-NEXT:          "DirectCallees": [],
# JSON-NEXT:          "NumIndirectTargetTypeIDs": 0,
# JSON-NEXT:          "IndirectTypeIDs": []
# JSON-NEXT:        }
# JSON-NEXT:      },
# JSON-NEXT:      {
# JSON-NEXT:        "Function": {
# JSON-NEXT:          "Name": "_Z3bazc",
# JSON-NEXT:          "Version": 0,
# JSON-NEXT:          "IsIndirectTarget": true,
# JSON-NEXT:          "TypeId": 3498816979441845844,
# JSON-NEXT:          "NumDirectCallees": 0,
# JSON-NEXT:          "DirectCallees": [],
# JSON-NEXT:          "NumIndirectTargetTypeIDs": 0,
# JSON-NEXT:          "IndirectTypeIDs": []
# JSON-NEXT:        }
# JSON-NEXT:      },
# JSON-NEXT:      {
# JSON-NEXT:        "Function": {
# JSON-NEXT:          "Name": "main",
# JSON-NEXT:          "Version": 0,
# JSON-NEXT:          "IsIndirectTarget": true,
# JSON-NEXT:          "TypeId": 762397922298560988,
# JSON-NEXT:          "NumDirectCallees": 3,
# JSON-NEXT:          "DirectCallees": [
# JSON-NEXT:            {
# JSON-NEXT:              "Name": "_Z3foov"
# JSON-NEXT:            },
# JSON-NEXT:            {
# JSON-NEXT:              "Name": "_Z3barv"
# JSON-NEXT:            },
# JSON-NEXT:            {
# JSON-NEXT:              "Name": "_Z3bazc"
# JSON-NEXT:            }
# JSON-NEXT:          ],
# JSON-NEXT:          "NumIndirectTargetTypeIDs": 2,
# JSON-NEXT:          "IndirectTypeIDs": [
# JSON-NEXT:            17896295136807035042,
# JSON-NEXT:            3498816979441845844
# JSON-NEXT:          ]
# JSON-NEXT:        }
# JSON-NEXT:      }
# JSON-NEXT:    ]

	.file	"cg.cpp"
	.text
	.globl	_Z3foov                         # -- Begin function _Z3foov
	.p2align	4
	.type	_Z3foov,@function
_Z3foov:                                # @_Z3foov
.Lfunc_begin0:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	_Z3foov, .Lfunc_end0-_Z3foov
	.cfi_endproc
	.section	.llvm.callgraph,"o",@llvm_call_graph,.text
	.byte	0
	.byte	1
	.quad	_Z3foov
	.quad	-550448936902516574
	.text
                                        # -- End function
	.globl	_Z3barv                         # -- Begin function _Z3barv
	.p2align	4
	.type	_Z3barv,@function
_Z3barv:                                # @_Z3barv
.Lfunc_begin1:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end1:
	.size	_Z3barv, .Lfunc_end1-_Z3barv
	.cfi_endproc
	.section	.llvm.callgraph,"o",@llvm_call_graph,.text
	.byte	0
	.byte	1
	.quad	_Z3barv
	.quad	-550448936902516574
	.text
                                        # -- End function
	.globl	_Z3bazc                         # -- Begin function _Z3bazc
	.p2align	4
	.type	_Z3bazc,@function
_Z3bazc:                                # @_Z3bazc
.Lfunc_begin2:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movb	%dil, %al
	movb	%al, -1(%rbp)
	xorl	%eax, %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end2:
	.size	_Z3bazc, .Lfunc_end2-_Z3bazc
	.cfi_endproc
	.section	.llvm.callgraph,"o",@llvm_call_graph,.text
	.byte	0
	.byte	1
	.quad	_Z3bazc
	.quad	3498816979441845844
	.text
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4
	.type	main,@function
main:                                   # @main
.Lfunc_begin3:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movl	$0, -4(%rbp)
	leaq	_Z3foov(%rip), %rax
	movq	%rax, -16(%rbp)
	callq	*-16(%rbp)
	leaq	_Z3barv(%rip), %rax
	movq	%rax, -24(%rbp)
	callq	*-24(%rbp)
	leaq	_Z3bazc(%rip), %rax
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movsbl	-25(%rbp), %edi
	callq	*%rax
	callq	_Z3foov
	callq	_Z3barv
	movsbl	-25(%rbp), %edi
	callq	_Z3bazc
	xorl	%eax, %eax
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end3:
	.size	main, .Lfunc_end3-main
	.cfi_endproc
	.section	.llvm.callgraph,"o",@llvm_call_graph,.text
	.byte	0
	.byte	7
	.quad	main
	.quad	762397922298560988
	.byte	3
	.quad	_Z3foov
	.quad	_Z3barv
	.quad	_Z3bazc
	.byte	2
	.quad	-550448936902516574
	.quad	3498816979441845844
	.text
                                        # -- End function
	.ident	"Fuchsia clang version 22.0.0git (git@github.com:Prabhuk/llvm-project.git aaa2cd4fc523537be583bc5a2b2f4d447575a6b4)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z3foov
	.addrsig_sym _Z3barv
	.addrsig_sym _Z3bazc
