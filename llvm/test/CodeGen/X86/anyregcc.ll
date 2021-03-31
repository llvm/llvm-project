; RUN: llc < %s -mtriple=x86_64-apple-darwin                  -frame-pointer=all | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7     -frame-pointer=all | FileCheck --check-prefix=SSE %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -frame-pointer=all | FileCheck --check-prefix=AVX %s


; Stackmap Header: no constants - 6 callsites
; CHECK-LABEL:  .section __LLVM_STACKMAPS,__llvm_stackmaps
; CHECK-NEXT:   __LLVM_StackMaps:
; Header
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 0
; Num Functions
; CHECK-NEXT:   .long 8
; Num Constants
; CHECK-NEXT:   .long 0
; Num Callsites
; CHECK-NEXT:   .long 8

; Functions and stack size
; CHECK-NEXT:   .quad _test
; CHECK-NEXT:   .quad 8
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _property_access1
; CHECK-NEXT:   .quad 8
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _property_access2
; CHECK-NEXT:   .quad 24
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _property_access3
; CHECK-NEXT:   .quad 24
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _anyreg_test1
; CHECK-NEXT:   .quad 56
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _anyreg_test2
; CHECK-NEXT:   .quad 8
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _patchpoint_spilldef
; CHECK-NEXT:   .quad 56
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad _patchpoint_spillargs
; CHECK-NEXT:   .quad 88
; CHECK-NEXT:   .quad 1

; No constants

; Callsites
; test
; CHECK-LABEL:  .long   L{{.*}}-_test
; CHECK-NEXT:   .short  0
; 3 locations
; CHECK-NEXT:   .short  3
; Loc 0: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 4
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 4
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 2: Constant 3
; CHECK-NEXT:   .byte 4
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 3
define i64 @test() nounwind ssp uwtable {
entry:
  call anyregcc void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 0, i32 15, i8* null, i32 2, i32 1, i32 2, i64 3)
  ret i64 0
}

; property access 1 - %obj is an anyreg call argument and should therefore be in a register
; CHECK-LABEL:  .long   L{{.*}}-_property_access1
; CHECK-NEXT:   .short  0
; 2 locations
; CHECK-NEXT:   .short  2
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define i64 @property_access1(i8* %obj) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 12297829382473034410 to i8*
  %ret = call anyregcc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 1, i32 15, i8* %f, i32 1, i8* %obj)
  ret i64 %ret
}

; property access 2 - %obj is an anyreg call argument and should therefore be in a register
; CHECK-LABEL:  .long   L{{.*}}-_property_access2
; CHECK-NEXT:   .short  0
; 2 locations
; CHECK-NEXT:   .short  2
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define i64 @property_access2() nounwind ssp uwtable {
entry:
  %obj = alloca i64, align 8
  %f = inttoptr i64 12297829382473034410 to i8*
  %ret = call anyregcc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 2, i32 15, i8* %f, i32 1, i64* %obj)
  ret i64 %ret
}

; property access 3 - %obj is a frame index
; CHECK-LABEL:  .long   L{{.*}}-_property_access3
; CHECK-NEXT:   .short  0
; 2 locations
; CHECK-NEXT:   .short  2
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Direct RBP - ofs
; CHECK-NEXT:   .byte 2
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short 6
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long
define i64 @property_access3() nounwind ssp uwtable {
entry:
  %obj = alloca i64, align 8
  %f = inttoptr i64 12297829382473034410 to i8*
  %ret = call anyregcc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 3, i32 15, i8* %f, i32 0, i64* %obj)
  ret i64 %ret
}

; anyreg_test1
; CHECK-LABEL:  .long   L{{.*}}-_anyreg_test1
; CHECK-NEXT:   .short  0
; 14 locations
; CHECK-NEXT:   .short  14
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 2: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 3: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 4: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 5: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 6: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 7: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 8: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 9: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 10: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 11: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 12: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 13: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
define i64 @anyreg_test1(i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9, i8* %a10, i8* %a11, i8* %a12, i8* %a13) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 12297829382473034410 to i8*
  %ret = call anyregcc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 4, i32 15, i8* %f, i32 13, i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9, i8* %a10, i8* %a11, i8* %a12, i8* %a13)
  ret i64 %ret
}

; anyreg_test2
; CHECK-LABEL:  .long   L{{.*}}-_anyreg_test2
; CHECK-NEXT:   .short  0
; 14 locations
; CHECK-NEXT:   .short  14
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 2: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 3: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 4: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 5: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 6: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 7: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 8: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 8
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .long 0
; Loc 9: Argument, still on stack
; CHECK-NEXT: .byte  3
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short 6
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long
; Loc 10: Argument, still on stack
; CHECK-NEXT: .byte  3
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short 6
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long
; Loc 11: Argument, still on stack
; CHECK-NEXT: .byte  3
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short 6
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long
; Loc 12: Argument, still on stack
; CHECK-NEXT: .byte  3
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short 6
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long
; Loc 13: Argument, still on stack
; CHECK-NEXT: .byte  3
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short 6
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long
define i64 @anyreg_test2(i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9, i8* %a10, i8* %a11, i8* %a12, i8* %a13) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 12297829382473034410 to i8*
  %ret = call anyregcc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 5, i32 15, i8* %f, i32 8, i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9, i8* %a10, i8* %a11, i8* %a12, i8* %a13)
  ret i64 %ret
}

; Test spilling the return value of an anyregcc call.
;
; <rdar://problem/15432754> [JS] Assertion: "Folded a def to a non-store!"
;
; CHECK-LABEL: .long L{{.*}}-_patchpoint_spilldef
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 3
; Loc 0: Register (some register that will be spilled to the stack)
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 1: Register RDI
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short 5
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 1: Register RSI
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short 4
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
define i64 @patchpoint_spilldef(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
  %result = tail call anyregcc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 12, i32 15, i8* inttoptr (i64 0 to i8*), i32 2, i64 %p1, i64 %p2)
  tail call void asm sideeffect "nop", "~{ax},~{bx},~{cx},~{dx},~{bp},~{si},~{di},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"() nounwind
  ret i64 %result
}

; Test spilling the arguments of an anyregcc call.
;
; <rdar://problem/15487687> [JS] AnyRegCC argument ends up being spilled
;
; CHECK-LABEL: .long L{{.*}}-_patchpoint_spillargs
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 5
; Loc 0: Return a register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 1: Arg0 in a Register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 2: Arg1 in a Register
; CHECK-NEXT: .byte  1
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short {{[0-9]+}}
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long  0
; Loc 3: Arg2 spilled to RBP +
; CHECK-NEXT: .byte  3
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short 6
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long
; Loc 4: Arg3 spilled to RBP +
; CHECK-NEXT: .byte  3
; CHECK-NEXT: .byte  0
; CHECK-NEXT: .short  8
; CHECK-NEXT: .short 6
; CHECK-NEXT: .short 0
; CHECK-NEXT: .long
define i64 @patchpoint_spillargs(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
  tail call void asm sideeffect "nop", "~{ax},~{bx},~{cx},~{dx},~{bp},~{si},~{di},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"() nounwind
  %result = tail call anyregcc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 13, i32 15, i8* inttoptr (i64 0 to i8*), i32 2, i64 %p1, i64 %p2, i64 %p3, i64 %p4)
  ret i64 %result
}

; Make sure all regs are spilled
define anyregcc void @anyregcc1() {
entry:
;SSE-LABEL: anyregcc1
;SSE:      pushq %rbp
;SSE:      pushq %rax
;SSE:      pushq %r15
;SSE:      pushq %r14
;SSE:      pushq %r13
;SSE:      pushq %r12
;SSE:      pushq %r11
;SSE:      pushq %r10
;SSE:      pushq %r9
;SSE:      pushq %r8
;SSE:      pushq %rdi
;SSE:      pushq %rsi
;SSE:      pushq %rdx
;SSE:      pushq %rcx
;SSE:      pushq %rbx
;SSE:      movaps %xmm15
;SSE-NEXT: movaps %xmm14
;SSE-NEXT: movaps %xmm13
;SSE-NEXT: movaps %xmm12
;SSE-NEXT: movaps %xmm11
;SSE-NEXT: movaps %xmm10
;SSE-NEXT: movaps %xmm9
;SSE-NEXT: movaps %xmm8
;SSE-NEXT: movaps %xmm7
;SSE-NEXT: movaps %xmm6
;SSE-NEXT: movaps %xmm5
;SSE-NEXT: movaps %xmm4
;SSE-NEXT: movaps %xmm3
;SSE-NEXT: movaps %xmm2
;SSE-NEXT: movaps %xmm1
;SSE-NEXT: movaps %xmm0
;AVX-LABEL:anyregcc1
;AVX:      pushq %rbp
;AVX:      pushq %rax
;AVX:      pushq %r15
;AVX:      pushq %r14
;AVX:      pushq %r13
;AVX:      pushq %r12
;AVX:      pushq %r11
;AVX:      pushq %r10
;AVX:      pushq %r9
;AVX:      pushq %r8
;AVX:      pushq %rdi
;AVX:      pushq %rsi
;AVX:      pushq %rdx
;AVX:      pushq %rcx
;AVX:      pushq %rbx
;AVX:      vmovups %ymm15
;AVX-NEXT: vmovups %ymm14
;AVX-NEXT: vmovups %ymm13
;AVX-NEXT: vmovups %ymm12
;AVX-NEXT: vmovups %ymm11
;AVX-NEXT: vmovups %ymm10
;AVX-NEXT: vmovups %ymm9
;AVX-NEXT: vmovups %ymm8
;AVX-NEXT: vmovups %ymm7
;AVX-NEXT: vmovups %ymm6
;AVX-NEXT: vmovups %ymm5
;AVX-NEXT: vmovups %ymm4
;AVX-NEXT: vmovups %ymm3
;AVX-NEXT: vmovups %ymm2
;AVX-NEXT: vmovups %ymm1
;AVX-NEXT: vmovups %ymm0
  call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{rbp},~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  ret void
}

; Make sure we don't spill any XMMs/YMMs
declare anyregcc void @foo()
define void @anyregcc2() {
entry:
;SSE-LABEL: anyregcc2
;SSE-NOT: movaps %xmm
;AVX-LABEL: anyregcc2
;AVX-NOT: vmovups %ymm
  %a0 = call <2 x double> asm sideeffect "", "={xmm0}"() nounwind
  %a1 = call <2 x double> asm sideeffect "", "={xmm1}"() nounwind
  %a2 = call <2 x double> asm sideeffect "", "={xmm2}"() nounwind
  %a3 = call <2 x double> asm sideeffect "", "={xmm3}"() nounwind
  %a4 = call <2 x double> asm sideeffect "", "={xmm4}"() nounwind
  %a5 = call <2 x double> asm sideeffect "", "={xmm5}"() nounwind
  %a6 = call <2 x double> asm sideeffect "", "={xmm6}"() nounwind
  %a7 = call <2 x double> asm sideeffect "", "={xmm7}"() nounwind
  %a8 = call <2 x double> asm sideeffect "", "={xmm8}"() nounwind
  %a9 = call <2 x double> asm sideeffect "", "={xmm9}"() nounwind
  %a10 = call <2 x double> asm sideeffect "", "={xmm10}"() nounwind
  %a11 = call <2 x double> asm sideeffect "", "={xmm11}"() nounwind
  %a12 = call <2 x double> asm sideeffect "", "={xmm12}"() nounwind
  %a13 = call <2 x double> asm sideeffect "", "={xmm13}"() nounwind
  %a14 = call <2 x double> asm sideeffect "", "={xmm14}"() nounwind
  %a15 = call <2 x double> asm sideeffect "", "={xmm15}"() nounwind
  call anyregcc void @foo()
  call void asm sideeffect "", "{xmm0},{xmm1},{xmm2},{xmm3},{xmm4},{xmm5},{xmm6},{xmm7},{xmm8},{xmm9},{xmm10},{xmm11},{xmm12},{xmm13},{xmm14},{xmm15}"(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, <2 x double> %a3, <2 x double> %a4, <2 x double> %a5, <2 x double> %a6, <2 x double> %a7, <2 x double> %a8, <2 x double> %a9, <2 x double> %a10, <2 x double> %a11, <2 x double> %a12, <2 x double> %a13, <2 x double> %a14, <2 x double> %a15)
  ret void
}

declare void @llvm.experimental.patchpoint.void(i64, i32, i8*, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, i8*, i32, ...)
