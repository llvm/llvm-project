; RUN: llc < %s -stack-symbol-ordering=0 -mcpu=generic -mattr=+avx -mtriple=x86_64-apple-darwin10 | FileCheck %s
; RUN: llc < %s -stack-symbol-ordering=0 -mcpu=generic -stackrealign -stack-alignment=32 -mattr=+avx -mtriple=x86_64-apple-darwin10 | FileCheck %s -check-prefix=FORCE-ALIGN
; rdar://11496434

; no VLAs or dynamic alignment
define i32 @t1() nounwind uwtable ssp {
entry:
  %a = alloca i32, align 4
  call void @t1_helper(i32* %a) nounwind
  %0 = load i32, i32* %a, align 4
  %add = add nsw i32 %0, 13
  ret i32 %add

; CHECK: _t1
; CHECK-NOT: andq $-{{[0-9]+}}, %rsp
; CHECK: leaq [[OFFSET:[0-9]*]](%rsp), %rdi
; CHECK: callq _t1_helper
; CHECK: movl [[OFFSET]](%rsp), %eax
; CHECK: addl $13, %eax
}

declare void @t1_helper(i32*)

; dynamic realignment
define i32 @t2() nounwind uwtable ssp {
entry:
  %a = alloca i32, align 4
  %v = alloca <8 x float>, align 32
  call void @t2_helper(i32* %a, <8 x float>* %v) nounwind
  %0 = load i32, i32* %a, align 4
  %add = add nsw i32 %0, 13
  ret i32 %add

; CHECK: _t2
; CHECK: pushq %rbp
; CHECK: movq %rsp, %rbp
; CHECK: andq $-32, %rsp
; CHECK: subq ${{[0-9]+}}, %rsp
;
; CHECK: leaq {{[0-9]*}}(%rsp), %rdi
; CHECK: movq %rsp, %rsi
; CHECK: callq _t2_helper
;
; CHECK: movq %rbp, %rsp
; CHECK: popq %rbp
}

declare void @t2_helper(i32*, <8 x float>*)

; VLAs
define i32 @t3(i64 %sz) nounwind uwtable ssp {
entry:
  %a = alloca i32, align 4
  %vla = alloca i32, i64 %sz, align 16
  call void @t3_helper(i32* %a, i32* %vla) nounwind
  %0 = load i32, i32* %a, align 4
  %add = add nsw i32 %0, 13
  ret i32 %add

; CHECK: _t3
; CHECK: pushq %rbp
; CHECK: movq %rsp, %rbp
; CHECK-NOT: andq $-{{[0-9]+}}, %rsp
; CHECK: subq ${{[0-9]+}}, %rsp
;
; CHECK: movq %rbp, %rsp
; CHECK: popq %rbp
}

declare void @t3_helper(i32*, i32*)

; VLAs + Dynamic realignment
define i32 @t4(i64 %sz) nounwind uwtable ssp {
entry:
  %a = alloca i32, align 4
  %v = alloca <8 x float>, align 32
  %vla = alloca i32, i64 %sz, align 16
  call void @t4_helper(i32* %a, i32* %vla, <8 x float>* %v) nounwind
  %0 = load i32, i32* %a, align 4
  %add = add nsw i32 %0, 13
  ret i32 %add

; CHECK: _t4
; CHECK: pushq %rbp
; CHECK: movq %rsp, %rbp
; CHECK: pushq %rbx
; CHECK: andq $-32, %rsp
; CHECK: subq ${{[0-9]+}}, %rsp
; CHECK: movq %rsp, %rbx
;
; CHECK: leaq {{[0-9]*}}(%rbx), %rdi
; CHECK: movq %rbx, %rdx
; CHECK: callq   _t4_helper
;
; CHECK: leaq -{{[0-9]+}}(%rbp), %rsp
; CHECK: popq %rbx
; CHECK: popq %rbp
}

declare void @t4_helper(i32*, i32*, <8 x float>*)

; Spilling an AVX register shouldn't cause dynamic realignment
define i32 @t5(float* nocapture %f) nounwind uwtable ssp {
entry:
  %a = alloca i32, align 4
  %0 = bitcast float* %f to <8 x float>*
  %1 = load <8 x float>, <8 x float>* %0, align 32
  call void @t5_helper1(i32* %a) nounwind
  call void @t5_helper2(<8 x float> %1) nounwind
  %2 = load i32, i32* %a, align 4
  %add = add nsw i32 %2, 13
  ret i32 %add

; CHECK: _t5
; CHECK: subq ${{[0-9]+}}, %rsp
;
; CHECK: vmovaps (%rdi), [[AVXREG:%ymm[0-9]+]]
; CHECK: vmovups [[AVXREG]], (%rsp)
; CHECK: leaq {{[0-9]+}}(%rsp), %rdi
; CHECK: callq   _t5_helper1
; CHECK: vmovups (%rsp), %ymm0
; CHECK: callq   _t5_helper2
; CHECK: movl {{[0-9]+}}(%rsp), %eax
}

declare void @t5_helper1(i32*)

declare void @t5_helper2(<8 x float>)

; VLAs + Dynamic realignment + Spill
; FIXME: RA has already reserved RBX, so we can't do dynamic realignment.
define i32 @t6(i64 %sz, float* nocapture %f) nounwind uwtable ssp {
entry:
; CHECK: _t6
  %a = alloca i32, align 4
  %0 = bitcast float* %f to <8 x float>*
  %1 = load <8 x float>, <8 x float>* %0, align 32
  %vla = alloca i32, i64 %sz, align 16
  call void @t6_helper1(i32* %a, i32* %vla) nounwind
  call void @t6_helper2(<8 x float> %1) nounwind
  %2 = load i32, i32* %a, align 4
  %add = add nsw i32 %2, 13
  ret i32 %add
}

declare void @t6_helper1(i32*, i32*)

declare void @t6_helper2(<8 x float>)

; VLAs + Dynamic realignment + byval
; The byval adjust the sp after the prolog, but if we're restoring the sp from
; the base pointer we use the original adjustment.
%struct.struct_t = type { [5 x i32] }

define void @t7(i32 %size, %struct.struct_t* byval(%struct.struct_t) align 8 %arg1) nounwind uwtable {
entry:
  %x = alloca i32, align 32
  store i32 0, i32* %x, align 32
  %0 = zext i32 %size to i64
  %vla = alloca i32, i64 %0, align 16
  %1 = load i32, i32* %x, align 32
  call void @bar(i32 %1, i32* %vla, %struct.struct_t* byval(%struct.struct_t) align 8 %arg1)
  ret void

; CHECK: _t7
; CHECK:     pushq %rbp
; CHECK:     movq %rsp, %rbp
; CHECK:     pushq %rbx
; CHECK:     andq $-32, %rsp
; CHECK:     subq ${{[0-9]+}}, %rsp
; CHECK:     movq %rsp, %rbx

; Stack adjustment for byval
; CHECK:     subq {{.*}}, %rsp
; CHECK:     callq _bar
; CHECK-NOT: addq {{.*}}, %rsp
; CHECK:     leaq -8(%rbp), %rsp
; CHECK:     popq %rbx
; CHECK:     popq %rbp
}

declare i8* @llvm.stacksave() nounwind

declare void @bar(i32, i32*, %struct.struct_t* byval(%struct.struct_t) align 8)

declare void @llvm.stackrestore(i8*) nounwind


; Test when forcing stack alignment
define i32 @t8() nounwind uwtable {
entry:
  %a = alloca i32, align 4
  call void @t1_helper(i32* %a) nounwind
  %0 = load i32, i32* %a, align 4
  %add = add nsw i32 %0, 13
  ret i32 %add

; FORCE-ALIGN: _t8
; FORCE-ALIGN:      movq %rsp, %rbp
; FORCE-ALIGN:      andq $-32, %rsp
; FORCE-ALIGN-NEXT: subq $32, %rsp
; FORCE-ALIGN:      movq %rbp, %rsp
; FORCE-ALIGN:      popq %rbp
}

; VLAs
define i32 @t9(i64 %sz) nounwind uwtable {
entry:
  %a = alloca i32, align 4
  %vla = alloca i32, i64 %sz, align 16
  call void @t3_helper(i32* %a, i32* %vla) nounwind
  %0 = load i32, i32* %a, align 4
  %add = add nsw i32 %0, 13
  ret i32 %add

; FORCE-ALIGN: _t9
; FORCE-ALIGN: pushq %rbp
; FORCE-ALIGN: movq %rsp, %rbp
; FORCE-ALIGN: pushq %rbx
; FORCE-ALIGN: andq $-32, %rsp
; FORCE-ALIGN: subq $32, %rsp
; FORCE-ALIGN: movq %rsp, %rbx

; FORCE-ALIGN: leaq -8(%rbp), %rsp
; FORCE-ALIGN: popq %rbx
; FORCE-ALIGN: popq %rbp
}
