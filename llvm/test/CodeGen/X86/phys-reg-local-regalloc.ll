; RUN: llc < %s -stack-symbol-ordering=0 -mtriple=i386-apple-darwin9 -mcpu=generic -regalloc=fast -optimize-regalloc=0 -no-x86-call-frame-opt | FileCheck %s
; RUN: llc -O0 < %s -stack-symbol-ordering=0 -mtriple=i386-apple-darwin9 -mcpu=generic -regalloc=fast -no-x86-call-frame-opt | FileCheck %s
; RUN: llc < %s -stack-symbol-ordering=0 -mtriple=i386-apple-darwin9 -mcpu=atom -regalloc=fast -optimize-regalloc=0 -no-x86-call-frame-opt | FileCheck -check-prefix=ATOM %s
; CHECKed instructions should be the same with or without -O0 except on Intel Atom due to instruction scheduling.

@.str = private constant [12 x i8] c"x + y = %i\0A\00", align 1 ; <[12 x i8]*> [#uses=1]

define i32 @main() nounwind {
entry:
; CHECK: movl 24(%esp), %eax
; CHECK-NOT: movl
; CHECK: movl	%eax, 36(%esp)
; CHECK-NOT: movl
; CHECK: movl 28(%esp), %ebx
; CHECK-NOT: movl
; CHECK: movl	%ebx, 40(%esp)
; CHECK-NOT: movl
; CHECK: addl %ebx, %eax

; On Intel Atom the scheduler moves a movl instruction
; used for the printf call to follow movl 24(%esp), %eax
; ATOM: movl 24(%esp), %eax
; ATOM: movl
; ATOM: movl   %eax, 36(%esp)
; ATOM-NOT: movl
; ATOM: movl 28(%esp), %ebx
; ATOM-NOT: movl
; ATOM: movl   %ebx, 40(%esp)
; ATOM-NOT: movl
; ATOM: addl %ebx, %eax

  %retval = alloca i32                            ; <i32*> [#uses=2]
  %"%ebx" = alloca i32                            ; <i32*> [#uses=1]
  %"%eax" = alloca i32                            ; <i32*> [#uses=2]
  %result = alloca i32                            ; <i32*> [#uses=2]
  %y = alloca i32                                 ; <i32*> [#uses=2]
  %x = alloca i32                                 ; <i32*> [#uses=2]
  %0 = alloca i32                                 ; <i32*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store i32 1, i32* %x, align 4
  store i32 2, i32* %y, align 4
  call void asm sideeffect alignstack "# top of block", "~{dirflag},~{fpsr},~{flags},~{edi},~{esi},~{edx},~{ecx},~{eax}"() nounwind
  %asmtmp = call i32 asm sideeffect alignstack "movl $1, $0", "=={eax},*m,~{dirflag},~{fpsr},~{flags},~{memory}"(i32* %x) nounwind ; <i32> [#uses=1]
  store i32 %asmtmp, i32* %"%eax"
  %asmtmp1 = call i32 asm sideeffect alignstack "movl $1, $0", "=={ebx},*m,~{dirflag},~{fpsr},~{flags},~{memory}"(i32* %y) nounwind ; <i32> [#uses=1]
  store i32 %asmtmp1, i32* %"%ebx"
  %1 = call i32 asm "", "={bx}"() nounwind        ; <i32> [#uses=1]
  %2 = call i32 asm "", "={ax}"() nounwind        ; <i32> [#uses=1]
  %asmtmp2 = call i32 asm sideeffect alignstack "addl $1, $0", "=={eax},{ebx},{eax},~{dirflag},~{fpsr},~{flags},~{memory}"(i32 %1, i32 %2) nounwind ; <i32> [#uses=1]
  store i32 %asmtmp2, i32* %"%eax"
  %3 = call i32 asm "", "={ax}"() nounwind        ; <i32> [#uses=1]
  call void asm sideeffect alignstack "movl $0, $1", "{eax},*m,~{dirflag},~{fpsr},~{flags},~{memory}"(i32 %3, i32* %result) nounwind
  %4 = load i32, i32* %result, align 4                 ; <i32> [#uses=1]
  %5 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str, i32 0, i32 0), i32 %4) nounwind ; <i32> [#uses=0]
  store i32 0, i32* %0, align 4
  %6 = load i32, i32* %0, align 4                      ; <i32> [#uses=1]
  store i32 %6, i32* %retval, align 4
  br label %return

return:                                           ; preds = %entry
  %retval3 = load i32, i32* %retval                    ; <i32> [#uses=1]
  ret i32 %retval3
}

declare i32 @printf(i8*, ...) nounwind
