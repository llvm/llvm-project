; RUN: llc < %s -stack-symbol-ordering=0 -mtriple=i386-apple-darwin9 -mcpu=generic -regalloc=fast -optimize-regalloc=0 -no-x86-call-frame-opt | FileCheck %s
; RUN: llc -O0 < %s -stack-symbol-ordering=0 -mtriple=i386-apple-darwin9 -mcpu=generic -regalloc=fast -no-x86-call-frame-opt | FileCheck %s
; RUN: llc < %s -stack-symbol-ordering=0 -mtriple=i386-apple-darwin9 -mcpu=atom -regalloc=fast -optimize-regalloc=0 -no-x86-call-frame-opt | FileCheck %s

@.str = private constant [12 x i8] c"x + y = %i\0A\00", align 1 ; <ptr> [#uses=1]

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

  %retval = alloca i32                            ; <ptr> [#uses=2]
  %"%ebx" = alloca i32                            ; <ptr> [#uses=1]
  %"%eax" = alloca i32                            ; <ptr> [#uses=2]
  %result = alloca i32                            ; <ptr> [#uses=2]
  %y = alloca i32                                 ; <ptr> [#uses=2]
  %x = alloca i32                                 ; <ptr> [#uses=2]
  %0 = alloca i32                                 ; <ptr> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store i32 1, ptr %x, align 4
  store i32 2, ptr %y, align 4
  call void asm sideeffect alignstack "# top of block", "~{dirflag},~{fpsr},~{flags},~{edi},~{esi},~{edx},~{ecx},~{eax}"() nounwind
  %asmtmp = call i32 asm sideeffect alignstack "movl $1, $0", "=={eax},*m,~{dirflag},~{fpsr},~{flags},~{memory}"(ptr elementtype(i32) %x) nounwind ; <i32> [#uses=1]
  store i32 %asmtmp, ptr %"%eax"
  %asmtmp1 = call i32 asm sideeffect alignstack "movl $1, $0", "=={ebx},*m,~{dirflag},~{fpsr},~{flags},~{memory}"(ptr elementtype(i32) %y) nounwind ; <i32> [#uses=1]
  store i32 %asmtmp1, ptr %"%ebx"
  %1 = call i32 asm "", "={bx}"() nounwind        ; <i32> [#uses=1]
  %2 = call i32 asm "", "={ax}"() nounwind        ; <i32> [#uses=1]
  %asmtmp2 = call i32 asm sideeffect alignstack "addl $1, $0", "=={eax},{ebx},{eax},~{dirflag},~{fpsr},~{flags},~{memory}"(i32 %1, i32 %2) nounwind ; <i32> [#uses=1]
  store i32 %asmtmp2, ptr %"%eax"
  %3 = call i32 asm "", "={ax}"() nounwind        ; <i32> [#uses=1]
  call void asm sideeffect alignstack "movl $0, $1", "{eax},*m,~{dirflag},~{fpsr},~{flags},~{memory}"(i32 %3, ptr elementtype(i32) %result) nounwind
  %4 = load i32, ptr %result, align 4                 ; <i32> [#uses=1]
  %5 = call i32 (ptr, ...) @printf(ptr @.str, i32 %4) nounwind ; <i32> [#uses=0]
  store i32 0, ptr %0, align 4
  %6 = load i32, ptr %0, align 4                      ; <i32> [#uses=1]
  store i32 %6, ptr %retval, align 4
  br label %return

return:                                           ; preds = %entry
  %retval3 = load i32, ptr %retval                    ; <i32> [#uses=1]
  ret i32 %retval3
}

declare i32 @printf(ptr, ...) nounwind
