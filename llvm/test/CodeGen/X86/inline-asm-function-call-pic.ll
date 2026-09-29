; RUN: llc -O2 --relocation-model=pic -mtriple=i386-unknown-linux-gnu < %s 2>&1 | FileCheck %s

; List the source code:
; // clang  -m32 -fasm-blocks -S t.c -O2  -fpic -emit-llvm
; int GV = 17;
;
; extern unsigned int extern_func();
; static unsigned int static_func()  __attribute__((noinline));
; static unsigned int static_func() {
;   return GV++;
; }
;
; void func() {
;   static_func();
;   __asm {
;           call static_func
;           call extern_func
;           jmp extern_func
;           shr eax, 0
;           shr ebx, 0
;           shr ecx, 0
;           shr edx, 0
;           shr edi, 0
;           shr esi, 0
;           shr ebp, 0
;           shr esp, 0
;         }
; }

@GV = local_unnamed_addr global i32 17, align 4

define void @func() local_unnamed_addr #0 {
; CHECK-LABEL: func:
; CHECK:         calll .L0$pb
; CHECK-NEXT:  .L0$pb:
; CHECK-NEXT:    popl %ebx
; CHECK-NEXT:  .Ltmp0:
; CHECK-NEXT:    addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp0-.L0$pb), %ebx
; CHECK-NEXT:    calll static_func
; CHECK-NEXT:    pushl %ebp
; CHECK-NEXT:    subl $12, %esp
; CHECK-NEXT:    #APP
; CHECK-EMPTY:
; CHECK-NEXT:    calll static_func
; CHECK-NEXT:    calll extern_func@PLT
; CHECK-NEXT:    jmp extern_func@PLT
; CHECK-NEXT:    shrl $0, %eax
; CHECK-NEXT:    shrl $0, %ebx
; CHECK-NEXT:    shrl $0, %ecx
; CHECK-NEXT:    shrl $0, %edx
; CHECK-NEXT:    shrl $0, %edi
; CHECK-NEXT:    shrl $0, %esi
; CHECK-NEXT:    shrl $0, %ebp
; CHECK-NEXT:    shrl $0, %esp
; CHECK-EMPTY:
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    addl $12, %esp
; CHECK-NEXT:    popl %ebp
entry:
  %call = tail call i32 @static_func()
;; We test call, CALL, and jmp.
  tail call void asm sideeffect inteldialect "call ${0:P}\0A\09CALL ${1:P}\0A\09jmp ${1:P}\0A\09shr eax, $$0\0A\09shr ebx, $$0\0A\09shr ecx, $$0\0A\09shr edx, $$0\0A\09shr edi, $$0\0A\09shr esi, $$0\0A\09shr ebp, $$0\0A\09shr esp, $$0", "*m,*m,~{eax},~{ebp},~{ebx},~{ecx},~{edi},~{edx},~{flags},~{esi},~{esp},~{dirflag},~{fpsr},~{flags}"(ptr nonnull elementtype(i32 (...)) @static_func, ptr nonnull elementtype(i32 (...)) @extern_func) #0
  ret void
}

declare i32 @extern_func(...) #0

define internal i32 @static_func() #0 {
entry:
  %0 = load i32, ptr @GV, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr @GV, align 4
  ret i32 %0
}

attributes #0 = { nounwind }
