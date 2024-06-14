; RUN: split-file %s %t
; RUN: cat %t/main.ll %t/_align32.ll > %t/align32.ll
; RUN: llc -mtriple=x86_64-pc-linux-gnu -mattr=+mwaitx -x86-use-base-pointer=true -stackrealign %t/align32.ll -o - | FileCheck --check-prefix=CHECK --check-prefix=USE_BASE_64 %s
; RUN: llc -mtriple=x86_64-pc-linux-gnux32 -mattr=+mwaitx -x86-use-base-pointer=true -stackrealign %t/align32.ll -o - | FileCheck --check-prefix=CHECK --check-prefix=USE_BASE_32 %s
; RUN: llc -mtriple=x86_64-pc-linux-gnu -mattr=+mwaitx -x86-use-base-pointer=true %t/main.ll -o - | FileCheck --check-prefix=CHECK --check-prefix=NO_BASE_64 %s
; RUN: llc -mtriple=x86_64-pc-linux-gnux32 -mattr=+mwaitx -x86-use-base-pointer=true %t/main.ll -o - | FileCheck --check-prefix=CHECK --check-prefix=NO_BASE_32 %s

;--- main.ll

; This test checks that we save and restore the base pointer (ebx or rbx) in the
; presence of the mwaitx intrinsic which requires to use ebx for one of its
; argument.
; This function uses a dynamically allocated stack to force the use
; of a base pointer.
; After the call to the mwaitx intrinsic we do a volatile store to the
; dynamically allocated memory which will require the use of the base pointer.
; The base pointer should therefore be restored straight after the mwaitx
; instruction.

define void @test_baseptr(i64 %x, i64 %y, i32 %E, i32 %H, i32 %C) nounwind {
entry:
  %ptr = alloca ptr, align 8
  %0 = alloca i8, i64 %x, align 16
  store ptr %0, ptr %ptr, align 8
  call void @llvm.x86.mwaitx(i32 %E, i32 %H, i32 %C)
  %1 = load ptr, ptr %ptr, align 8
  %arrayidx = getelementptr inbounds i8, ptr %1, i64 %y
  store volatile i8 42, ptr %arrayidx, align 1
  ret void
}
; CHECK-LABEL: test_baseptr:
; USE_BASE_64: movq %rsp, %rbx
; Pass mwaitx first 2 arguments in eax and ecx respectively.
; USE_BASE_64: movl %ecx, %eax
; USE_BASE_64: movl %edx, %ecx
; Save base pointer.
; USE_BASE_64: movq %rbx, [[SAVE_rbx:%r([8-9]|1[0-5]|di|si)]]
; Set mwaitx ebx argument.
; USE_BASE_64: movl %r8d, %ebx
; USE_BASE_64-NEXT: mwaitx
; Restore base pointer.
; USE_BASE_64-NEXT: movq [[SAVE_rbx]], %rbx

; USE_BASE_32: movl %esp, %ebx
; Pass mwaitx first 2 arguments in eax and ecx respectively.
; USE_BASE_32: movl %ecx, %eax
; USE_BASE_32: movl %edx, %ecx
; Save base pointer.
; USE_BASE_32: movq %rbx, [[SAVE_rbx:%r(di|si)]]
; Set mwaitx ebx argument.
; USE_BASE_32: movl %r8d, %ebx
; USE_BASE_32-NEXT: mwaitx
; Restore base pointer.
; USE_BASE_32-NEXT: movq [[SAVE_rbx]], %rbx

; Pass mwaitx 3 arguments in eax, ecx, ebx
; NO_BASE_64: movl %r8d, %ebx
; NO_BASE_64: movl %ecx, %eax
; NO_BASE_64: movl %edx, %ecx
; No need to save base pointer.
; NO_BASE_64-NOT: movq %rbx
; NO_BASE_64: mwaitx
; No need to restore base pointer.
; NO_BASE_64-NOT: movq {{.*}}, %rbx
; NO_BASE_64-NEXT: {{.+$}}

; Pass mwaitx 3 arguments in eax, ecx, ebx
; NO_BASE_32: movl %r8d, %ebx
; NO_BASE_32: movl %ecx, %eax
; NO_BASE_32: movl %edx, %ecx
; No need to save base pointer.
; NO_BASE_32-NOT: movl %ebx
; NO_BASE_32: mwaitx
; No need to restore base pointer.
; NO_BASE_32-NOT: movl {{.*}}, %ebx
; NO_BASE_32-NEXT: {{.+$}}

; Test of the case where an opaque sp adjustement is introduced by a separate
; basic block which, combined with stack realignment, requires a base pointer.
@g = global i32 0, align 8

define void @test_opaque_sp_adjustment(i32 %E, i32 %H, i32 %C, i64 %x) {
entry:
  %ptr = alloca ptr, align 8
  call void @llvm.x86.mwaitx(i32 %E, i32 %H, i32 %C)
  %g = load i32, ptr @g, align 4
  %tobool = icmp ne i32 %g, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:
  call void asm sideeffect "", "~{rsp},~{esp},~{dirflag},~{fpsr},~{flags}"()
  br label %if.end

if.end:
  %ptr2 = load ptr, ptr %ptr, align 8
  %arrayidx = getelementptr inbounds i8, ptr %ptr2, i64 %x
  store volatile i8 42, ptr %arrayidx, align 1
  ret void
}
; CHECK-LABEL: test_opaque_sp_adjustment:
; USE_BASE_64: movq %rsp, %rbx
; Pass mwaitx first 2 arguments in eax and ecx respectively.
; USE_BASE_64: movl %esi, %eax
; USE_BASE_64: movl %edi, %ecx
; Save base pointer.
; USE_BASE_64: movq %rbx, [[SAVE_rbx:%r([8-9]|1[0-5]|di|si)]]
; Set mwaitx ebx argument.
; USE_BASE_64: movl %edx, %ebx
; USE_BASE_64-NEXT: mwaitx
; Restore base pointer.
; USE_BASE_64-NEXT: movq [[SAVE_rbx]], %rbx

; USE_BASE_32: movl %esp, %ebx
; Pass mwaitx first 2 arguments in eax and ecx respectively.
; USE_BASE_32: movl %esi, %eax
; USE_BASE_32: movl %edi, %ecx
; Save base pointer.
; USE_BASE_32: movq %rbx, [[SAVE_rbx:%r(di|si)]]
; Set mwaitx ebx argument.
; USE_BASE_32: movl %edx, %ebx
; USE_BASE_32-NEXT: mwaitx
; Restore base pointer.
; USE_BASE_32-NEXT: movq [[SAVE_rbx]], %rbx

; Pass mwaitx 3 arguments in eax, ecx, ebx
; NO_BASE_64: movl %edx, %ebx
; NO_BASE_64: movl %esi, %eax
; NO_BASE_64: movl %edi, %ecx
; No need to save base pointer.
; NO_BASE_64-NOT: movq %rbx
; NO_BASE_64: mwaitx
; NO_BASE_64-NOT: movq {{.*}}, %rbx
; NO_BASE_64-NEXT: {{.+$}}

; Pass mwaitx 3 arguments in eax, ecx, ebx
; NO_BASE_32: movl %edx, %ebx
; NO_BASE_32: movl %esi, %eax
; NO_BASE_32: movl %edi, %ecx
; No need to save base pointer.
; NO_BASE_32-NOT: movl %ebx
; NO_BASE_32: mwaitx
; No need to restore base pointer.
; NO_BASE_32-NOT: movl {{.*}}, %ebx
; NO_BASE_32-NEXT: {{.+$}}

; Test of the case where a variable size object is introduced by a separate
; basic block which, combined with stack realignment, requires a base pointer.
define void @test_variable_size_object(i32 %E, i32 %H, i32 %C, i64 %x) {
entry:
  %ptr = alloca ptr, align 8
  call void @llvm.x86.mwaitx(i32 %E, i32 %H, i32 %C)
  %g = load i32, ptr @g, align 4
  %tobool = icmp ne i32 %g, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:
  %i5 = alloca i8, i64 %x, align 16
  store ptr %i5, ptr %ptr, align 8
  br label %if.end

if.end:
  %ptr2 = load ptr, ptr %ptr, align 8
  %arrayidx = getelementptr inbounds i8, ptr %ptr2, i64 %x
  store volatile i8 42, ptr %arrayidx, align 1
  ret void
}

; CHECK-LABEL: test_variable_size_object:
; USE_BASE_64: movq %rsp, %rbx
; Pass mwaitx first 2 arguments in eax and ecx respectively.
; USE_BASE_64: movl %esi, %eax
; USE_BASE_64: movl %edi, %ecx
; Save base pointer.
; USE_BASE_64: movq %rbx, [[SAVE_rbx:%r([8-9]|1[0-5]|di|si)]]
; Set mwaitx ebx argument.
; USE_BASE_64: movl %edx, %ebx
; USE_BASE_64-NEXT: mwaitx
; Restore base pointer.
; USE_BASE_64-NEXT: movq [[SAVE_rbx]], %rbx

; USE_BASE_32: movl %esp, %ebx
; Pass mwaitx first 2 arguments in eax and ecx respectively.
; USE_BASE_32: movl %esi, %eax
; USE_BASE_32: movl %edi, %ecx
; Save base pointer.
; USE_BASE_32: movq %rbx, [[SAVE_rbx:%r(di|si)]]
; Set mwaitx ebx argument.
; USE_BASE_32: movl %edx, %ebx
; USE_BASE_32-NEXT: mwaitx
; Restore base pointer.
; USE_BASE_32-NEXT: movq [[SAVE_rbx]], %rbx

; Pass mwaitx 3 arguments in eax, ecx, ebx
; NO_BASE_64: movl %edx, %ebx
; NO_BASE_64: movl %esi, %eax
; NO_BASE_64: movl %edi, %ecx
; No need to save base pointer.
; NO_BASE_64-NOT: movq %rbx
; NO_BASE_64: mwaitx
; NO_BASE_64-NOT: movq {{.*}}, %rbx
; NO_BASE_64-NEXT: {{.+$}}

; Pass mwaitx 3 arguments in eax, ecx, ebx
; NO_BASE_32: movl %edx, %ebx
; NO_BASE_32: movl %esi, %eax
; NO_BASE_32: movl %edi, %ecx
; No need to save base pointer.
; NO_BASE_32-NOT: movl %ebx
; NO_BASE_32: mwaitx
; No need to restore base pointer.
; NO_BASE_32-NOT: movl {{.*}}, %ebx
; NO_BASE_32-NEXT: {{.+$}}

declare void @llvm.x86.mwaitx(i32, i32, i32) nounwind
;--- _align32.ll
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"override-stack-alignment", i32 32}
