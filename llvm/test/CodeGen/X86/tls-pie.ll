; RUN: llc < %s -march=x86 -mcpu=generic -mtriple=i386-linux-gnu -relocation-model=pic -enable-pie \
; RUN:   | FileCheck -check-prefix=X32 %s
; RUN: llc < %s -march=x86-64 -mcpu=generic -mtriple=x86_64-linux-gnu -relocation-model=pic -enable-pie \
; RUN:   | FileCheck -check-prefix=X64 %s

@i = thread_local global i32 15
@i2 = external thread_local global i32

define i32 @f1() {
; X32-LABEL: f1:
; X32:      movl %gs:i@NTPOFF, %eax
; X32-NEXT: ret
; X64-LABEL: f1:
; X64:      movl %fs:i@TPOFF, %eax
; X64-NEXT: ret

entry:
	%tmp1 = load i32, i32* @i
	ret i32 %tmp1
}

define i32* @f2() {
; X32-LABEL: f2:
; X32:      movl %gs:0, %eax
; X32-NEXT: leal i@NTPOFF(%eax), %eax
; X32-NEXT: ret
; X64-LABEL: f2:
; X64:      movq %fs:0, %rax
; X64-NEXT: leaq i@TPOFF(%rax), %rax
; X64-NEXT: ret

entry:
	ret i32* @i
}

define i32 @f3() {
; X32-LABEL: f3:
; X32:      calll .L{{[0-9]+}}$pb
; X32-NEXT: .Ltmp{{[0-9]+}}:
; X32-NEXT: .cfi_adjust_cfa_offset 4
; X32-NEXT: .L{{[0-9]+}}$pb:
; X32-NEXT: popl %eax
; X32-NEXT: .Ltmp{{[0-9]+}}:
; X32-NEXT: .cfi_adjust_cfa_offset -4
; X32-NEXT: .Ltmp{{[0-9]+}}:
; X32-NEXT: addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp{{[0-9]+}}-.L{{[0-9]+}}$pb), %eax
; X32-NEXT: movl i2@GOTNTPOFF(%eax), %eax
; X32-NEXT: movl %gs:(%eax), %eax
; X32-NEXT: ret
; X64-LABEL: f3:
; X64:      movq i2@GOTTPOFF(%rip), %rax
; X64-NEXT: movl %fs:(%rax), %eax
; X64-NEXT: ret

entry:
	%tmp1 = load i32, i32* @i2
	ret i32 %tmp1
}

define i32* @f4() {
; X32-LABEL: f4:
; X32:      calll .L{{[0-9]+}}$pb
; X32-NEXT: .Ltmp{{[0-9]+}}:
; X32-NEXT: .cfi_adjust_cfa_offset 4
; X32-NEXT: .L{{[0-9]+}}$pb:
; X32-NEXT: popl %ecx
; X32-NEXT: .Ltmp{{[0-9]+}}:
; X32-NEXT: .cfi_adjust_cfa_offset -4
; X32-NEXT: .Ltmp{{[0-9]+}}:
; X32-NEXT: addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp{{[0-9]+}}-.L{{[0-9]+}}$pb), %ecx
; X32-NEXT: movl %gs:0, %eax
; X32-NEXT: addl i2@GOTNTPOFF(%ecx), %eax
; X32-NEXT: ret
; X64-LABEL: f4:
; X64:      movq %fs:0, %rax
; X64-NEXT: addq i2@GOTTPOFF(%rip), %rax
; X64-NEXT: ret

entry:
	ret i32* @i2
}
