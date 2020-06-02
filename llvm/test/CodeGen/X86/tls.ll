; RUN: llc < %s -mtriple=i386-linux-gnu | FileCheck -check-prefix=X86_LINUX %s
; RUN: llc < %s -mtriple=x86_64-linux-gnu | FileCheck -check-prefix=X64_LINUX %s
; RUN: llc < %s -mtriple=i386-linux-gnu -fast-isel | FileCheck -check-prefix=X86_ISEL_LINUX %s
; RUN: llc < %s -mtriple=x86_64-linux-gnu -fast-isel | FileCheck -check-prefix=X64_ISEL_LINUX %s
; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck -check-prefix=X86_WIN %s
; RUN: llc < %s -mtriple=x86_64-pc-win32 | FileCheck -check-prefix=X64_WIN %s
; RUN: llc < %s -mtriple=i686-pc-windows-gnu | FileCheck -check-prefix=MINGW32 %s
; RUN: llc < %s -mtriple=x86_64-pc-windows-gnu | FileCheck -check-prefix=X64_WIN %s

@i1 = thread_local global i32 15
@i2 = external thread_local global i32
@i3 = internal thread_local global i32 15
@i4 = hidden thread_local global i32 15
@i5 = external hidden thread_local global i32
@i6 = external protected thread_local global i32
@s1 = thread_local global i16 15
@b1 = thread_local global i8 0
@b2 = thread_local(localexec) global i8 0

define i32 @f1() {
; X86_LINUX-LABEL: f1:
; X86_LINUX:      movl %gs:i1@NTPOFF, %eax
; X86_LINUX-NEXT: ret
; X64_LINUX-LABEL: f1:
; X64_LINUX:      movl %fs:i1@TPOFF, %eax
; X64_LINUX-NEXT: ret
; X86_WIN-LABEL: f1:
; X86_WIN:      movl __tls_index, %eax
; X86_WIN-NEXT: movl %fs:__tls_array, %ecx
; X86_WIN-NEXT: movl (%ecx,%eax,4), %eax
; X86_WIN-NEXT: movl _i1@SECREL32(%eax), %eax
; X86_WIN-NEXT: ret
; X64_WIN-LABEL: f1:
; X64_WIN:      movl _tls_index(%rip), %eax
; X64_WIN-NEXT: movq %gs:88, %rcx
; X64_WIN-NEXT: movq (%rcx,%rax,8), %rax
; X64_WIN-NEXT: movl i1@SECREL32(%rax), %eax
; X64_WIN-NEXT: ret
; MINGW32-LABEL: _f1:
; MINGW32: movl __tls_index, %eax
; MINGW32-NEXT: movl %fs:44, %ecx
; MINGW32-NEXT: movl (%ecx,%eax,4), %eax
; MINGW32-NEXT: movl _i1@SECREL32(%eax), %eax
; MINGW32-NEXT: retl

entry:
	%tmp1 = load i32, i32* @i1
	ret i32 %tmp1
}

define i32* @f2() {
; X86_LINUX-LABEL: f2:
; X86_LINUX:      movl %gs:0, %eax
; X86_LINUX-NEXT: leal i1@NTPOFF(%eax), %eax
; X86_LINUX-NEXT: ret
; X64_LINUX-LABEL: f2:
; X64_LINUX:      movq %fs:0, %rax
; X64_LINUX-NEXT: leaq i1@TPOFF(%rax), %rax
; X64_LINUX-NEXT: ret
; X86_WIN-LABEL: f2:
; X86_WIN:      movl __tls_index, %eax
; X86_WIN-NEXT: movl %fs:__tls_array, %ecx
; X86_WIN-NEXT: movl (%ecx,%eax,4), %eax
; X86_WIN-NEXT: leal _i1@SECREL32(%eax), %eax
; X86_WIN-NEXT: ret
; X64_WIN-LABEL: f2:
; X64_WIN:      movl _tls_index(%rip), %eax
; X64_WIN-NEXT: movq %gs:88, %rcx
; X64_WIN-NEXT: movq (%rcx,%rax,8), %rax
; X64_WIN-NEXT: leaq i1@SECREL32(%rax), %rax
; X64_WIN-NEXT: ret
; MINGW32-LABEL: _f2:
; MINGW32: movl __tls_index, %eax
; MINGW32-NEXT: movl %fs:44, %ecx
; MINGW32-NEXT: movl (%ecx,%eax,4), %eax
; MINGW32-NEXT: leal _i1@SECREL32(%eax), %eax
; MINGW32-NEXT: retl

entry:
	ret i32* @i1
}

define i32 @f3() nounwind {
; X86_LINUX-LABEL: f3:
; X86_LINUX:      movl i2@INDNTPOFF, %eax
; X86_LINUX-NEXT: movl %gs:(%eax), %eax
; X86_LINUX-NEXT: ret
; X64_LINUX-LABEL: f3:
; X64_LINUX:      movq i2@GOTTPOFF(%rip), %rax
; X64_LINUX-NEXT: movl %fs:(%rax), %eax
; X64_LINUX-NEXT: ret
; X86_WIN-LABEL: f3:
; X86_WIN:      movl __tls_index, %eax
; X86_WIN-NEXT: movl %fs:__tls_array, %ecx
; X86_WIN-NEXT: movl (%ecx,%eax,4), %eax
; X86_WIN-NEXT: movl _i2@SECREL32(%eax), %eax
; X86_WIN-NEXT: ret
; X64_WIN-LABEL: f3:
; X64_WIN:      movl _tls_index(%rip), %eax
; X64_WIN-NEXT: movq %gs:88, %rcx
; X64_WIN-NEXT: movq (%rcx,%rax,8), %rax
; X64_WIN-NEXT: movl i2@SECREL32(%rax), %eax
; X64_WIN-NEXT: ret
; MINGW32-LABEL: _f3:
; MINGW32: movl __tls_index, %eax
; MINGW32-NEXT: movl %fs:44, %ecx
; MINGW32-NEXT: movl (%ecx,%eax,4), %eax
; MINGW32-NEXT: movl _i2@SECREL32(%eax), %eax
; MINGW32-NEXT: retl

entry:
	%tmp1 = load i32, i32* @i2
	ret i32 %tmp1
}

define i32* @f4() {
; X86_LINUX-LABEL: f4:
; X86_LINUX:      movl %gs:0, %eax
; X86_LINUX-NEXT: addl i2@INDNTPOFF, %eax
; X86_LINUX-NEXT: ret
; X64_LINUX-LABEL: f4:
; X64_LINUX:      movq %fs:0, %rax
; X64_LINUX-NEXT: addq i2@GOTTPOFF(%rip), %rax
; X64_LINUX-NEXT: ret
; X86_WIN-LABEL: f4:
; X86_WIN:      movl __tls_index, %eax
; X86_WIN-NEXT: movl %fs:__tls_array, %ecx
; X86_WIN-NEXT: movl (%ecx,%eax,4), %eax
; X86_WIN-NEXT: leal _i2@SECREL32(%eax), %eax
; X86_WIN-NEXT: ret
; X64_WIN-LABEL: f4:
; X64_WIN:      movl _tls_index(%rip), %eax
; X64_WIN-NEXT: movq %gs:88, %rcx
; X64_WIN-NEXT: movq (%rcx,%rax,8), %rax
; X64_WIN-NEXT: leaq i2@SECREL32(%rax), %rax
; X64_WIN-NEXT: ret
; MINGW32-LABEL: _f4:
; MINGW32: movl __tls_index, %eax
; MINGW32-NEXT: movl %fs:44, %ecx
; MINGW32-NEXT: movl (%ecx,%eax,4), %eax
; MINGW32-NEXT: leal _i2@SECREL32(%eax), %eax
; MINGW32-NEXT: retl

entry:
	ret i32* @i2
}

define i32 @f5() nounwind {
; X86_LINUX-LABEL: f5:
; X86_LINUX:      movl %gs:i3@NTPOFF, %eax
; X86_LINUX-NEXT: ret
; X64_LINUX-LABEL: f5:
; X64_LINUX:      movl %fs:i3@TPOFF, %eax
; X64_LINUX-NEXT: ret
; X86_WIN-LABEL: f5:
; X86_WIN:      movl __tls_index, %eax
; X86_WIN-NEXT: movl %fs:__tls_array, %ecx
; X86_WIN-NEXT: movl (%ecx,%eax,4), %eax
; X86_WIN-NEXT: movl _i3@SECREL32(%eax), %eax
; X86_WIN-NEXT: ret
; X64_WIN-LABEL: f5:
; X64_WIN:      movl _tls_index(%rip), %eax
; X64_WIN-NEXT: movq %gs:88, %rcx
; X64_WIN-NEXT: movq (%rcx,%rax,8), %rax
; X64_WIN-NEXT: movl i3@SECREL32(%rax), %eax
; X64_WIN-NEXT: ret
; MINGW32-LABEL: _f5:
; MINGW32: movl __tls_index, %eax
; MINGW32-NEXT: movl %fs:44, %ecx
; MINGW32-NEXT: movl (%ecx,%eax,4), %eax
; MINGW32-NEXT: movl _i3@SECREL32(%eax), %eax
; MINGW32-NEXT: retl

entry:
	%tmp1 = load i32, i32* @i3
	ret i32 %tmp1
}

define i32* @f6() {
; X86_LINUX-LABEL: f6:
; X86_LINUX:      movl %gs:0, %eax
; X86_LINUX-NEXT: leal i3@NTPOFF(%eax), %eax
; X86_LINUX-NEXT: ret
; X64_LINUX-LABEL: f6:
; X64_LINUX:      movq %fs:0, %rax
; X64_LINUX-NEXT: leaq i3@TPOFF(%rax), %rax
; X64_LINUX-NEXT: ret
; X86_WIN-LABEL: f6:
; X86_WIN:      movl __tls_index, %eax
; X86_WIN-NEXT: movl %fs:__tls_array, %ecx
; X86_WIN-NEXT: movl (%ecx,%eax,4), %eax
; X86_WIN-NEXT: leal _i3@SECREL32(%eax), %eax
; X86_WIN-NEXT: ret
; X64_WIN-LABEL: f6:
; X64_WIN:      movl _tls_index(%rip), %eax
; X64_WIN-NEXT: movq %gs:88, %rcx
; X64_WIN-NEXT: movq (%rcx,%rax,8), %rax
; X64_WIN-NEXT: leaq i3@SECREL32(%rax), %rax
; X64_WIN-NEXT: ret
; MINGW32-LABEL: _f6:
; MINGW32: movl __tls_index, %eax
; MINGW32-NEXT: movl %fs:44, %ecx
; MINGW32-NEXT: movl (%ecx,%eax,4), %eax
; MINGW32-NEXT: leal _i3@SECREL32(%eax), %eax
; MINGW32-NEXT: retl

entry:
	ret i32* @i3
}

define i32 @f7() {
; X86_LINUX-LABEL: f7:
; X86_LINUX:      movl %gs:i4@NTPOFF, %eax
; X86_LINUX-NEXT: ret
; X64_LINUX-LABEL: f7:
; X64_LINUX:      movl %fs:i4@TPOFF, %eax
; X64_LINUX-NEXT: ret
; MINGW32-LABEL: _f7:
; MINGW32: movl __tls_index, %eax
; MINGW32-NEXT: movl %fs:44, %ecx
; MINGW32-NEXT: movl (%ecx,%eax,4), %eax
; MINGW32-NEXT: movl _i4@SECREL32(%eax), %eax
; MINGW32-NEXT: retl

entry:
	%tmp1 = load i32, i32* @i4
	ret i32 %tmp1
}

define i32* @f8() {
; X86_LINUX-LABEL: f8:
; X86_LINUX:      movl %gs:0, %eax
; X86_LINUX-NEXT: leal i4@NTPOFF(%eax), %eax
; X86_LINUX-NEXT: ret
; X64_LINUX-LABEL: f8:
; X64_LINUX:      movq %fs:0, %rax
; X64_LINUX-NEXT: leaq i4@TPOFF(%rax), %rax
; X64_LINUX-NEXT: ret
; MINGW32-LABEL: _f8:
; MINGW32: movl __tls_index, %eax
; MINGW32-NEXT: movl %fs:44, %ecx
; MINGW32-NEXT: movl (%ecx,%eax,4), %eax
; MINGW32-NEXT: leal _i4@SECREL32(%eax), %eax
; MINGW32-NEXT: retl

entry:
	ret i32* @i4
}

define i32 @f9() {
; X86_LINUX-LABEL: f9:
; X86_LINUX:      movl %gs:i5@NTPOFF, %eax
; X86_LINUX-NEXT: ret
; X64_LINUX-LABEL: f9:
; X64_LINUX:      movl %fs:i5@TPOFF, %eax
; X64_LINUX-NEXT: ret
; MINGW32-LABEL: _f9:
; MINGW32: movl __tls_index, %eax
; MINGW32-NEXT: movl %fs:44, %ecx
; MINGW32-NEXT: movl (%ecx,%eax,4), %eax
; MINGW32-NEXT: movl _i5@SECREL32(%eax), %eax
; MINGW32-NEXT: retl

entry:
	%tmp1 = load i32, i32* @i5
	ret i32 %tmp1
}

define i32* @f10() {
; X86_LINUX-LABEL: f10:
; X86_LINUX:      movl %gs:0, %eax
; X86_LINUX-NEXT: leal i5@NTPOFF(%eax), %eax
; X86_LINUX-NEXT: ret
; X64_LINUX-LABEL: f10:
; X64_LINUX:      movq %fs:0, %rax
; X64_LINUX-NEXT: leaq i5@TPOFF(%rax), %rax
; X64_LINUX-NEXT: ret
; MINGW32-LABEL: _f10:
; MINGW32: movl __tls_index, %eax
; MINGW32-NEXT: movl %fs:44, %ecx
; MINGW32-NEXT: movl (%ecx,%eax,4), %eax
; MINGW32-NEXT: leal _i5@SECREL32(%eax), %eax
; MINGW32-NEXT: retl

entry:
	ret i32* @i5
}

define i16 @f11() {
; X86_LINUX-LABEL: f11:
; X86_LINUX:      movzwl %gs:s1@NTPOFF, %eax
; X86_LINUX:      ret
; X64_LINUX-LABEL: f11:
; X64_LINUX:      movzwl %fs:s1@TPOFF, %eax
; X64_LINUX:      ret
; X86_WIN-LABEL: f11:
; X86_WIN:      movl __tls_index, %eax
; X86_WIN-NEXT: movl %fs:__tls_array, %ecx
; X86_WIN-NEXT: movl (%ecx,%eax,4), %eax
; X86_WIN-NEXT: movzwl _s1@SECREL32(%eax), %eax
; X86_WIN:      ret
; X64_WIN-LABEL: f11:
; X64_WIN:      movl _tls_index(%rip), %eax
; X64_WIN-NEXT: movq %gs:88, %rcx
; X64_WIN-NEXT: movq (%rcx,%rax,8), %rax
; X64_WIN-NEXT: movzwl s1@SECREL32(%rax), %eax
; X64_WIN:      ret
; MINGW32-LABEL: _f11:
; MINGW32: movl __tls_index, %eax
; MINGW32-NEXT: movl %fs:44, %ecx
; MINGW32-NEXT: movl (%ecx,%eax,4), %eax
; MINGW32-NEXT: movzwl  _s1@SECREL32(%eax), %eax
; MINGW32: retl

entry:
	%tmp1 = load i16, i16* @s1
	ret i16 %tmp1
}

define i32 @f12() {
; X86_LINUX-LABEL: f12:
; X86_LINUX:      movswl %gs:s1@NTPOFF, %eax
; X86_LINUX-NEXT: ret
; X64_LINUX-LABEL: f12:
; X64_LINUX:      movswl %fs:s1@TPOFF, %eax
; X64_LINUX-NEXT: ret
; X86_WIN-LABEL: f12:
; X86_WIN:      movl __tls_index, %eax
; X86_WIN-NEXT: movl %fs:__tls_array, %ecx
; X86_WIN-NEXT: movl (%ecx,%eax,4), %eax
; X86_WIN-NEXT: movswl _s1@SECREL32(%eax), %eax
; X86_WIN-NEXT: ret
; X64_WIN-LABEL: f12:
; X64_WIN:      movl _tls_index(%rip), %eax
; X64_WIN-NEXT: movq %gs:88, %rcx
; X64_WIN-NEXT: movq (%rcx,%rax,8), %rax
; X64_WIN-NEXT: movswl s1@SECREL32(%rax), %eax
; X64_WIN-NEXT: ret
; MINGW32-LABEL: _f12:
; MINGW32: movl __tls_index, %eax
; MINGW32-NEXT: movl %fs:44, %ecx
; MINGW32-NEXT: movl (%ecx,%eax,4), %eax
; MINGW32-NEXT: movswl _s1@SECREL32(%eax), %eax
; MINGW32-NEXT: retl


entry:
	%tmp1 = load i16, i16* @s1
  %tmp2 = sext i16 %tmp1 to i32
	ret i32 %tmp2
}

define i8 @f13() {
; X86_LINUX-LABEL: f13:
; X86_LINUX:      movb %gs:b1@NTPOFF, %al
; X86_LINUX-NEXT: ret
; X64_LINUX-LABEL: f13:
; X64_LINUX:      movb %fs:b1@TPOFF, %al
; X64_LINUX-NEXT: ret
; X86_WIN-LABEL: f13:
; X86_WIN:      movl __tls_index, %eax
; X86_WIN-NEXT: movl %fs:__tls_array, %ecx
; X86_WIN-NEXT: movl (%ecx,%eax,4), %eax
; X86_WIN-NEXT: movb _b1@SECREL32(%eax), %al
; X86_WIN-NEXT: ret
; X64_WIN-LABEL: f13:
; X64_WIN:      movl _tls_index(%rip), %eax
; X64_WIN-NEXT: movq %gs:88, %rcx
; X64_WIN-NEXT: movq (%rcx,%rax,8), %rax
; X64_WIN-NEXT: movb b1@SECREL32(%rax), %al
; X64_WIN-NEXT: ret
; MINGW32-LABEL: _f13:
; MINGW32: movl __tls_index, %eax
; MINGW32-NEXT: movl %fs:44, %ecx
; MINGW32-NEXT: movl (%ecx,%eax,4), %eax
; MINGW32-NEXT: movb _b1@SECREL32(%eax), %al
; MINGW32-NEXT: retl

entry:
	%tmp1 = load i8, i8* @b1
	ret i8 %tmp1
}

define i32 @f14() {
; X86_LINUX-LABEL: f14:
; X86_LINUX:      movsbl %gs:b1@NTPOFF, %eax
; X86_LINUX-NEXT: ret
; X64_LINUX-LABEL: f14:
; X64_LINUX:      movsbl %fs:b1@TPOFF, %eax
; X64_LINUX-NEXT: ret
; X86_WIN-LABEL: f14:
; X86_WIN:      movl __tls_index, %eax
; X86_WIN-NEXT: movl %fs:__tls_array, %ecx
; X86_WIN-NEXT: movl (%ecx,%eax,4), %eax
; X86_WIN-NEXT: movsbl _b1@SECREL32(%eax), %eax
; X86_WIN-NEXT: ret
; X64_WIN-LABEL: f14:
; X64_WIN:      movl _tls_index(%rip), %eax
; X64_WIN-NEXT: movq %gs:88, %rcx
; X64_WIN-NEXT: movq (%rcx,%rax,8), %rax
; X64_WIN-NEXT: movsbl b1@SECREL32(%rax), %eax
; X64_WIN-NEXT: ret
; MINGW32-LABEL: _f14:
; MINGW32: movl __tls_index, %eax
; MINGW32-NEXT: movl %fs:44, %ecx
; MINGW32-NEXT: movl (%ecx,%eax,4), %eax
; MINGW32-NEXT: movsbl  _b1@SECREL32(%eax), %eax
; MINGW32-NEXT: retl

entry:
	%tmp1 = load i8, i8* @b1
  %tmp2 = sext i8 %tmp1 to i32
	ret i32 %tmp2
}

define i8* @f15() {
; X86_LINUX-LABEL: f15:
; X86_LINUX:      movl %gs:0, %eax
; X86_LINUX-NEXT: leal b2@NTPOFF(%eax), %eax
; X86_LINUX-NEXT: ret
; X64_LINUX-LABEL: f15:
; X64_LINUX:      movq %fs:0, %rax
; X64_LINUX-NEXT: leaq b2@TPOFF(%rax), %rax
; X64_LINUX-NEXT: ret
; X86_WIN-LABEL: f15:
; X86_WIN:      movl %fs:__tls_array, %eax
; X86_WIN-NEXT: movl (%eax), %eax
; X86_WIN-NEXT: leal _b2@SECREL32(%eax), %eax
; X86_WIN-NEXT: ret
; X64_WIN-LABEL: f15:
; X64_WIN:      movq %gs:88, %rax
; X64_WIN-NEXT: movq (%rax), %rax
; X64_WIN-NEXT: leaq b2@SECREL32(%rax), %rax
; X64_WIN-NEXT: ret
; MINGW32-LABEL: f15:
; MINGW32:      movl %fs:44, %eax
; MINGW32-NEXT: movl (%eax), %eax
; MINGW32-NEXT: leal _b2@SECREL32(%eax), %eax
; MINGW32-NEXT: ret
entry:
	ret i8* @b2
}


define i32* @f16() {
; X86_LINUX-LABEL: f16:
; X86_LINUX:       movl %gs:0, %eax
; X86_LINUX-NEXT:  leal i6@NTPOFF(%eax), %eax
; X86_LINUX-NEXT:  ret

; X64_LINUX-LABEL: f16:
; X64_LINUX:       movq %fs:0, %rax
; X64_LINUX-NEXT:  leaq i6@TPOFF(%rax), %rax
; X64_LINUX-NEXT:  ret

  ret i32* @i6
}

; NOTE: Similar to f1() but with direct TLS segment access disabled
define i32 @f17() #0 {
; X86_LINUX-LABEL: f17:
; X86_LINUX:      movl %gs:0, %eax
; X86_LINUX-NEXT: movl i1@NTPOFF(%eax), %eax
; X86_LINUX-NEXT: ret
; X64_LINUX-LABEL: f17:
; X64_LINUX:      movq %fs:0, %rax
; X64_LINUX-NEXT: movl i1@TPOFF(%rax), %eax
; X64_LINUX-NEXT: ret
; X86_ISEL_LINUX-LABEL: f17:
; X86_ISEL_LINUX:      movl %gs:0, %eax
; X86_ISEL_LINUX-NEXT: movl i1@NTPOFF(%eax), %eax
; X86_ISEL_LINUX-NEXT: ret
; X64_ISEL_LINUX-LABEL: f17:
; X64_ISEL_LINUX:      movq %fs:0, %rax
; X64_ISEL_LINUX-NEXT: movl i1@TPOFF(%rax), %eax
; X64_ISEL_LINUX-NEXT: ret

entry:
	%tmp1 = load i32, i32* @i1
	ret i32 %tmp1
}

; NOTE: Similar to f3() but with direct TLS segment access disabled
define i32 @f18() #1 {
; X86_LINUX-LABEL: f18:
; X86_LINUX:      movl i2@INDNTPOFF, %eax
; X86_LINUX-NEXT: movl %gs:0, %ecx
; X86_LINUX-NEXT: movl (%ecx,%eax), %eax
; X86_LINUX-NEXT: ret
; X64_LINUX-LABEL: f18:
; X64_LINUX:      movq i2@GOTTPOFF(%rip), %rax
; X64_LINUX-NEXT: movq %fs:0, %rcx
; X64_LINUX-NEXT: movl (%rcx,%rax), %eax
; X64_LINUX-NEXT: ret
; X86_ISEL_LINUX-LABEL: f18:
; X86_ISEL_LINUX:      movl i2@INDNTPOFF, %eax
; X86_ISEL_LINUX-NEXT: movl %gs:0, %ecx
; X86_ISEL_LINUX-NEXT: movl (%ecx,%eax), %eax
; X86_ISEL_LINUX-NEXT: ret
; X64_ISEL_LINUX-LABEL: f18:
; X64_ISEL_LINUX:      movq i2@GOTTPOFF(%rip), %rax
; X64_ISEL_LINUX-NEXT: movq %fs:0, %rcx
; X64_ISEL_LINUX-NEXT: movl (%rcx,%rax), %eax
; X64_ISEL_LINUX-NEXT: ret


entry:
	%tmp1 = load i32, i32* @i2
	ret i32 %tmp1
}

attributes #0 = { "indirect-tls-seg-refs" }
attributes #1 = { nounwind "indirect-tls-seg-refs" }
