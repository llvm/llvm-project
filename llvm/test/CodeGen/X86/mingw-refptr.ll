; RUN: llc < %s -mtriple=x86_64-w64-mingw32 | FileCheck %s -check-prefix=CHECK-X64
; RUN: llc < %s -mtriple=i686-w64-mingw32 | FileCheck %s -check-prefix=CHECK-X86
; RUN: llc < %s -mtriple=i686-w64-mingw32-none-elf | FileCheck %s -check-prefix=CHECK-X86-ELF

@var = external local_unnamed_addr global i32, align 4
@dsolocalvar = external dso_local local_unnamed_addr global i32, align 4
@localvar = dso_local local_unnamed_addr global i32 0, align 4
@localcommon = common dso_local local_unnamed_addr global i32 0, align 4
@extvar = external dllimport local_unnamed_addr global i32, align 4

define dso_local i32 @getVar() {
; CHECK-X64-LABEL: getVar:
; CHECK-X64:    movq .refptr.var(%rip), %rax
; CHECK-X64:    movl (%rax), %eax
; CHECK-X64:    retq
; CHECK-X86-LABEL: _getVar:
; CHECK-X86:    movl .refptr._var, %eax
; CHECK-X86:    movl (%eax), %eax
; CHECK-X86:    retl
; CHECK-X86-ELF-LABEL: getVar:
; CHECK-X86-ELF:    movl var, %eax
; CHECK-X86-ELF:    retl
entry:
  %0 = load i32, i32* @var, align 4
  ret i32 %0
}

define dso_local i32 @getDsoLocalVar() {
; CHECK-X64-LABEL: getDsoLocalVar:
; CHECK-X64:    movl dsolocalvar(%rip), %eax
; CHECK-X64:    retq
; CHECK-X86-LABEL: _getDsoLocalVar:
; CHECK-X86:    movl _dsolocalvar, %eax
; CHECK-X86:    retl
entry:
  %0 = load i32, i32* @dsolocalvar, align 4
  ret i32 %0
}

define dso_local i32 @getLocalVar() {
; CHECK-X64-LABEL: getLocalVar:
; CHECK-X64:    movl localvar(%rip), %eax
; CHECK-X64:    retq
; CHECK-X86-LABEL: _getLocalVar:
; CHECK-X86:    movl _localvar, %eax
; CHECK-X86:    retl
entry:
  %0 = load i32, i32* @localvar, align 4
  ret i32 %0
}

define dso_local i32 @getLocalCommon() {
; CHECK-X64-LABEL: getLocalCommon:
; CHECK-X64:    movl localcommon(%rip), %eax
; CHECK-X64:    retq
; CHECK-X86-LABEL: _getLocalCommon:
; CHECK-X86:    movl _localcommon, %eax
; CHECK-X86:    retl
entry:
  %0 = load i32, i32* @localcommon, align 4
  ret i32 %0
}

define dso_local i32 @getExtVar() {
; CHECK-X64-LABEL: getExtVar:
; CHECK-X64:    movq __imp_extvar(%rip), %rax
; CHECK-X64:    movl (%rax), %eax
; CHECK-X64:    retq
; CHECK-X86-LABEL: _getExtVar:
; CHECK-X86:    movl __imp__extvar, %eax
; CHECK-X86:    movl (%eax), %eax
; CHECK-X86:    retl
; CHECK-X86-ELF-LABEL: getExtVar:
; CHECK-X86-ELF:    movl extvar, %eax
; CHECK-X86-ELF:    retl
entry:
  %0 = load i32, i32* @extvar, align 4
  ret i32 %0
}

define dso_local void @callFunc() {
; CHECK-X64-LABEL: callFunc:
; CHECK-X64:    jmp otherFunc
; CHECK-X86-LABEL: _callFunc:
; CHECK-X86:    jmp _otherFunc
entry:
  tail call void @otherFunc()
  ret void
}

declare dso_local void @otherFunc()

; CHECK-X64:        .section        .rdata$.refptr.var,"dr",discard,.refptr.var
; CHECK-X64:        .globl  .refptr.var
; CHECK-X64: .refptr.var:
; CHECK-X64:        .quad   var

; CHECK-X86:        .section        .rdata$.refptr._var,"dr",discard,.refptr._var
; CHECK-X86:        .globl  .refptr._var
; CHECK-X86: .refptr._var:
; CHECK-X86:        .long   _var
