; RUN: llc < %s -mtriple=i686-apple-darwin | FileCheck %s --check-prefix=DARWIN
; RUN: llc < %s -mtriple=i686-windows-msvc | FileCheck %s --check-prefix=WIN32
; RUN: llc < %s -mtriple=x86_64-windows-msvc | FileCheck %s --check-prefix=WIN64

declare extern_weak void @foo(...)

define void @bar() {
entry:
  br i1 icmp ne (ptr @foo, ptr null), label %if.then, label %if.end

if.then:
  tail call void (...) @foo( )
  ret void

if.end:
  ret void
}

; DARWIN-LABEL: _bar:
; DARWIN: cmpl $0, L_foo$non_lazy_ptr
; DARWIN: jmp _foo ## TAILCALL

; WIN32-LABEL: _bar:
; WIN32: cmpl $0, .refptr._foo
; WIN32: jmpl *.refptr._foo

; WIN64-LABEL: bar:
; WIN64: cmpq $0, .refptr.foo(%rip)
; WIN64: jmpq *.refptr.foo


declare extern_weak i32 @X(ptr)

@Y = global ptr @X               ; <ptr> [#uses=0]

; DARWIN-LABEL: _Y:
; DARWIN: .long _X

; WIN32-LABEL: _Y:
; WIN32: .long _X

; WIN64-LABEL: Y:
; WIN64: .quad X


; DARWIN: .weak_reference _foo
; DARWIN: .weak_reference _X

; WIN32:         .section        .rdata$.refptr._foo,"dr",discard,.refptr._foo
; WIN32:         .globl  .refptr._foo
; WIN32: .refptr._foo:
; WIN32:         .long   _foo

; WIN32: .weak _foo
; WIN32: .weak _X

; WIN64:         .section        .rdata$.refptr.foo,"dr",discard,.refptr.foo
; WIN64:         .globl  .refptr.foo
; WIN64: .refptr.foo:
; WIN64:         .quad   foo

; WIN64: .weak foo
; WIN64: .weak X

