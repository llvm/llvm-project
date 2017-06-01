; RUN: llc < %s -mcpu=generic -mtriple=i386-linux | FileCheck %s -check-prefix=LINUX-I386
; RUN: llc < %s -mcpu=generic -mtriple=i386-kfreebsd | FileCheck %s -check-prefix=KFREEBSD-I386
; RUN: llc < %s -mcpu=generic -mtriple=i386-netbsd | FileCheck %s -check-prefix=NETBSD-I386
; RUN: llc < %s -mcpu=generic -mtriple=i686-apple-darwin8 | FileCheck %s -check-prefix=DARWIN-I386
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux | FileCheck %s -check-prefix=LINUX-X86_64
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-kfreebsd | FileCheck %s -check-prefix=KFREEBSD-X86_64
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-netbsd | FileCheck %s -check-prefix=NETBSD-X86_64
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-apple-darwin8 | FileCheck %s -check-prefix=DARWIN-X86_64

define i32 @test() nounwind {
entry:
  call void @test2()
  ret i32 0

; LINUX-I386:     subl	$12, %esp
; KFREEBSD-I386:  subl	$12, %esp
; DARWIN-I386:    subl	$12, %esp
; NETBSD-I386-NOT: subl	{{.*}}, %esp

; LINUX-X86_64:      pushq %{{.*}}
; LINUX-X86_64-NOT:  subq	{{.*}}, %rsp
; DARWIN-X86_64:     pushq %{{.*}}
; DARWIN-X86_64-NOT: subq	{{.*}}, %rsp
; NETBSD-X86_64:     pushq %{{.*}}
; NETBSD-X86_64-NOT: subq	{{.*}}, %rsp
; KFREEBSD-X86_64:     pushq %{{.*}}
; KFREEBSD-X86_64-NOT: subq	{{.*}}, %rsp
}

declare void @test2()
