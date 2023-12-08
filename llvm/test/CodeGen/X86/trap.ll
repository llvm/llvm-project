; RUN: llc < %s -mtriple=i686-apple-darwin8 -mcpu=yonah | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -mtriple=i686-unknown-linux -mcpu=yonah | FileCheck %s -check-prefix=LINUX
; RUN: llc < %s -mtriple=x86_64-scei-ps4 | FileCheck %s -check-prefix=PS4
; RUN: llc < %s -mtriple=x86_64-sie-ps5  | FileCheck %s -check-prefix=PS4
; RUN: llc < %s -mtriple=x86_64-windows-msvc | FileCheck %s -check-prefix=WIN64

; DARWIN-LABEL: test0:
; DARWIN: ud2
; LINUX-LABEL: test0:
; LINUX: ud2
; FIXME: PS4 probably doesn't want two ud2s.
; PS4-LABEL: test0:
; PS4: ud2
; PS4: ud2
; WIN64-LABEL: test0:
; WIN64: ud2
; WIN64-NOT: ud2
define i32 @test0() noreturn nounwind  {
entry:
	tail call void @llvm.trap( )
	unreachable
}

; DARWIN-LABEL: test1:
; DARWIN: int3
; LINUX-LABEL: test1:
; LINUX: int3
; PS4-LABEL: test1:
; PS4: int     $65
; WIN64-LABEL: test1:
; WIN64: int3
; WIN64-NOT: ud2
define i32 @test1() noreturn nounwind  {
entry:
	tail call void @llvm.debugtrap( )
	unreachable
}

declare void @llvm.trap() nounwind 
declare void @llvm.debugtrap() nounwind 

