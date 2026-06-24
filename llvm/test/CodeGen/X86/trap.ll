; RUN: llc < %s -mtriple=i686-apple-darwin8 -mcpu=yonah | FileCheck %s -check-prefixes=CHECK,DARWIN
; RUN: llc < %s -mtriple=i686-unknown-linux -mcpu=yonah | FileCheck %s -check-prefixes=CHECK,LINUX
; RUN: llc < %s -mtriple=x86_64-scei-ps4 | FileCheck %s -check-prefixes=CHECK,PS4
; RUN: llc < %s -mtriple=x86_64-sie-ps5  | FileCheck %s -check-prefixes=CHECK,PS4
; RUN: llc < %s -mtriple=x86_64-windows-msvc | FileCheck %s -check-prefixes=CHECK,WIN64

; CHECK-LABEL: test0:
; CHECK: ud2
; CHECK-NOT: ud2
define i32 @test0() noreturn nounwind  {
entry:
	tail call void @llvm.trap( )
	unreachable
}

; CHECK-LABEL: test1:
; DARWIN: int3
; LINUX: int3
; PS4: int     $65
; WIN64: int3
; WIN64-NOT: ud2
define i32 @test1() noreturn nounwind  {
entry:
	tail call void @llvm.debugtrap( )
	unreachable
}

declare void @llvm.trap() nounwind 
declare void @llvm.debugtrap() nounwind 
