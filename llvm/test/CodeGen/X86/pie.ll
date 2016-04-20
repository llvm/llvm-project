; RUN: llc < %s -O0 -mcpu=generic -mtriple=i686-linux-gnu -relocation-model=pic -enable-pie | FileCheck %s 
; RUN: llc < %s -O0 -mcpu=generic -mtriple=i686-linux-gnu -fast-isel -relocation-model=pic -enable-pie | FileCheck %s 
; RUN: llc < %s -O0 -mcpu=generic -mtriple=x86_64-linux-gnu -relocation-model=pic -enable-pie | FileCheck %s 
; RUN: llc < %s -O0 -mcpu=generic -mtriple=x86_64-linux-gnu -fast-isel -relocation-model=pic -enable-pie | FileCheck %s 

; CHECK-LABEL:  bar:
; CHECK:  call{{l|q}}  foo{{$}}
; CHECK:  call{{l|q}}  weak_odr_foo{{$}}
; CHECK:  call{{l|q}}  weak_foo{{$}}
; CHECK:  call{{l|q}}  internal_foo{{$}}
; CHECK:  call{{l|q}}  ext_baz@PLT

define weak void @weak_foo() {
  ret void
}

define weak_odr void @weak_odr_foo() {
  ret void
}

define internal void @internal_foo() {
  ret void
}

declare i32 @ext_baz()

define void @foo() {
  ret void
}

define void @bar() {
entry:
  call void @foo()
  call void @weak_odr_foo()
  call void @weak_foo()
  call void @internal_foo()
  call i32 @ext_baz()
  ret void
}

; -fpie for local global data tests should be added here
