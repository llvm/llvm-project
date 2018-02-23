; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux-gnu -relocation-model=pic \
; RUN:   | FileCheck -check-prefix=X64 --check-prefix=PIC %s
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux-gnu \
; RUN:   | FileCheck -check-prefix=X64 --check-prefix=STATIC %s

define void @memset_call(i8* nocapture %a, i8 %c, i32 %n) {
; X64: callq *memset@GOTPCREL(%rip)
  call void @llvm.memset.p0i8.i32(i8* %a, i8 %c, i32 %n, i1 false)
  ret void
}

define void @memcpy_call(i8* nocapture %a, i8* nocapture readonly %b, i64 %n) {
; X64: callq *memcpy@GOTPCREL(%rip)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* %b, i64 %n, i32 1, i1 false)
  ret void
}

define i32 @main() {
; X64:    callq *foo@GOTPCREL(%rip)
; PIC:    callq bar@PLT
; STATIC: callq bar{{$}}
; X64:    callq baz

  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %call1 = call i32 @foo()
  %call2 = call i32 @bar()
  %call3 = call i32 @baz()
  ret i32 0
}

declare i32 @foo() nonlazybind
declare i32 @bar()
declare hidden i32 @baz() nonlazybind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1)
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1)

!llvm.module.flags = !{!1}
!1 = !{i32 7, !"RtLibUseGOT", i32 1}
