; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux-gnu -relocation-model=pic \
; RUN:   | FileCheck -check-prefix=X64 --check-prefix=PIC %s
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux-gnu \
; RUN:   | FileCheck -check-prefix=X64 --check-prefix=STATIC %s

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
