; Test to ensure that uses and defs in module level asm are handled
; appropriately. Specifically, we should conservatively block importing
; of any references to these values, as they can't be renamed.
; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/module_asm2.ll -o %t2.bc

; RUN: llvm-lto -thinlto-action=run -exported-symbol=main -exported-symbol=func1 -exported-symbol=func2 -exported-symbol=func3 -exported-symbol=callglobalfunc -exported-symbol=callweakfunc %t1.bc %t2.bc
; RUN:  llvm-nm %t1.bc.thinlto.o | FileCheck  %s --check-prefix=NM0
; RUN:  llvm-nm %t2.bc.thinlto.o | FileCheck  %s --check-prefix=NM1

; RUN: llvm-lto2 %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -r=%t1.bc,foo,plx \
; RUN:     -r=%t1.bc,globalfunc,plx \
; RUN:     -r=%t1.bc,globalfunc,plx \
; RUN:     -r=%t1.bc,weakfunc,plx \
; RUN:     -r=%t1.bc,weakfunc,plx \
; RUN:     -r=%t1.bc,b,pl \
; RUN:     -r=%t1.bc,x,pl \
; RUN:     -r=%t1.bc,func1,pl \
; RUN:     -r=%t1.bc,func2,pl \
; RUN:     -r=%t1.bc,func3,pl \
; RUN:     -r=%t1.bc,callglobalfunc,plx \
; RUN:     -r=%t1.bc,callweakfunc,plx \
; RUN:     -r=%t2.bc,main,plx \
; RUN:     -r=%t2.bc,func1,l \
; RUN:     -r=%t2.bc,func2,l \
; RUN:     -r=%t2.bc,func3,l \
; RUN:     -r=%t2.bc,callglobalfunc,l \
; RUN:     -r=%t2.bc,callweakfunc,l
; RUN: llvm-nm %t.o.0 | FileCheck  %s --check-prefix=NM0
; RUN: llvm-nm %t.o.1 | FileCheck  %s --check-prefix=NM1

; Check that local values b and x, which are referenced on
; llvm.used and llvm.compiler.used, respectively, are not promoted.
; Similarly, foo which is defined in module level asm should not be
; promoted.
; NM0-DAG: d b
; NM0-DAG: d x
; NM0-DAG: t foo
; NM0-DAG: T func1
; NM0-DAG: T func2
; NM0-DAG: T func3
; NM0-DAG: T callglobalfunc
; NM0-DAG: T callweakfunc
; NM0-DAG: T globalfunc
; NM0-DAG: W weakfunc

; Ensure that foo, b and x are likewise not exported (imported as refs
; into the other module), since they can't be promoted. Additionally,
; referencing functions func2 and func3 should not have been
; imported. However, we should have been able to import callglobalfunc
; and callweakfunc (leaving undefined symbols globalfunc and weakfunc)
; since globalfunc and weakfunc were defined but not local in module asm.
; NM1-NOT: foo
; NM1-NOT: b
; NM1-NOT: x
; NM1-DAG: U func1
; NM1-DAG: U func2
; NM1-DAG: U func3
; NM1-DAG: U globalfunc
; NM1-DAG: U weakfunc
; NM1-DAG: T main
; NM1-NOT: foo
; NM1-NOT: b
; NM1-NOT: x

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@b = internal global i32 1, align 4
@x = internal global i32 1, align 4

@llvm.compiler.used = appending global [1 x i8*] [i8* bitcast (i32* @b to i8*)], section "llvm.metadata"
@llvm.used = appending global [1 x i8*] [i8* bitcast (i32* @x to i8*)], section "llvm.metadata"

module asm "\09.text"
module asm "\09.type\09foo,@function"
module asm "foo:"
module asm "\09movl    b, %eax"
module asm "\09movl    x, %edx"
module asm "\09ret "
module asm "\09.size\09foo, .-foo"
module asm ""
module asm "\09.globl\09globalfunc"
module asm "\09.type\09globalfunc,@function"
module asm "globalfunc:"
module asm "\09movl    b, %eax"
module asm "\09movl    x, %edx"
module asm "\09ret "
module asm "\09.size\09globalfunc, .-globalfunc"
module asm ""
module asm "\09.weak\09weakfunc"
module asm "\09.type\09weakfunc,@function"
module asm "weakfunc:"
module asm "\09movl    b, %eax"
module asm "\09movl    x, %edx"
module asm "\09ret "
module asm "\09.size\09weakfunc, .-weakfunc"
module asm ""

declare i16 @foo() #0
declare i16 @globalfunc() #0
declare i16 @weakfunc() #0

define i32 @func1() #1 {
  call i16 @foo()
  ret i32 1
}

define i32 @func2() #1 {
  %1 = load i32, i32* @b, align 4
  ret i32 %1
}

define i32 @func3() #1 {
  %1 = load i32, i32* @x, align 4
  ret i32 %1
}

define i32 @callglobalfunc() #1 {
  call i16 @globalfunc()
  ret i32 1
}

define i32 @callweakfunc() #1 {
  call i16 @weakfunc()
  ret i32 1
}
