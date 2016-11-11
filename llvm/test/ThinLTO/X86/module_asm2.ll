; Test to ensure that uses and defs in module level asm are handled
; appropriately. Specifically, we should conservatively block importing
; of any references to these values, as they can't be renamed.
; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/module_asm2.ll -o %t2.bc

; RUN: llvm-lto -thinlto-action=run -exported-symbol=main -exported-symbol=func1 -exported-symbol=func2 -exported-symbol=func3 %t1.bc %t2.bc
; RUN: llvm-nm %t1.bc.thinlto.o | FileCheck  %s --check-prefix=NM0
; RUN: llvm-nm %t2.bc.thinlto.o | FileCheck  %s --check-prefix=NM1

; RUN: llvm-lto2 %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -r=%t1.bc,foo,plx \
; RUN:     -r=%t1.bc,b,pl \
; RUN:     -r=%t1.bc,x,pl \
; RUN:     -r=%t1.bc,func1,pl \
; RUN:     -r=%t1.bc,func2,pl \
; RUN:     -r=%t1.bc,func3,pl \
; RUN:     -r=%t2.bc,main,plx \
; RUN:     -r=%t2.bc,func1,l \
; RUN:     -r=%t2.bc,func2,l \
; RUN:     -r=%t2.bc,func3,l
; RUN: llvm-nm %t.o.0 | FileCheck  %s --check-prefix=NM0
; RUN: llvm-nm %t.o.1 | FileCheck  %s --check-prefix=NM1

; Check that local values b and x, which are referenced on
; llvm.used and llvm.compiler.used, respectively, are not promoted.
; Similarly, foo which is defined in module level asm should not be
; promoted. We have to check in the importing module, however, as we
; don't currently generate a summary for values defined in module asm,
; so they couldn't get promoted even if we exported a reference.
; NM0-DAG: d b
; NM0-DAG: d x
; NM0-DAG: t foo
; NM0-DAG: T func1
; NM0-DAG: T func2
; NM0-DAG: T func3

; Ensure that b and x are likewise not exported (imported as refs
; into the other module), since they can't be promoted. Additionally,
; referencing functions func2 and func3 should not have been
; imported.
; FIXME: Likewise, foo should not be exported, along with referencing function
; func1. However, this relies on being able to add a dependence from
; libAnalysis to libObject, which is currently blocked (see revert of
; r286297).
; NM1-NOT: b
; NM1-NOT: x
; NM1-DAG: U func2
; NM1-DAG: U func3
; NM1-DAG: T main
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

declare i16 @foo() #0

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
