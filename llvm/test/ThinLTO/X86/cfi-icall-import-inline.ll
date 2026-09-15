; Check that a function imported by ThinLTO which is also a CFI jump table
; member can still be inlined in the importing module. LowerTypeTests must keep
; direct calls pointing at the imported body (renamed to hot.cfi) rather than
; redirecting them to a declaration of the real function.

; REQUIRES: x86-registered-target

; RUN: rm -rf %t.dir && split-file %s %t.dir
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %t.dir/a.ll -o %t.dir/a.bc
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %t.dir/b.ll -o %t.dir/b.bc
; RUN: llvm-lto2 run -save-temps %t.dir/a.bc %t.dir/b.bc -o %t.dir/out \
; RUN:   -r=%t.dir/a.bc,hot,plx \
; RUN:   -r=%t.dir/a.bc,indirect,plx \
; RUN:   -r=%t.dir/a.bc,get,plx \
; RUN:   -r=%t.dir/b.bc,hot,l \
; RUN:   -r=%t.dir/b.bc,caller,plx
; RUN: llvm-dis %t.dir/out.2.3.import.bc -o - | FileCheck %s --check-prefix=IMPORT
; RUN: llvm-dis %t.dir/out.2.4.opt.bc -o - | FileCheck %s --check-prefix=OPT
; RUN: llvm-dis %t.dir/out.0.4.opt.bc -o - | FileCheck %s --check-prefix=JT

; @hot is imported into b.ll as an available_externally definition.
; IMPORT: define available_externally hidden i32 @hot(i32 %x)

; After optimization the imported body has been inlined into @caller.
; OPT-LABEL: define hidden {{.*}}i32 @caller(i32
; OPT-NOT: call
; OPT: mul {{.*}}i32 %x, 3
; OPT-NEXT: ret i32

; In the regular-LTO module, @hot aliases the jump table trampoline to @hot.cfi.
; JT-DAG: @hot = {{.*}}alias {{.*}}ptr @.cfi.jumptable
; JT-DAG: @__typeid__ZTSFiiE_global_addr = {{.*}}alias {{.*}}ptr @.cfi.jumptable
; JT-DAG: declare {{.*}}void @hot.cfi()
; JT: define private void @.cfi.jumptable()
; JT: call void asm sideeffect "jmp {{.*}}@plt{{.*}}"{{.*}}(ptr @hot.cfi)

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define hidden i32 @hot(i32 %x) !type !0 {
  %r = mul i32 %x, 3
  ret i32 %r
}

; An indirect call through the type of @hot puts @hot into a jump table.
define hidden i32 @indirect(ptr %f, i32 %x) {
  %t = call i1 @llvm.type.test(ptr %f, metadata !"_ZTSFiiE")
  br i1 %t, label %cont, label %trap

trap:
  call void @llvm.ubsantrap(i8 2)
  unreachable

cont:
  %r = call i32 %f(i32 %x)
  ret i32 %r
}

define hidden ptr @get() {
  ret ptr @hot
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.ubsantrap(i8)

!0 = !{i64 0, !"_ZTSFiiE"}

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare hidden i32 @hot(i32)

define hidden i32 @caller(i32 %x) {
  %r = call i32 @hot(i32 %x)
  ret i32 %r
}
