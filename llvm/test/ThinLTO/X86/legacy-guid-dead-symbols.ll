; RUN: split-file %s %t
; Assemble the summary directly so the caller's definitions have no GUIDs,
; like bitcode from older LLVM versions.
; RUN: llvm-as %t/caller.ll -o %t/caller.bc
; RUN: llvm-dis %t/caller.bc -o - | FileCheck %s --check-prefix=NO-GUID --implicit-check-not=!guid
; RUN: opt -module-summary %t/provider.ll -o %t/provider.bc
; RUN: llvm-lto2 run %t/caller.bc %t/provider.bc -o %t/out -save-temps \
; RUN:   -r=%t/caller.bc,main,plx -r=%t/caller.bc,unused,pl \
; RUN:   -r=%t/caller.bc,missing,l -r=%t/provider.bc,unused, \
; RUN:   -r=%t/provider.bc,missing,pl -r=%t/provider.bc,implementation,pl
; RUN: llvm-dis %t/out.1.1.promote.bc -o - | FileCheck %s --implicit-check-not=@unused

; NO-GUID: define i32 @main()
; CHECK: define{{.*}} i32 @main()

;--- caller.ll
source_filename = "legacy-guid-dead-symbols-caller.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$unused = comdat any

define i32 @main() {
  ret i32 0
}

define linkonce_odr hidden void @unused() comdat {
  call void @missing()
  ret void
}

declare void @missing()

^0 = module: (path: "legacy-guid-dead-symbols-caller", hash: (0, 0, 0, 0, 0))
^1 = gv: (name: "main", summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))
^2 = gv: (name: "unused", summaries: (function: (module: ^0, flags: (linkage: linkonce_odr, visibility: hidden, dsoLocal: 1), insts: 2, calls: ((callee: ^3)))))
^3 = gv: (name: "missing")

;--- provider.ll
target triple = "x86_64-unknown-linux-gnu"

$unused = comdat any

define linkonce_odr hidden void @unused() comdat {
  call void @missing()
  ret void
}

@missing = alias void (), ptr @implementation

define void @implementation() {
  ret void
}
