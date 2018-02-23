; RUN: llvm-as -disable-verify %s -o %t.bc
; ---- Full LTO ---------------------------------------------
; RUN: not llvm-lto -lto-strip-invalid-debug-info=false \
; RUN:     -o %t.o %t.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty -check-prefix=CHECK-ERR
; RUN: llvm-lto -lto-strip-invalid-debug-info=true \
; RUN:     -exported-symbol foo -exported-symbol _foo \
; RUN:     -o %t.o %t.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty -check-prefix=CHECK-WARN
; RUN: llvm-nm %t.o | FileCheck %s 
; ---- Thin LTO (codegen only) ------------------------------
; RUN: not llvm-lto -thinlto -thinlto-action=codegen \
; RUN:     -lto-strip-invalid-debug-info=false \
; RUN:     %t.bc -disable-verify 2>&1 | \
; RUN:     FileCheck %s -allow-empty -check-prefix=CHECK-ERR
; RUN: llvm-lto -thinlto -thinlto-action=codegen \
; RUN:     -lto-strip-invalid-debug-info=true \
; RUN:     %t.bc -disable-verify 2>&1 | \
; RUN:     FileCheck %s -allow-empty -check-prefix=CHECK-WARN
; ---- Thin LTO (optimize, strip main file) -----------------
; RUN: opt -disable-verify -module-summary %s -o %t.bc
; RUN: opt -disable-verify -module-summary %S/Inputs/strip-debug-info-bar.ll \
; RUN:     -o %t2.bc
; RUN: not llvm-lto -thinlto -thinlto-action=run \
; RUN:     -lto-strip-invalid-debug-info=false \
; RUN:     %t.bc -disable-verify 2>&1 | \
; RUN:     FileCheck %s -allow-empty -check-prefix=CHECK-ERR
; RUN: llvm-lto -thinlto -thinlto-action=run \
; RUN:     -lto-strip-invalid-debug-info=true \
; RUN:     %t.bc -disable-verify 2>&1 | \
; RUN:     FileCheck %s -allow-empty -check-prefix=CHECK-WARN
; ---- Thin LTO (optimize, strip imported file) -------------
; RUN: opt -disable-verify -strip-debug -module-summary %t.bc -o %t-stripped.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %t-stripped.bc %t2.bc
; RUN: not llvm-lto -thinlto -thinlto-action=import \
; RUN:     -thinlto-index=%t.index.bc \
; RUN:     -lto-strip-invalid-debug-info=false \
; RUN:     -exported-symbol foo -exported-symbol _foo \
; RUN:     %t-stripped.bc -disable-verify 2>&1 | \
; RUN:     FileCheck %s -allow-empty -check-prefix=CHECK-ERR
; RUN: llvm-lto -thinlto -thinlto-action=import \
; RUN:     -lto-strip-invalid-debug-info=true \
; RUN:     -thinlto-index=%t.index.bc \
; RUN:     -exported-symbol foo -exported-symbol _foo \
; RUN:     %t-stripped.bc -disable-verify 2>&1 | \
; RUN:     FileCheck %s -allow-empty -check-prefix=CHECK-WARN

; CHECK-ERR: Broken module found, compilation aborted
; CHECK-WARN: Invalid debug info found, debug info will be stripped
; CHECK-WARN-NOT: Broken module found
; CHECK: foo
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12"

declare void @bar()

define void @foo() {
  call void @bar()
  ret void
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DIFile(filename: "broken", directory: "")
