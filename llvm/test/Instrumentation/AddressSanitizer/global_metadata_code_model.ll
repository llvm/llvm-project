; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes=asan -S | FileCheck %s --check-prefix=LARGE
; RUN: opt < %s -mtriple=aarch64-unknown-linux-gnu -passes=asan -S | FileCheck %s --check-prefix=NORMAL
; RUN: opt < %s -mtriple=x86_64-pc-windows -passes=asan -S | FileCheck %s --check-prefix=NORMAL

; check that asan globals metadata are emitted to a large section for x86-64 ELF

; LARGE: @__asan_global_global = {{.*}}global {{.*}}, code_model "large"
; NORMAL-NOT: code_model "large"

@global = global i32 0, align 4
