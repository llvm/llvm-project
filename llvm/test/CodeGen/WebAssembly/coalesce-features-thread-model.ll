; RUN: split-file %s %t

; Test that wasm-coalesce-features-and-strip-atomics sets the "thread-model"
; module flag to "posix" when threading features (+atomics or cooperative
; multithreading with +bulk-memory) are enabled and atomics/TLS were not
; stripped, unless an explicit "thread-model" flag is already present.

; RUN: opt -mtriple=wasm32-unknown-unknown -passes=wasm-coalesce-features-and-strip-atomics -S %t/default.ll | FileCheck %s --check-prefix=NO-FLAG
; RUN: opt -mtriple=wasm32-unknown-unknown -mattr=+atomics -passes=wasm-coalesce-features-and-strip-atomics -S %t/default.ll | FileCheck %s --check-prefix=POSIX
; RUN: opt -mtriple=wasm32-wasip3 -mattr=+bulk-memory -passes=wasm-coalesce-features-and-strip-atomics -S %t/default.ll | FileCheck %s --check-prefix=POSIX
; RUN: opt -mtriple=wasm32-wasip3 -mattr=-bulk-memory -passes=wasm-coalesce-features-and-strip-atomics -S %t/default.ll | FileCheck %s --check-prefix=NO-FLAG

; RUN: opt -mtriple=wasm32-unknown-unknown -passes=wasm-coalesce-features-and-strip-atomics -S %t/attr.ll | FileCheck %s --check-prefix=POSIX

; RUN: opt -mtriple=wasm32-unknown-unknown -mattr=+atomics,-bulk-memory -passes=wasm-coalesce-features-and-strip-atomics -S %t/tls.ll | FileCheck %s --check-prefix=NO-FLAG
; RUN: opt -mtriple=wasm32-unknown-unknown -mattr=+atomics,+bulk-memory -passes=wasm-coalesce-features-and-strip-atomics -S %t/tls.ll | FileCheck %s --check-prefix=POSIX

; RUN: opt -mtriple=wasm32-unknown-unknown -mattr=+atomics -passes=wasm-coalesce-features-and-strip-atomics -S %t/explicit.ll | FileCheck %s --check-prefix=EXPLICIT-SINGLE

; NO-FLAG-NOT: !"thread-model"
; POSIX: !{i32 1, !"thread-model", !"posix"}
; EXPLICIT-SINGLE: !{i32 1, !"thread-model", !"single"}
; EXPLICIT-SINGLE-NOT: !"posix"

;--- default.ll
define void @foo() {
  ret void
}

;--- attr.ll
define void @foo() "target-features"="+atomics" {
  ret void
}

;--- tls.ll
@tls = thread_local global i32 0

;--- explicit.ll
define void @foo() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"thread-model", !"single"}
