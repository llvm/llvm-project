; Require 64 bits here as the hash will change depending upon whether we are on a 32-bit
; or 64-bit platform.
; REQUIRE: llvm-64-bits

; RUN: opt -passes='print<structural-hash><detailed>' -disable-output %s 2>&1 | FileCheck %s

define i64 @f1(i64 %a) {
	ret i64 %a
}

; These values here are explicitly defined to ensure that they are deterministic
; on all 64-bit platforms and across runs.

; CHECK: Module Hash: 81f1328ced269bd
; CHECK: Function f1 Hash: 81f1328ced269bd

