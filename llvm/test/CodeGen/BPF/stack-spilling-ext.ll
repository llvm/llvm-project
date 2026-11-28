; RUN: llc -mtriple=bpfel -mcpu=v1 -bpf-stack-spilling -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,SHIFT
; RUN: llc -mtriple=bpfel -mcpu=v3 -bpf-stack-spilling -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,SHIFT
; RUN: llc -mtriple=bpfeb -mcpu=v3 -bpf-stack-spilling -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,SHIFT
; RUN: llc -mtriple=bpfel -mcpu=v4 -bpf-stack-spilling -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,V4

; Narrow stack arguments retain the same 64-bit slot extension as in the
; default ABI, including when ALU32 is enabled.
declare i64 @ext(i64, i64, i64, i64, i64, i32 zeroext, i32 signext)

define i64 @caller(i64 %v) {
; CHECK-LABEL: caller:
; CHECK-DAG: r[[Z:[0-9]+]] >>= 32
; CHECK-DAG: *(u64 *)(r10 + 0) = r[[Z]]
; SHIFT-DAG: r[[S:[0-9]+]] s>>= 32
; V4-DAG: r[[S:[0-9]+]] = (s32)r{{[0-9]+}}
; CHECK-DAG: *(u64 *)(r10 + 8) = r[[S]]
; CHECK: call ext
  %t = trunc i64 %v to i32
  %r = call i64 @ext(i64 1, i64 2, i64 3, i64 4, i64 5, i32 zeroext %t, i32 signext %t)
  ret i64 %r
}

define i64 @callee(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i32 zeroext %f, i32 signext %g) {
; CHECK-LABEL: callee:
; CHECK-DAG: r[[Z:[0-9]+]] = *(u64 *)(r10 - 4096)
; CHECK-DAG: r[[S:[0-9]+]] = *(u64 *)(r10 - 4088)
; CHECK: r[[Z]] ^= r[[S]]
; CHECK: exit
  %x = zext i32 %f to i64
  %y = sext i32 %g to i64
  %r = xor i64 %x, %y
  ret i64 %r
}

; Small aggregate elements also occupy separate 8-byte slots.
define i64 @small_aggregate(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, [2 x i8] %pair) {
; CHECK-LABEL: small_aggregate:
; CHECK-DAG: r{{[0-9]+}} = *(u64 *)(r10 - 4096)
; CHECK-DAG: r{{[0-9]+}} = *(u64 *)(r10 - 4088)
; CHECK: exit
  %lo = extractvalue [2 x i8] %pair, 0
  %hi = extractvalue [2 x i8] %pair, 1
  %x = zext i8 %lo to i64
  %y = zext i8 %hi to i64
  %r = add i64 %x, %y
  ret i64 %r
}
