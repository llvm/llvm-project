; RUN: llc -O0 --show-mc-encoding < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; CHECK-LABEL: f1:
; CHECK: addq %{{.*}}, ({{.*}}){{.*}}encoding: [0xf0,0x48,0x01,0x37]
; CHECK: ret
define void @f1(i64* %a, i64 %b) nounwind {
  %1 = atomicrmw add i64* %a, i64 %b monotonic
  ret void
}

; CHECK-LABEL: f2:
; CHECK: subq %{{.*}}, ({{.*}}){{.*}}encoding: [0xf0,0x48,0x29,0x37]
; CHECK: ret
define void @f2(i64* %a, i64 %b) nounwind {
  %1 = atomicrmw sub i64* %a, i64 %b monotonic
  ret void
}

; CHECK-LABEL: f3:
; CHECK: andq %{{.*}}, ({{.*}}){{.*}}encoding: [0xf0,0x48,0x21,0x37]
; CHECK: ret
define void @f3(i64* %a, i64 %b) nounwind {
  %1 = atomicrmw and i64* %a, i64 %b monotonic
  ret void
}

; CHECK-LABEL: f4:
; CHECK: orq %{{.*}}, ({{.*}}){{.*}}encoding: [0xf0,0x48,0x09,0x37]
; CHECK: ret
define void @f4(i64* %a, i64 %b) nounwind {
  %1 = atomicrmw or i64* %a, i64 %b monotonic
  ret void
}

; CHECK-LABEL: f5:
; CHECK: xorq %{{.*}}, ({{.*}}){{.*}}encoding: [0xf0,0x48,0x31,0x37]
; CHECK: ret
define void @f5(i64* %a, i64 %b) nounwind {
  %1 = atomicrmw xor i64* %a, i64 %b monotonic
  ret void
}
