; Avoid `!DL->isLittleEndian() && !CLI->enableBigEndian()` missmatch on PPC64BE.
; REQUIRES: target-byteorder-little-endian

; -global-isel=1 is unsupported.
; XFAIL: target=loongarch{{.*}}
; XFAIL: target=nvptx{{.*}}
; XFAIL: target=sparc{{.*}}
; XFAIL: target=hexagon-{{.*}}

; RUN: llc < %s -O3 -global-isel=0 -fast-isel=0
; RUN: llc < %s -O3 -global-isel=1 -fast-isel=0
; RUN: llc < %s -O3 -global-isel=0 -fast-isel=1

; RUN: llc < %s -O0 -global-isel=0 -fast-isel=0
; RUN: llc < %s -O0 -global-isel=1 -fast-isel=0
; RUN: llc < %s -O0 -global-isel=0 -fast-isel=1

define i1 @test_runtime() local_unnamed_addr {
entry:
  %allow = call i1 @llvm.allow.runtime.check(metadata !"test_check")
  ret i1 %allow
}

declare i1 @llvm.allow.runtime.check(metadata) nounwind

define i1 @test_ubsan() local_unnamed_addr {
entry:
  %allow = call i1 @llvm.allow.ubsan.check(i8 7)
  ret i1 %allow
}

declare i1 @llvm.allow.ubsan.check(i8) nounwind
