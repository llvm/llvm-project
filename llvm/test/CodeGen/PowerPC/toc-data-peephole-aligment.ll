; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -verify-machineinstrs < %s | FileCheck %s

@underaligned = dso_local global i32 123, align 1 #0

define i64 @read() {
entry:
  %0  = load i32, ptr @underaligned, align 1
  %1 = sext i32 %0 to i64
  ret i64 %1
}

attributes #0 = { "toc-data"  }

; CHECK-LABEL: .read
; CHECK:       la [[DEF:[0-9]+]], underaligned[TD](2)
; CHCEK:       lwa {{[0-9]+}}, 0([[DEF]])
