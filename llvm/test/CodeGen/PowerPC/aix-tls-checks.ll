; RUN: not --crash llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec \
; RUN:   -mtriple powerpc64-ibm-aix-xcoff < %s - 2>&1 | FileCheck %s

; CHECK: TLS is not yet supported on AIX PPC64

@tls1 = thread_local global i32 0, align 4

define i32* @getTls1Addr() {
entry:
  ret i32* @tls1
}
