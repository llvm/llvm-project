; RUN: not llc < %s -mtriple=bpfel 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: llc < %s -mtriple=bpfel -bpf-allows-libcalls | FileCheck %s --check-prefix=LIBCALL

; `uitofp i128` lowers to the non-whitelisted libcall `__floatuntidf`.
define dso_local double @accept_non_whitelisted_libcall(i128 %x) local_unnamed_addr {
entry:
  %conv = uitofp i128 %x to double
  ret double %conv
}

; ERR: A call to built-in function '__floatuntidf' is not supported.

; LIBCALL-LABEL: accept_non_whitelisted_libcall:
; LIBCALL:       call __floatuntidf
