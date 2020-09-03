; REQUIRES: powerpc-registered-target
; RUN: llc -mtriple=powerpc-ibm-aix < %s 2>&1 >/dev/null | FileCheck --allow-empty %s
; RUN: llc -mtriple=powerpc-ibm-aix --relocation-model=pic < %s 2>&1 >/dev/null | FileCheck --allow-empty %s
; RUN: llc -mtriple=powerpc64-ibm-aix --relocation-model=pic < %s 2>&1 >/dev/null | FileCheck --allow-empty %s
; RUN: not llc -mtriple=powerpc-ibm-aix --relocation-model=static < %s 2>&1 | FileCheck --check-prefix=CHECK-NON-PIC %s
; RUN: not llc -mtriple=powerpc64-ibm-aix --relocation-model=ropi-rwpi < %s 2>&1 | FileCheck --check-prefix=CHECK-NON-PIC %s

; CHECK-NOT: {{.}}
; CHECK-NON-PIC: invalid relocation model, AIX only supports PIC.
