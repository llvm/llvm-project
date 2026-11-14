; RUN: llc -mtriple=s390x-ibm-zos -filetype=obj -o %t %s
; RUN: llvm-nm %t | FileCheck %s
; RUN: llvm-nm %t | FileCheck %s --check-prefix=NODEFAULT
; RUN: llc -mtriple=s390x-ibm-zos < %s | FileCheck %s --check-prefix=HLASM

define i32 @foo() {
entry:
  ret i32 0
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"zos_csect_code", !"CSECTCODE"}
!1 = !{i32 1, !"zos_csect_static", !"CSECTSTATIC"}

; CHECK: T CSECTCODE
; CHECK: D CSECTSTATIC

; NODEFAULT-NOT: csect-custom#C
; NODEFAULT-NOT: csect-custom#S

; HLASM: CSECTCODE XATTR LINKAGE(XPLINK),PSECT(CSECTSTATIC),SCOPE(LIBRARY)
; HLASM: CSECTSTATIC XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(LIBRARY)
