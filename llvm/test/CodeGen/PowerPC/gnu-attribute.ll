; RUN: sed -e 's/FLTABI/ieeequad/' %s | llc -mtriple=powerpc64le | FileCheck %s --check-prefix=IEEE
; RUN: sed -e 's/FLTABI/doubledouble/' %s | llc -mtriple=powerpc64le | FileCheck %s --check-prefix=IBM
; RUN: sed -e 's/FLTABI/ieeedouble/' %s | llc -mtriple=powerpc64le | FileCheck %s --check-prefix=DBL
; RUN: sed -e 's/FLTABI//' %s | llc -mtriple=powerpc64le | FileCheck %s --check-prefix=NONE

; IEEE: .gnu_attribute 4, 13
; IBM: .gnu_attribute 4, 5
; DBL: .gnu_attribute 4, 9
; NONE-NOT: .gnu_attribute 4,

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{i32 1, !"float-abi", !"FLTABI"}
