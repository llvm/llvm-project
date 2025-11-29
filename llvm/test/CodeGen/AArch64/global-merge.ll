; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -O0 | FileCheck --check-prefix=NO-MERGE %s
; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -O1 | FileCheck --check-prefix=NO-MERGE %s
; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -O2 | FileCheck --check-prefix=NO-MERGE %s
; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -O3 | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -O3 -global-merge-max-offset=0 | FileCheck %s --check-prefix=NO-MERGE
; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -O0 -global-merge-on-external=true | FileCheck --check-prefix=NO-MERGE %s

; RUN: llc < %s -mtriple=aarch64-apple-ios -O0 | FileCheck %s --check-prefix=CHECK-APPLE-IOS-NO-MERGE
; RUN: llc < %s -mtriple=aarch64-apple-ios -O1 | FileCheck %s --check-prefix=CHECK-APPLE-IOS-NO-MERGE
; RUN: llc < %s -mtriple=aarch64-apple-ios -O2 | FileCheck %s --check-prefix=CHECK-APPLE-IOS-NO-MERGE
; RUN: llc < %s -mtriple=aarch64-apple-ios -O3 | FileCheck %s --check-prefix=CHECK-APPLE-IOS
; RUN: llc < %s -mtriple=aarch64-apple-ios -O0 -global-merge-on-external=true | FileCheck %s --check-prefix=CHECK-APPLE-IOS-NO-MERGE

@m = internal global i32 0, align 4
@n = internal global i32 0, align 4

define void @f1(i32 %a1, i32 %a2) {
; CHECK-LABEL: f1:
; CHECK: adrp x{{[0-9]+}}, .L_MergedGlobals
; CHECK-NOT: adrp

; CHECK-APPLE-IOS-LABEL: f1:
; CHECK-APPLE-IOS: adrp x{{[0-9]+}}, __MergedGlobals
; CHECK-APPLE-IOS-NOT: adrp
  store i32 %a1, ptr @m, align 4
  store i32 %a2, ptr @n, align 4
  ret void
}

; CHECK:        .local .L_MergedGlobals
; CHECK:        .comm  .L_MergedGlobals,8,4
; NO-MERGE-NOT: .local _MergedGlobals

; CHECK-APPLE-IOS: .zerofill __DATA,__bss,__MergedGlobals,8,2
; CHECK-APPLE-IOS-NO-MERGE-NOT: .zerofill __DATA,__bss,__MergedGlobals,8,2
