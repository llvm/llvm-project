; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple powerpc-ibm-aix-xcoff
; RUN:   --xcoff-inline-glue-code=false < %s | FileCheck --check-prefixes=CHECK,CHECK32 %s

; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple powerpc64-ibm-aix-xcoff
; RUN:   --xcoff-inline-glue-code=false < %s | FileCheck --check-prefixes=CHECK,CHECK64 %s

@a = dso_local global i32 55, align 4
@d = dso_local local_unnamed_addr global double 3.141590e+00, align 8
@fp = dso_local local_unnamed_addr global ptr null, align 8

define i32 @caller1(ptr noundef readonly captures(none) %fp) local_unnamed_addr {
entry:
  %call = tail call i32 %fp(i32 signext 1, i32 signext 2, i32 signext 3)
  ret i32 %call
}

; CHECK-LABEL: .caller1
; CHECK-DAG:    mr 11, 3
; CHECK-DAG:    li 3, 1
; CHECK-DAG:    li 4, 2
; CHECK-DAG:    li 5, 3
; CHECK: bl .__ptrgl[PR]A
; CHECK32-NEXT: ld 2  28(r1)
; CHECK64-NEXT: ld 2, 40(r1)

define dso_local zeroext i1 @caller2() local_unnamed_addr {
entry:
  %0 = load ptr, ptr @fp
  %1 = load i32, ptr @a
  %2 = load double, ptr @d
  %call = tail call zeroext i1 %0(i32 noundef signext %1, double noundef %2, ptr noundef nonnull @a)
  ret i1 %call
}

; CHECK-LABEL: .caller2
; CHECK: ld , L..C{{.*}}(2)                          # @fp
; CHECK: ld 11, 0([[REG]])
; CHECK: lwa 3, 0(5)
; CHECK: bl .__ptrgl[PR]
; CHECK32-NEXT: ld 2, 28(r1)
; CHECK64-NEXT: ld 2, 40(r1)
