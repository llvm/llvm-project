; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck --check-prefixes CHECK,CHECK32 %s
; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 | FileCheck --check-prefixes CHECK,CHECK64  %s

@a = external global i32, align 4
@b = external global i64, align 8
@c = external global i16, align 2
@globa = common global i32 0, align 4

@ptr = internal global void (...)* null, align 4

declare void @foo()

define void @bar() {
  %1 = alloca i8*, align 8
  store i32 0, i32* @a, align 4
  store i64 0, i64* @b, align 8
  store i16 0, i16* @c, align 2
  store i32 0, i32* @globa, align 4
  store void (...)* bitcast (void ()* @bar to void (...)*), void (...)** @ptr, align 4
  store i8* bitcast (void ()* @foo to i8*), i8** %1, align 8
  ret void
}

; CHECK-NOT: .comm a
; CHECK-NOT: .lcomm a
; CHECK-NOT: .comm b
; CHECK-NOT: .lcomm b
; CHECK-NOT: .comm c
; CHECK-NOT: .lcomm c
; CHECK: .comm globa[RW],4,2
; CHECK32: .lcomm ptr,4,ptr[BS],2
; CHECK64: .lcomm ptr,8,ptr[BS],2
; CHECK:      .toc
; CHECK-NEXT: LC0:
; CHECK-NEXT: .tc   a[TC],a[UA]
; CHECK-NEXT: LC1:
; CHECK-NEXT: .tc   b[TC],b[UA]
; CHECK-NEXT: LC2:
; CHECK-NEXT: .tc   c[TC],c[UA]
; CHECK-NEXT: LC3:
; CHECK-NEXT: .tc   globa[TC],globa[RW]
; CHECK-NEXT: LC4:
; CHECK-NEXT: .tc   ptr[TC],ptr[BS]
; CHECK-NEXT: LC5:
; CHECK-NEXT: .tc   bar[TC],bar[DS]
; CHECK-NEXT: LC6:
; CHECK-NEXT: .tc   foo[TC],foo[DS]

