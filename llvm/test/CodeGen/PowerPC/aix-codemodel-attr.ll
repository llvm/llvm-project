; RUN: llc --verify-machineinstrs -mtriple powerpc-ibm-aix --code-model=small < \
; RUN: %s | FileCheck --check-prefixes=CHECK,CHECK32,CHECK-SMALL,CHECK-SMALL32 %s

; RUN: llc --verify-machineinstrs -mtriple powerpc-ibm-aix --code-model=large < \
; RUN: %s | FileCheck --check-prefixes=CHECK,CHECK32,CHECK-LARGE,CHECK-LARGE32 %s

; RUN: llc --verify-machineinstrs -mtriple powerpc64-ibm-aix --code-model=small < \
; RUN: %s | FileCheck --check-prefixes=CHECK,CHECK64,CHECK-SMALL,CHECK-SMALL64 %s

; RUN: llc --verify-machineinstrs -mtriple powerpc64-ibm-aix --code-model=large < \
; RUN: %s | FileCheck --check-prefixes=CHECK,CHECK64,CHECK-LARGE,CHECK-LARGE64 %s

@a = external dso_local global i32, code_model "small", align 4
@b = external dso_local global i32, code_model "large", align 4
@c = dso_local global i32 55, code_model "small", align 4
@d = dso_local global i32 41, code_model "large", align 4
@e = external dso_local global i32, align 4
@f = dso_local global i32 2748, align 4

@large_aliasee = global i32 10, code_model "large", align 4
@small_aliasee = global i32 171, code_model "small", align 4
@normal_aliasee = global i32 2748, align 4

@al = alias i32, ptr @large_aliasee
@as = alias i32, ptr @small_aliasee
@an = alias i32, ptr @normal_aliasee

define i32 @A() local_unnamed_addr {
entry:
  %0 = load i32, ptr @a, align 4
  ret i32 %0
}
; CHECK32:  lwz [[SCRATCH:[0-9]+]], L..C[[TL_A:[0-9]+]](2)                         # @a
; CHECK64:  ld [[SCRATCH:[0-9]+]], L..C[[TL_A:[0-9]+]](2)                         # @a
; CHECK:    lwz 3, 0([[SCRATCH]])
; CHECK:    blr

define i32 @B() local_unnamed_addr {
entry:
  %0 = load i32, ptr @b, align 4
  ret i32 %0
}
; CHECK:   addis [[HI:[0-9]+]], L..C[[TL_B:[0-9]+]]@u(2)
; CHECK32: lwz [[ADDR:[0-9]+]], L..C[[TL_B]]@l([[HI]])
; CHECK64: ld [[ADDR:[0-9]+]], L..C[[TL_B]]@l([[HI]])
; CHECK:   lwz 3, 0([[ADDR]])
; CHECK:   blr

define i32 @C() local_unnamed_addr {
entry:
  %0 = load i32, ptr @c, align 4
  ret i32 %0
}
; CHECK32:  lwz [[SCRATCH:[0-9]+]], L..C[[TL_C:[0-9]+]](2)                         # @c
; CHECK64:  ld [[SCRATCH:[0-9]+]], L..C[[TL_C:[0-9]+]](2)                         # @c
; CHECK:    lwz 3, 0([[SCRATCH]])
; CHECK:    blr

define i32 @D() local_unnamed_addr {
entry:
  %0 = load i32, ptr @d, align 4
  ret i32 %0
}
; CHECK: addis [[HI:[0-9]+]], L..C[[TL_D:[0-9]+]]@u(2)
; CHECK32: lwz [[ADDR:[0-9]+]], L..C[[TL_D]]@l([[HI]])
; CHECK64: ld [[ADDR:[0-9]+]], L..C[[TL_D]]@l([[HI]])
; CHECK: lwz 3, 0([[ADDR]])
; CHECK: blr

define i32 @E() {
entry:
  %0 = load i32, ptr @e, align 4
  ret i32 %0
}
; CHECK-LARGE: addis [[HI:[0-9]+]], L..C[[TL_E:[0-9]+]]@u(2)
; CHECK-LARGE32: lwz [[SCRATCH:[0-9]+]], L..C[[TL_E]]@l([[HI]])
; CHECK-SMALL32: lwz [[SCRATCH:[0-9]+]], L..C[[TL_E:[0-9]+]](2)
; CHECK-LARGE64: ld  [[SCRATCH:[0-9]+]], L..C[[TL_E]]@l([[HI]])
; CHECK-SMALL64: ld  [[SCRATCH:[0-9]+]], L..C[[TL_E:[0-9]+]](2)
; CHECK: lwz 3, 0([[SCRATCH]])
; CHECK: blr

define i32 @F() {
entry:
  %0 = load i32, ptr @f, align 4
  ret i32 %0
}
; CHECK-LARGE: addis [[HI:[0-9]+]], L..C[[TL_F:[0-9]+]]@u(2)
; CHECK-LARGE32: lwz [[SCRATCH:[0-9]+]], L..C[[TL_F]]@l([[HI]])
; CHECK-SMALL32: lwz [[SCRATCH:[0-9]+]], L..C[[TL_F:[0-9]+]](2)
; CHECK-LARGE64: ld [[SCRATCH:[0-9]+]], L..C[[TL_F]]@l([[HI]])
; CHECK-SMALL64: ld  [[SCRATCH:[0-9]+]], L..C[[TL_F:[0-9]+]](2)
; CHECK: lwz 3, 0([[SCRATCH]])
; CHECK: blr

define noundef nonnull ptr @addr_a() local_unnamed_addr {
entry:
  ret ptr @a
}
; CHECK32:  lwz 3, L..C[[TL_A]](2)                         # @a
; CHECK64:  ld 3, L..C[[TL_A]](2)                         # @a
; CHECK:    blr

define noundef nonnull ptr @addr_b() local_unnamed_addr {
entry:
  ret ptr @b
}
; CHECK:    addis [[HI:[0-9]+]], L..C[[TL_B]]@u(2)
; CHECK32:  lwz 3, L..C[[TL_B]]@l([[HI]])
; CHECK64:  ld 3, L..C[[TL_B]]@l([[HI]])
; CHECK:    blr


define noundef nonnull ptr @addr_c() local_unnamed_addr {
entry:
  ret ptr @c
}
; CHECK32:  lwz 3, L..C[[TL_C]](2)                         # @c
; CHECK64:  ld 3, L..C[[TL_C]](2)                         # @c
; CHECK:    blr

define noundef nonnull ptr @addr_d() local_unnamed_addr {
entry:
  ret ptr @d
}
; CHECK:   addis [[HI:[0-9]+]], L..C[[TL_D]]@u(2)
; CHECK32: lwz 3, L..C[[TL_D]]@l([[HI]])
; CHECK64: ld 3, L..C[[TL_D]]@l([[HI]])
; CHECK:   blr

define i32 @G() {
   %tmp = load i32, ptr @al
   ret i32 %tmp
}
; CHECK:   addis [[HI:[0-9]+]], L..C[[TL_AL:[0-9]+]]@u(2)
; CHECK32: lwz [[ADDR:[0-9]+]], L..C[[TL_AL]]@l([[HI]])
; CHECK64: ld  [[ADDR:[0-9]+]], L..C[[TL_AL]]@l([[HI]])
; CHECK:   lwz 3, 0([[ADDR]])

define i32 @H() {
   %tmp = load i32, ptr @as
   ret i32 %tmp
}
; CHECK32: lwz [[ADDR:[0-9]+]], L..C[[TL_AS:[0-9]+]](2)
; CHECK64: ld [[ADDR:[0-9]+]], L..C[[TL_AS:[0-9]+]](2)
; CHECK:   lwz 3, 0([[ADDR]])

;; Check TOC entires have correct storage mapping class
; CHECK:         L..C[[TL_A]]:
; CHECK:           .tc a[TC],a[UA]
; CHECK:         L..C[[TL_B]]:
; CHECK:           .tc b[TE],b[UA]
; CHECK:         L..C[[TL_C]]:
; CHECK:           .tc c[TC],c[RW]
; CHECK:         L..C[[TL_D]]:
; CHECK:           .tc d[TE],d[RW]
; CHECK:         L..C[[TL_E]]:
; CHECK-SMALL:     .tc e[TC],e[UA]
; CHECK-LARGE:     .tc e[TE],e[UA]
; CHECK:         L..C[[TL_F]]:
; CHECK-SMALL:     .tc f[TC],f[RW]
; CHECK-LARGE:     .tc f[TE],f[RW]
; CHECK:         L..C[[TL_AL]]:
; CHECK:           .tc al[TE],al
; CHECK:         L..C[[TL_AS]]:
; CHECK:           .tc as[TC],as
