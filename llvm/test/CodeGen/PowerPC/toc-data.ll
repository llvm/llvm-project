; REQUIRES: asserts
; RUN: llc -mtriple powerpc-ibm-aix-xcoff -verify-machineinstrs < %s \
; RUN:     -stop-before=ppc-ctr-loops-verify | FileCheck %s
; RUN: llc -mtriple powerpc-ibm-aix-xcoff -verify-machineinstrs < %s | FileCheck %s --check-prefix TEST

@i = dso_local global i32 0, align 4 #0
@d = dso_local local_unnamed_addr global double 3.141590e+00, align 8
@f = dso_local local_unnamed_addr global float 0x4005BE76C0000000, align 4 #0
@ll = dso_local local_unnamed_addr global i64 55, align 8
@ilocal = internal global i32 0, align 4

define dso_local void @write_int(i32 signext %in) {
  entry:
    store i32 %in, i32* @i, align 4
    ret void
}
; CHECK: name:            write_int
; CHECK:      %[[SCRATCH:[0-9]+]]:gprc_and_gprc_nor0 = ADDItoc @i, $r2
; CHECK-NEXT: STW %{{[0-9]+}}, 0, killed %[[SCRATCH]] :: (store (s32) into @i)

; TEST:         .write_int:
; TEST:           la 4, i[TD](2)
; TEST-NEXT:      stw 3, 0(4)

define dso_local i64 @read_ll() {
  entry:
    %0 = load i64, i64* @ll, align 8
    ret i64 %0
}
; CHECK: name:            read_ll
; CHECK: LWZtoc @ll, $r2 :: (load (s32) from got)

; TEST:       .read_ll:
; TEST:         lwz 4, L..C0(2)
; TEST-NEXT:    lwz 3, 0(4)
; TEST-NEXT:    lwz 4, 4(4)

define dso_local float @read_float() {
  entry:
    %0 = load float, float* @f, align 4
    ret float %0
}
; CHECK: name:            read_float
; CHECK: %[[SCRATCH:[0-9]+]]:gprc_and_gprc_nor0 = ADDItoc @f, $r2
; CHECK: %{{[0-9]+}}:f4rc = LFS 0, killed %[[SCRATCH]] :: (dereferenceable load (s32) from @f)

; TEST:       .read_float:
; TEST:         la 3, f[TD](2)
; TEST-NEXT:    lfs 1, 0(3)

define dso_local void @write_double(double %in) {
  entry:
    store double %in, double* @d, align 8
    ret void
}
; CHECK: name:            write_double
; CHECK: LWZtoc @d, $r2 :: (load (s32) from got)

; TEST:       .write_double
; TEST:         lwz 3, L..C1(2)
; TEST-NEXT:    stfd 1, 0(3)

define dso_local nonnull i32* @addr() {
  entry:
    ret i32* @i
}
; CHECK: name:            addr
; CHECK:       %[[SCRATCH:[0-9]+]]:gprc = ADDItoc @i, $r2
; CHECK-NEXT:  $r3 = COPY %[[SCRATCH]]

; TEST:       .addr
; TEST:         la 3, i[TD](2)


attributes #0 = { "toc-data" }
