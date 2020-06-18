; REQUIRES: asserts
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -verify-misched -debug-only=machine-scheduler \
; RUN:  -o - 2>&1 > /dev/null | FileCheck %s --check-prefix=CHECK-P8

@m = local_unnamed_addr global i64 0, align 8

define i64 @fuse_addis_ld() {
entry:
; CHECK-P8: ********** MI Scheduling **********
; CHECK-P8-LABEL: fuse_addis_ld:%bb.0 entry
; CHECK-P8: Macro fuse: SU([[SU0:[0-9]+]]) - SU([[SU1:[0-9]+]]) /  ADDIStocHA8 - LD
; CHECK-P8: SU([[SU0]]):   %[[REG3:[0-9]+]]:g8rc_and_g8rc_nox0 = ADDIStocHA8 $x2, @m
; CHECK-P8: SU([[SU1]]):   %{{[0-9]+}}:g8rc = LD target-flags(ppc-toc-lo) @m, %[[REG3]]
; CHECK-P8: ********** MI Scheduling **********
; CHECK-P8-LABEL: fuse_addis_ld:%bb.0 entry
; CHECK-P8: Macro fuse: SU([[SU0:[0-9]+]]) - SU([[SU1:[0-9]+]]) /  ADDIStocHA8 - LD
; CHECK-P8: SU([[SU0]]):   renamable $x[[REG3:[0-9]+]] = ADDIStocHA8 $x2, @m
; CHECK-P8: SU([[SU1]]):   renamable $x[[REG3]] = LD target-flags(ppc-toc-lo) @m, renamable $x[[REG3]]
  %0 = load i64, i64* @m, align 8
  ret i64 %0
}
