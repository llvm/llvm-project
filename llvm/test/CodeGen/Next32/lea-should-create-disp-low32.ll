; RUN: llc -mtriple=next32 -mattr=+lea < %s | FileCheck %s --check-prefix=HAS-LEA
; RUN: llc -mtriple=next32 -mattr=-lea < %s | FileCheck %s --check-prefix=NO-LEA

; Original C source:
; #include <stdint.h>
;
; int64_t test(int64_t a) {
;     return a+0x200000000;
; }

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define dso_local i64 @test(i64 noundef %0) {
; HAS-LEA-LABEL: test:
; HAS-LEA: LBB0_0:
; HAS-LEA: # %bb.0:
; HAS-LEA: feeder.32 tid
; HAS-LEA: feeder.32 ret_fid
; HAS-LEA: feeder.64 r1
; HAS-LEA: feeder.64 r2
; HAS-LEA: movl    r3, 0x2
; HAS-LEA: add     r2, r3
; HAS-LEA: chain   ret_fid, 0x64
; HAS-LEA: writer.32 ret_fid, tid
; HAS-LEA: writer.64 ret_fid, r1
; HAS-LEA: writer.64 ret_fid, r2
;
; NO-LEA-LABEL: test:
; NO-LEA: LBB0_0:
; NO-LEA: # %bb.0:
; NO-LEA: feeder.32 tid
; NO-LEA: feeder.32 ret_fid
; NO-LEA: feeder.64 r1
; NO-LEA: feeder.64 r2
; NO-LEA: movl    r3, 0x2
; NO-LEA: add     r2, r3
; NO-LEA: chain   ret_fid, 0x64
; NO-LEA: writer.32 ret_fid, tid
; NO-LEA: writer.64 ret_fid, r1
; NO-LEA: writer.64 ret_fid, r2
  %2 = add nsw i64 %0, 8589934592
  ret i64 %2
}

