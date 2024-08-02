; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux -mcpu=pwr7 \
; RUN:   -O2 -o - < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-64
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux -mcpu=pwr7 \
; RUN:   -O2 -o - < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-64
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-aix -mcpu=pwr7 \
; RUN:   -O2 -o - < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-64
; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-aix -mcpu=pwr7 \
; RUN:   -O2 -o - < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-AIX-32

%union.tu_c = type { i8 }
%union.tu_s = type { i16 }
%union.tu_us = type { i16 }
%union.tu_l = type { i64 }

define void @ftest0(i8 noundef zeroext %uc.coerce) {
; CHECK-LABEL: ftest0:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    stb 3, -1(1)
; CHECK-NEXT:    blr
entry:
  %uc = alloca %union.tu_c, align 1
  %coerce.dive = getelementptr inbounds %union.tu_c, ptr %uc, i32 0, i32 0
  store i8 %uc.coerce, ptr %coerce.dive, align 1
  ret void
}

define void @ftest1(i16 noundef signext %uc.coerce) {
; CHECK-LABEL: ftest1:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    sth 3, -2(1)
; CHECK-NEXT:    blr
entry:
  %uc = alloca %union.tu_s, align 2
  %coerce.dive = getelementptr inbounds %union.tu_s, ptr %uc, i32 0, i32 0
  store i16 %uc.coerce, ptr %coerce.dive, align 2
  ret void
}

define void @ftest2(i16 noundef zeroext %uc.coerce) {
; CHECK-LABEL: ftest2:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    sth 3, -2(1)
; CHECK-NEXT:    blr
entry:
  %uc = alloca %union.tu_us, align 2
  %coerce.dive = getelementptr inbounds %union.tu_us, ptr %uc, i32 0, i32 0
  store i16 %uc.coerce, ptr %coerce.dive, align 2
  ret void
}

define void @ftest3(i64 %uc.coerce) {
; CHECK-64-LABEL: ftest3:
; CHECK-64:       # %bb.0: # %entry
; CHECK-64-NEXT:    std 3, -8(1)
; CHECK-64-NEXT:    blr
;
; CHECK-AIX-32-LABEL: ftest3:
; CHECK-AIX-32:       # %bb.0: # %entry
; CHECK-AIX-32-NEXT:    stw 4, -4(1)
; CHECK-AIX-32-NEXT:    stw 3, -8(1)
; CHECK-AIX-32-NEXT:    blr
entry:
  %uc = alloca %union.tu_l, align 8
  %coerce.dive = getelementptr inbounds %union.tu_l, ptr %uc, i32 0, i32 0
  store i64 %uc.coerce, ptr %coerce.dive, align 8
  ret void
}
