; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define noundef float @ifelse(i1 noundef zeroext %x) unnamed_addr {
start:
; CHECK: %.= uitofp i1 %x to float
  %. = select i1 %x, float 1.000000e+00, float 0.000000e+00
  ret float %.
}