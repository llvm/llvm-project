; REQUIRES: asserts, abi_breaking_checks
; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:   -verify-machineinstrs -debug 2>&1 | FileCheck %s

define zeroext i1 @addiCmpiUnsigned(i32 zeroext %x) {
entry:
  %add = add nuw i32 10, %x
  %cmp = icmp ugt i32 %add, 100
  ret i1 %cmp

; CHECK: === addiCmpiUnsigned
; CHECK: Optimized lowered selection DAG: %bb.0 'addiCmpiUnsigned:entry'
; CHECK:   [[REG1:t[0-9]+]]: i32 = truncate {{t[0-9]+}}
; CHECK:   [[REG2:t[0-9]+]]: i32 = add nuw [[REG1]], Constant:i32<10>
; CHECK:   {{t[0-9]+}}: i1 = setcc [[REG2]], Constant:i32<100>, setugt:ch
}

define zeroext i1 @addiCmpiSigned(i32 signext %x) {
entry:
  %add = add nsw i32 16, %x
  %cmp = icmp sgt i32 %add, 30
  ret i1 %cmp

; CHECK: === addiCmpiSigned
; CHECK: Optimized lowered selection DAG: %bb.0 'addiCmpiSigned:entry'
; CHECK:   [[REG1:t[0-9]+]]: i32 = truncate {{t[0-9]+}}
; CHECK:   [[REG2:t[0-9]+]]: i32 = add nsw [[REG1]], Constant:i32<16>
; CHECK:   {{t[0-9]+}}: i1 = setcc [[REG2]], Constant:i32<30>, setgt:ch
}

define zeroext i1 @addiCmpiUnsignedOverflow(i32 zeroext %x) {
entry:
  %add = add nuw i32 110, %x
  %cmp = icmp ugt i32 %add, 100
  ret i1 %cmp

; CHECK: === addiCmpiUnsignedOverflow
; CHECK: Optimized lowered selection DAG: %bb.0 'addiCmpiUnsignedOverflow:entry'
; CHECK:   [[REG1:t[0-9]+]]: i32 = truncate {{t[0-9]+}}
; CHECK:   [[REG2:t[0-9]+]]: i32 = add nuw [[REG1]], Constant:i32<110>
; CHECK:   {{t[0-9]+}}: i1 = setcc [[REG2]], Constant:i32<100>, setugt:ch
}

define zeroext i1 @addiCmpiSignedOverflow(i16 signext %x) {
entry:
  %add = add nsw i16 16, %x
  %cmp = icmp sgt i16 %add, -32767
  ret i1 %cmp

; CHECK: === addiCmpiSignedOverflow
; CHECK: Optimized lowered selection DAG: %bb.0 'addiCmpiSignedOverflow:entry'
; CHECK:   [[REG1:t[0-9]+]]: i16 = truncate {{t[0-9]+}}
; CHECK:   [[REG2:t[0-9]+]]: i16 = add nsw [[REG1]], Constant:i16<16>
; CHECK:   {{t[0-9]+}}: i1 = setcc [[REG2]], Constant:i16<-32767>, setgt:ch
}

