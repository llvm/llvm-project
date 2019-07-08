; REQUIRES: asserts
; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:   -verify-machineinstrs -debug 2>&1 | FileCheck %s

define zeroext i1 @addiCmpiUnsign(i32 zeroext %x) {
entry:
  %add = add nuw i32 10, %x 
  %cmp = icmp ugt i32 %add, 100
  ret i1 %cmp

; CHECK: === addiCmpiUnsign
; CHECK: Optimized lowered selection DAG: %bb.0 'addiCmpiUnsign:entry'
; CHECK:   [[REG1:t[0-9]+]]: i32 = truncate {{t[0-9]+}}
; CHECK:   [[REG2:t[0-9]+]]: i32 = add nuw [[REG1]], Constant:i32<10>
; CHECK:   {{t[0-9]+}}: i1 = setcc [[REG2]], Constant:i32<100>, setugt:ch
}

define zeroext i1 @addiCmpiSign(i32 signext %x) {
entry:
  %add = add nsw i32 16, %x 
  %cmp = icmp sgt i32 %add, 30
  ret i1 %cmp

; CHECK: === addiCmpiSign
; CHECK: Optimized lowered selection DAG: %bb.0 'addiCmpiSign:entry'
; CHECK:   [[REG1:t[0-9]+]]: i32 = truncate {{t[0-9]+}}
; CHECK:   [[REG2:t[0-9]+]]: i32 = add nsw [[REG1]], Constant:i32<16>
; CHECK:   {{t[0-9]+}}: i1 = setcc [[REG2]], Constant:i32<30>, setgt:ch
}

define zeroext i1 @addiCmpiUnsignOverflow(i32 zeroext %x) {
entry:
  %add = add nuw i32 110, %x 
  %cmp = icmp ugt i32 %add, 100
  ret i1 %cmp

; CHECK: === addiCmpiUnsignOverflow
; CHECK: Optimized lowered selection DAG: %bb.0 'addiCmpiUnsignOverflow:entry'
; CHECK:   [[REG1:t[0-9]+]]: i32 = truncate {{t[0-9]+}}
; CHECK:   [[REG2:t[0-9]+]]: i32 = add nuw [[REG1]], Constant:i32<110>
; CHECK:   {{t[0-9]+}}: i1 = setcc [[REG2]], Constant:i32<100>, setugt:ch
}

define zeroext i1 @addiCmpiSignOverflow(i16 signext %x) {
entry:
  %add = add nsw i16 16, %x 
  %cmp = icmp sgt i16 %add, -32767
  ret i1 %cmp

; CHECK: === addiCmpiSignOverflow
; CHECK: Optimized lowered selection DAG: %bb.0 'addiCmpiSignOverflow:entry'
; CHECK:   [[REG1:t[0-9]+]]: i16 = truncate {{t[0-9]+}}
; CHECK:   [[REG2:t[0-9]+]]: i16 = add nsw [[REG1]], Constant:i16<16>
; CHECK:   {{t[0-9]+}}: i1 = setcc [[REG2]], Constant:i16<-32767>, setgt:ch
}

define zeroext i1 @addiCmpiNE(i16* %d) {
entry:
  %0 = load i16, i16* %d, align 2
  %dec = add i16 %0, 100
  store i16 %dec, i16* %d, align 2
  %tobool = icmp eq i16 %dec, 40
  br i1 %tobool, label %land.end, label %land.rhs

land.rhs:       
  ret i1 true

land.end:      
  ret i1 false

; CHECK: === addiCmpiNE
; CHECK: Optimized lowered selection DAG: %bb.0 'addiCmpiNE:entry'
; CHECK:   [[REG1:t[0-9]+]]: i16,ch = load
; CHECK:   [[REG2:t[0-9]+]]: i16 = add [[REG1]], Constant:i16<100>
; CHECK:   {{t[0-9]+}}: i1 = setcc [[REG2]], Constant:i16<40>, seteq:ch
}
