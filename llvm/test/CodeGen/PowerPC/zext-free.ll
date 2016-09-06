; RUN: llc -verify-machineinstrs -mcpu=ppc64 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: noreturn nounwind
define signext i32 @_Z1fRPc(i8** nocapture dereferenceable(8) %p) #0 {
entry:
  %.pre = load i8*, i8** %p, align 8
  br label %loop

loop:                                             ; preds = %loop.backedge, %entry
  %0 = phi i8* [ %.pre, %entry ], [ %.be, %loop.backedge ]
  %1 = load i8, i8* %0, align 1
  %tobool = icmp eq i8 %1, 0
  %incdec.ptr = getelementptr inbounds i8, i8* %0, i64 1
  store i8* %incdec.ptr, i8** %p, align 8
  %2 = load i8, i8* %incdec.ptr, align 1
  %tobool2 = icmp ne i8 %2, 0
  %or.cond = and i1 %tobool, %tobool2
  br i1 %or.cond, label %if.then3, label %loop.backedge

if.then3:                                         ; preds = %loop
  %incdec.ptr4 = getelementptr inbounds i8, i8* %0, i64 2
  store i8* %incdec.ptr4, i8** %p, align 8
  br label %loop.backedge

loop.backedge:                                    ; preds = %if.then3, %loop
  %.be = phi i8* [ %incdec.ptr4, %if.then3 ], [ %incdec.ptr, %loop ]
  br label %loop

; CHECK-LABEL: @_Z1fRPc
; CHECK-NOT: rlwinm {{[0-9]+}}, {{[0-9]+}}, 0, 24, 31
; CHECK-NOT: clrlwi {{[0-9]+}}, {{[0-9]+}}, 24
}

attributes #0 = { noreturn nounwind }

