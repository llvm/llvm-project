; RUN: llc < %s -march=x86-64 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=x86-64 -mattr=slow-incdec -verify-machineinstrs | FileCheck %s --check-prefix SLOW_INC

; rdar://7103704

define void @sub1(i32* nocapture %p, i32 %v) nounwind ssp {
entry:
; CHECK-LABEL: sub1:
; CHECK: subl
  %0 = atomicrmw sub i32* %p, i32 %v monotonic
  ret void
}

define void @inc4(i64* nocapture %p) nounwind ssp {
entry:
; CHECK-LABEL: inc4:
; CHECK: incq
; SLOW_INC-LABEL: inc4:
; SLOW_INC-NOT: incq
  %0 = atomicrmw add i64* %p, i64 1 monotonic
  ret void
}

define void @add8(i64* nocapture %p) nounwind ssp {
entry:
; CHECK-LABEL: add8:
; CHECK: addq $2
  %0 = atomicrmw add i64* %p, i64 2 monotonic
  ret void
}

define void @add4(i64* nocapture %p, i32 %v) nounwind ssp {
entry:
; CHECK-LABEL: add4:
; CHECK: addq
  %0 = sext i32 %v to i64		; <i64> [#uses=1]
  %1 = atomicrmw add i64* %p, i64 %0 monotonic
  ret void
}

define void @inc3(i8* nocapture %p) nounwind ssp {
entry:
; CHECK-LABEL: inc3:
; CHECK: incb
; SLOW_INC-LABEL: inc3:
; SLOW_INC-NOT: incb
  %0 = atomicrmw add i8* %p, i8 1 monotonic
  ret void
}

define void @add7(i8* nocapture %p) nounwind ssp {
entry:
; CHECK-LABEL: add7:
; CHECK: addb $2
  %0 = atomicrmw add i8* %p, i8 2 monotonic
  ret void
}

define void @add3(i8* nocapture %p, i32 %v) nounwind ssp {
entry:
; CHECK-LABEL: add3:
; CHECK: addb
  %0 = trunc i32 %v to i8		; <i8> [#uses=1]
  %1 = atomicrmw add i8* %p, i8 %0 monotonic
  ret void
}

define void @inc2(i16* nocapture %p) nounwind ssp {
entry:
; CHECK-LABEL: inc2:
; CHECK: incw
; SLOW_INC-LABEL: inc2:
; SLOW_INC-NOT: incw
  %0 = atomicrmw add i16* %p, i16 1 monotonic
  ret void
}

define void @add6(i16* nocapture %p) nounwind ssp {
entry:
; CHECK-LABEL: add6:
; CHECK: addw $2
  %0 = atomicrmw add i16* %p, i16 2 monotonic
  ret void
}

define void @add2(i16* nocapture %p, i32 %v) nounwind ssp {
entry:
; CHECK-LABEL: add2:
; CHECK: addw
	%0 = trunc i32 %v to i16		; <i16> [#uses=1]
  %1 = atomicrmw add i16* %p, i16 %0 monotonic
  ret void
}

define void @inc1(i32* nocapture %p) nounwind ssp {
entry:
; CHECK-LABEL: inc1:
; CHECK: incl
; SLOW_INC-LABEL: inc1:
; SLOW_INC-NOT: incl
  %0 = atomicrmw add i32* %p, i32 1 monotonic
  ret void
}

define void @add5(i32* nocapture %p) nounwind ssp {
entry:
; CHECK-LABEL: add5:
; CHECK: addl $2
  %0 = atomicrmw add i32* %p, i32 2 monotonic
  ret void
}

define void @add1(i32* nocapture %p, i32 %v) nounwind ssp {
entry:
; CHECK-LABEL: add1:
; CHECK: addl
  %0 = atomicrmw add i32* %p, i32 %v monotonic
  ret void
}

define void @dec4(i64* nocapture %p) nounwind ssp {
entry:
; CHECK-LABEL: dec4:
; CHECK: decq
; SLOW_INC-LABEL: dec4:
; SLOW_INC-NOT: decq
  %0 = atomicrmw sub i64* %p, i64 1 monotonic
  ret void
}

define void @sub8(i64* nocapture %p) nounwind ssp {
entry:
; CHECK-LABEL: sub8:
; CHECK: subq $2
  %0 = atomicrmw sub i64* %p, i64 2 monotonic
  ret void
}

define void @sub4(i64* nocapture %p, i32 %v) nounwind ssp {
entry:
; CHECK-LABEL: sub4:
; CHECK: subq
	%0 = sext i32 %v to i64		; <i64> [#uses=1]
  %1 = atomicrmw sub i64* %p, i64 %0 monotonic
  ret void
}

define void @dec3(i8* nocapture %p) nounwind ssp {
entry:
; CHECK-LABEL: dec3:
; CHECK: decb
; SLOW_INC-LABEL: dec3:
; SLOW_INC-NOT: decb
  %0 = atomicrmw sub i8* %p, i8 1 monotonic
  ret void
}

define void @sub7(i8* nocapture %p) nounwind ssp {
entry:
; CHECK-LABEL: sub7:
; CHECK: subb $2
  %0 = atomicrmw sub i8* %p, i8 2 monotonic
  ret void
}

define void @sub3(i8* nocapture %p, i32 %v) nounwind ssp {
entry:
; CHECK-LABEL: sub3:
; CHECK: subb
	%0 = trunc i32 %v to i8		; <i8> [#uses=1]
  %1 = atomicrmw sub i8* %p, i8 %0 monotonic
  ret void
}

define void @dec2(i16* nocapture %p) nounwind ssp {
entry:
; CHECK-LABEL: dec2:
; CHECK: decw
; SLOW_INC-LABEL: dec2:
; SLOW_INC-NOT: decw
  %0 = atomicrmw sub i16* %p, i16 1 monotonic
  ret void
}

define void @sub6(i16* nocapture %p) nounwind ssp {
entry:
; CHECK-LABEL: sub6:
; CHECK: subw $2
  %0 = atomicrmw sub i16* %p, i16 2 monotonic
  ret void
}

define void @sub2(i16* nocapture %p, i32 %v) nounwind ssp {
entry:
; CHECK-LABEL: sub2:
; CHECK-NOT: negl
; CHECK: subw
	%0 = trunc i32 %v to i16		; <i16> [#uses=1]
  %1 = atomicrmw sub i16* %p, i16 %0 monotonic
  ret void
}

define void @dec1(i32* nocapture %p) nounwind ssp {
entry:
; CHECK-LABEL: dec1:
; CHECK: decl
; SLOW_INC-LABEL: dec1:
; SLOW_INC-NOT: decl
  %0 = atomicrmw sub i32* %p, i32 1 monotonic
  ret void
}

define void @sub5(i32* nocapture %p) nounwind ssp {
entry:
; CHECK-LABEL: sub5:
; CHECK: subl $2
  %0 = atomicrmw sub i32* %p, i32 2 monotonic
  ret void
}
