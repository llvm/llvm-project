; RUN: llc -verify-machineinstrs -print-before=peephole-opt -print-after=peephole-opt -mtriple=powerpc64-unknown-linux-gnu -o /dev/null 2>&1 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -print-before=peephole-opt -print-after=peephole-opt -mtriple=powerpc64le-unknown-linux-gnu -o /dev/null 2>&1 < %s | FileCheck %s

; CHECK-LABEL: fn1
define signext i32 @fn1(i32 %baz) {
  %1 = mul nsw i32 %baz, 208
  %2 = zext i32 %1 to i64
  %3 = shl i64 %2, 48
  %4 = ashr exact i64 %3, 48
; CHECK: ANDIo8 killed {{[^,]+}}, 65520, implicit-def dead %cr0;
; CHECK: CMPLDI
; CHECK: BCC

; CHECK: ANDIo8 {{[^,]+}}, 65520, implicit-def %cr0;
; CHECK: COPY %cr0
; CHECK: BCC
  %5 = icmp eq i64 %4, 0
  br i1 %5, label %foo, label %bar

foo:
  ret i32 1

bar:
  ret i32 0
}

; CHECK-LABEL: fn2
define signext i32 @fn2(i64 %a, i64 %b) {
; CHECK: OR8o {{[^, ]+}}, {{[^, ]+}}, implicit-def %cr0;
; CHECK: [[CREG:[^, ]+]]:crrc = COPY killed %cr0
; CHECK: BCC 12, killed [[CREG]]
  %1 = or i64 %b, %a
  %2 = icmp sgt i64 %1, -1
  br i1 %2, label %foo, label %bar

foo:
  ret i32 1

bar:
  ret i32 0
}

; CHECK-LABEL: fn3
define signext i32 @fn3(i32 %a) {
; CHECK: ANDIo killed {{[%0-9]+}}, 10, implicit-def %cr0;
; CHECK: [[CREG:[^, ]+]]:crrc = COPY %cr0
; CHECK: BCC 76, killed [[CREG]]
  %1 = and i32 %a, 10
  %2 = icmp ne i32 %1, 0
  br i1 %2, label %foo, label %bar

foo:
  ret i32 1

bar:
  ret i32 0
}

; This test case confirms that a record-form instruction is
; generated even if the branch has a static branch hint.

; CHECK-LABEL: fn4
define i64 @fn4(i64 %a, i64 %b) {
; CHECK: ADD8o
; CHECK-NOT: CMP
; CHECK: BCC 71

entry:
  %add = add nsw i64 %b, %a
  %cmp = icmp eq i64 %add, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @exit(i32 signext 0) #3
  unreachable

if.end:
  ret i64 %add
}

declare void @exit(i32 signext)

; Since %v1 and %v2 are zero-extended 32-bit values, %1 is also zero-extended.
; In this case, we want to use ORo instead of OR + CMPLWI.

; CHECK-LABEL: fn5
define zeroext i32 @fn5(i32* %p1, i32* %p2) {
; CHECK: ORo
; CHECK-NOT: CMP
; CHECK: BCC
  %v1 = load i32, i32* %p1
  %v2 = load i32, i32* %p2
  %1 = or i32 %v1, %v2
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %foo, label %bar

foo:
  ret i32 1

bar:
  ret i32 0
}

; This test confirms record-form instructions are emitted for comparison
; against a non-zero value.

; CHECK-LABEL: fn6
define i8* @fn6(i8* readonly %p) {
; CHECK: LBZU
; CHECK: EXTSBo
; CHECK-NOT: CMP
; CHECK: BCC
; CHECK: LBZU
; CHECK: EXTSBo
; CHECK-NOT: CMP
; CHECK: BCC

entry:
  %incdec.ptr = getelementptr inbounds i8, i8* %p, i64 -1
  %0 = load i8, i8* %incdec.ptr
  %cmp = icmp sgt i8 %0, -1
  br i1 %cmp, label %out, label %if.end

if.end:
  %incdec.ptr2 = getelementptr inbounds i8, i8* %p, i64 -2
  %1 = load i8, i8* %incdec.ptr2
  %cmp4 = icmp sgt i8 %1, -1
  br i1 %cmp4, label %out, label %cleanup

out:
  %p.addr.0 = phi i8* [ %incdec.ptr, %entry ], [ %incdec.ptr2, %if.end ]
  br label %cleanup

cleanup:
  %retval.0 = phi i8* [ %p.addr.0, %out ], [ null, %if.end ]
  ret i8* %retval.0
}
