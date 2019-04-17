; RUN: opt -simplifycfg -S < %s | FileCheck %s

define void @ifconvertstore(i32* %A, i32 %B, i32 %C, i32 %D) {
; CHECK-LABEL: @ifconvertstore(
; CHECK:         store i32 %B, i32* %A
; CHECK-NEXT:    [[CMP:%.*]] = icmp sgt i32 %D, 42
; CHECK-NEXT:    [[C_B:%.*]] = select i1 [[CMP]], i32 %C, i32 %B, !prof !0
; CHECK-NEXT:    store i32 [[C_B]], i32* %A
; CHECK-NEXT:    ret void
;
entry:
; First store to the location.
  store i32 %B, i32* %A
  %cmp = icmp sgt i32 %D, 42
  br i1 %cmp, label %if.then, label %ret.end, !prof !0

; Make sure we speculate stores like the following one. It is cheap compared to
; a mispredicated branch.
if.then:
  store i32 %C, i32* %A
  br label %ret.end

ret.end:
  ret void
}

; Store to a different location.

define void @noifconvertstore1(i32* %A1, i32* %A2, i32 %B, i32 %C, i32 %D) {
; CHECK-LABEL: @noifconvertstore1(
; CHECK-NOT: select
;
entry:
  store i32 %B, i32* %A1
  %cmp = icmp sgt i32 %D, 42
  br i1 %cmp, label %if.then, label %ret.end

if.then:
  store i32 %C, i32* %A2
  br label %ret.end

ret.end:
  ret void
}

; This function could store to our address, so we can't repeat the first store a second time.
declare void @unknown_fun()

define void @noifconvertstore2(i32* %A, i32 %B, i32 %C, i32 %D) {
; CHECK-LABEL: @noifconvertstore2(
; CHECK-NOT: select
;
entry:
; First store to the location.
  store i32 %B, i32* %A
  call void @unknown_fun()
  %cmp6 = icmp sgt i32 %D, 42
  br i1 %cmp6, label %if.then, label %ret.end

if.then:
  store i32 %C, i32* %A
  br label %ret.end

ret.end:
  ret void
}

; Make sure we don't speculate volatile stores.

define void @noifconvertstore_volatile(i32* %A, i32 %B, i32 %C, i32 %D) {
; CHECK-LABEL: @noifconvertstore_volatile(
; CHECK-NOT: select
;
entry:
; First store to the location.
  store i32 %B, i32* %A
  %cmp6 = icmp sgt i32 %D, 42
  br i1 %cmp6, label %if.then, label %ret.end

if.then:
  store volatile i32 %C, i32* %A
  br label %ret.end

ret.end:
  ret void
}

; CHECK: !0 = !{!"branch_weights", i32 3, i32 5}
!0 = !{!"branch_weights", i32 3, i32 5}

