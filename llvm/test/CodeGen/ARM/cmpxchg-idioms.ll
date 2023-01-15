; RUN: llc -mtriple=thumbv7s-apple-ios7.0 -o - %s | FileCheck %s

define i32 @test_return(ptr %p, i32 %oldval, i32 %newval) {
; CHECK-LABEL: test_return:

; CHECK: ldrex [[LOADED:r[0-9]+]], [r0]
; CHECK: cmp [[LOADED]], r1
; CHECK: bne [[FAILED:LBB[0-9]+_[0-9]+]]

; CHECK: dmb ishst

; CHECK: [[LOOP:LBB[0-9]+_[0-9]+]]:
; CHECK: strex [[STATUS:r[0-9]+]], {{r[0-9]+}}, [r0]
; CHECK: cbz [[STATUS]], [[SUCCESS:LBB[0-9]+_[0-9]+]]

; CHECK: ldrex [[LOADED]], [r0]
; CHECK: cmp [[LOADED]], r1
; CHECK: beq [[LOOP]]

; CHECK: [[FAILED]]:
; CHECK-NOT: cmp {{r[0-9]+}}, {{r[0-9]+}}
; CHECK: clrex
; CHECK: movs r0, #0
; CHECK: dmb ish
; CHECK: bx lr

; CHECK: [[SUCCESS]]:
; CHECK-NOT: cmp {{r[0-9]+}}, {{r[0-9]+}}
; CHECK: movs r0, #1
; CHECK: dmb ish
; CHECK: bx lr

  %pair = cmpxchg ptr %p, i32 %oldval, i32 %newval seq_cst seq_cst
  %success = extractvalue { i32, i1 } %pair, 1
  %conv = zext i1 %success to i32
  ret i32 %conv
}

define i1 @test_return_bool(ptr %value, i8 %oldValue, i8 %newValue) {
; CHECK-LABEL: test_return_bool:

; CHECK: uxtb [[OLDBYTE:r[0-9]+]], r1

; CHECK: ldrexb [[LOADED:r[0-9]+]], [r0]
; CHECK: cmp [[LOADED]], [[OLDBYTE]]
; CHECK: bne [[FAIL:LBB[0-9]+_[0-9]+]]

; CHECK: dmb ishst

; CHECK: [[LOOP:LBB[0-9]+_[0-9]+]]:
; CHECK: strexb [[STATUS:r[0-9]+]], {{r[0-9]+}}, [r0]
; CHECK: cbz [[STATUS]], [[SUCCESS:LBB[0-9]+_[0-9]+]]

; CHECK: ldrexb [[LOADED]], [r0]
; CHECK: cmp [[LOADED]], [[OLDBYTE]]
; CHECK: beq [[LOOP]]


  ; FIXME: this eor is redundant. Need to teach DAG combine that.
; CHECK: [[FAIL]]:
; CHECK: clrex
; CHECK: movs [[TMP:r[0-9]+]], #0
; CHECK: eor r0, [[TMP]], #1
; CHECK: bx lr

; CHECK: [[SUCCESS]]:
; CHECK-NOT: cmp {{r[0-9]+}}, {{r[0-9]+}}
; CHECK: movs [[TMP:r[0-9]+]], #1
; CHECK: eor r0, [[TMP]], #1
; CHECK: bx lr


  %pair = cmpxchg ptr %value, i8 %oldValue, i8 %newValue acq_rel monotonic
  %success = extractvalue { i8, i1 } %pair, 1
  %failure = xor i1 %success, 1
  ret i1 %failure
}

define void @test_conditional(ptr %p, i32 %oldval, i32 %newval) {
; CHECK-LABEL: test_conditional:

; CHECK: ldrex [[LOADED:r[0-9]+]], [r0]
; CHECK: cmp [[LOADED]], r1
; CHECK: bne [[FAILED:LBB[0-9]+_[0-9]+]]

; CHECK: dmb ishst

; CHECK: [[LOOP:LBB[0-9]+_[0-9]+]]:
; CHECK: strex [[STATUS:r[0-9]+]], r2, [r0]
; CHECK: cbz [[STATUS]], [[SUCCESS:LBB[0-9]+_[0-9]+]]

; CHECK: ldrex [[LOADED]], [r0]
; CHECK: cmp [[LOADED]], r1
; CHECK: beq [[LOOP]]

; CHECK: [[FAILED]]:
; CHECK-NOT: cmp {{r[0-9]+}}, {{r[0-9]+}}
; CHECK: clrex
; CHECK: dmb ish
; CHECK: b.w _baz

; CHECK: [[SUCCESS]]:
; CHECK-NOT: cmp {{r[0-9]+}}, {{r[0-9]+}}
; CHECK: dmb ish
; CHECK: b.w _bar

  %pair = cmpxchg ptr %p, i32 %oldval, i32 %newval seq_cst seq_cst
  %success = extractvalue { i32, i1 } %pair, 1
  br i1 %success, label %true, label %false

true:
  tail call void @bar() #2
  br label %end

false:
  tail call void @baz() #2
  br label %end

end:
  ret void
}

declare void @bar()
declare void @baz()
