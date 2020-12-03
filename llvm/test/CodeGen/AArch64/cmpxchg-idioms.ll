; RUN: llc -mtriple=aarch64-apple-ios7.0 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-apple-ios7.0 -mattr=+outline-atomics -o - %s | FileCheck %s --check-prefix=OUTLINE-ATOMICS

define i32 @test_return(i32* %p, i32 %oldval, i32 %newval) {
; OUTLINE-ATOMICS: bl ___aarch64_cas4_acq_rel
; CHECK-LABEL: test_return:

; CHECK: [[LOOP:LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxr [[LOADED:w[0-9]+]], [x0]
; CHECK: cmp [[LOADED]], w1
; CHECK: b.ne [[FAILED:LBB[0-9]+_[0-9]+]]

; CHECK: stlxr [[STATUS:w[0-9]+]], {{w[0-9]+}}, [x0]
; CHECK: cbnz [[STATUS]], [[LOOP]]

; CHECK-NOT: cmp {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: mov w0, #1
; CHECK: ret

; CHECK: [[FAILED]]:
; CHECK-NOT: cmp {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: mov w0, wzr
; CHECK: ret

  %pair = cmpxchg i32* %p, i32 %oldval, i32 %newval seq_cst seq_cst
  %success = extractvalue { i32, i1 } %pair, 1
  %conv = zext i1 %success to i32
  ret i32 %conv
}

define i1 @test_return_bool(i8* %value, i8 %oldValue, i8 %newValue) {
; OUTLINE-ATOMICS: bl ___aarch64_cas1_acq_rel
; CHECK-LABEL: test_return_bool:

; CHECK: [[LOOP:LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxrb [[LOADED:w[0-9]+]], [x0]
; CHECK: cmp [[LOADED]], w1, uxtb
; CHECK: b.ne [[FAILED:LBB[0-9]+_[0-9]+]]

; CHECK: stlxrb [[STATUS:w[0-9]+]], {{w[0-9]+}}, [x0]
; CHECK: cbnz [[STATUS]], [[LOOP]]

; CHECK-NOT: cmp {{w[0-9]+}}, {{w[0-9]+}}
  ; FIXME: DAG combine should be able to deal with this.
; CHECK: mov [[TMP:w[0-9]+]], #1
; CHECK: eor w0, [[TMP]], #0x1
; CHECK: ret

; CHECK: [[FAILED]]:
; CHECK-NOT: cmp {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: eor w0, wzr, #0x1
; CHECK: ret

  %pair = cmpxchg i8* %value, i8 %oldValue, i8 %newValue acq_rel monotonic
  %success = extractvalue { i8, i1 } %pair, 1
  %failure = xor i1 %success, 1
  ret i1 %failure
}

define void @test_conditional(i32* %p, i32 %oldval, i32 %newval) {
; OUTLINE-ATOMICS: bl ___aarch64_cas4_acq_rel
; CHECK-LABEL: test_conditional:

; CHECK: [[LOOP:LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxr [[LOADED:w[0-9]+]], [x0]
; CHECK: cmp [[LOADED]], w1
; CHECK: b.ne [[FAILED:LBB[0-9]+_[0-9]+]]

; CHECK: stlxr [[STATUS:w[0-9]+]], w2, [x0]
; CHECK: cbnz [[STATUS]], [[LOOP]]

; CHECK-NOT: cmp {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: b _bar

; CHECK: [[FAILED]]:
; CHECK-NOT: cmp {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: b _baz

  %pair = cmpxchg i32* %p, i32 %oldval, i32 %newval seq_cst seq_cst
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

define i1 @test_conditional2(i32 %a, i32 %b, i32* %c) {
; OUTLINE-ATOMICS: bl ___aarch64_cas4_acq_rel
; CHECK-LABEL: test_conditional2:
; CHECK: [[LOOP:LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxr [[LOADED:w[0-9]+]], [x19]
; CHECK: cmp [[LOADED]], w21
; CHECK: b.ne [[FAILED:LBB[0-9]+_[0-9]+]]

; CHECK: stlxr [[STATUS:w[0-9]+]], w20, [x19]
; CHECK: cbnz [[STATUS]], [[LOOP]]
; CHECK: mov [[STATUS]], #1
; CHECK: b [[PH:LBB[0-9]+_[0-9]+]]

; CHECK: [[FAILED]]:
; CHECK-NOT: cmp {{w[0-9]+}}, {{w[0-9]+}}

; verify the preheader is simplified by simplifycfg.
; CHECK: [[PH]]:
; CHECK: mov w22, #2
; CHECK-NOT: mov w22, #4
; CHECK-NOT: cmn w22, #4
; CHECK: [[LOOP2:LBB[0-9]+_[0-9]+]]: ; %for.cond
; CHECK-NOT: b.ne [[LOOP2]]
; CHECK-NOT: b {{LBB[0-9]+_[0-9]+}}
; CHECK: bl _foo
entry:
  %pair = cmpxchg i32* %c, i32 %a, i32 %b seq_cst seq_cst
  %success = extractvalue { i32, i1 } %pair, 1
  br label %for.cond

for.cond:                                         ; preds = %if.end, %entry
  %i.0 = phi i32 [ 2, %entry ], [ %dec, %if.end ]
  %changed.0.off0 = phi i1 [ %success, %entry ], [ %changed.1.off0, %if.end ]
  %dec = add nsw i32 %i.0, -1
  %tobool = icmp eq i32 %i.0, 0
  br i1 %tobool, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  %changed.0.off0.lcssa = phi i1 [ %changed.0.off0, %for.cond ]
  ret i1 %changed.0.off0.lcssa

for.body:                                         ; preds = %for.cond
  %or = or i32 %a, %b
  %idxprom = sext i32 %dec to i64
  %arrayidx = getelementptr inbounds i32, i32* %c, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %cmp = icmp eq i32 %or, %0
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  store i32 %or, i32* %arrayidx, align 4
  tail call void @foo()
  br label %if.end

if.end:                                           ; preds = %for.body, %if.then
  %changed.1.off0 = phi i1 [ false, %if.then ], [ %changed.0.off0, %for.body ]
  br label %for.cond
}

declare void @foo()
