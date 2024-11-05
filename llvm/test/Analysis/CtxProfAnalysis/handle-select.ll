; Check that we handle `step` instrumentations. These addorn `select`s.
; We don't want to confuse the `step` with normal increments, the latter of which
; we use for BB ID-ing: we want to keep the `step`s after inlining, except if
; the `select` is elided.
;
; RUN: split-file %s %t
; RUN: llvm-ctxprof-util fromJSON --input=%t/profile.json --output=%t/profile.ctxprofdata
;
; RUN: opt -passes=ctx-instr-gen %t/example.ll -use-ctx-profile=%t/profile.ctxprofdata -S -o - | FileCheck %s --check-prefix=INSTR
; RUN: opt -passes=ctx-instr-gen,module-inline %t/example.ll -use-ctx-profile=%t/profile.ctxprofdata -S -o - | FileCheck %s --check-prefix=POST-INL
; RUN: opt -passes=ctx-instr-gen,module-inline,ctx-prof-flatten %t/example.ll -use-ctx-profile=%t/profile.ctxprofdata -S -o - | FileCheck %s --check-prefix=FLATTEN

; INSTR-LABEL: yes:
; INSTR-NEXT:   call void @llvm.instrprof.increment(ptr @foo, i64 [[#]], i32 2, i32 1)
; INSTR-NEXT:   call void @llvm.instrprof.callsite(ptr @foo, i64 [[#]], i32 2, i32 0, ptr @bar)

; INSTR-LABEL: no:
; INSTR-NEXT:   call void @llvm.instrprof.callsite(ptr @foo, i64 [[#]], i32 2, i32 1, ptr @bar)

; INSTR-LABEL: define i32 @bar
; INSTR-NEXT:   call void @llvm.instrprof.increment(ptr @bar, i64 [[#]], i32 2, i32 0)
; INSTR-NEXT:   %inc =
; INSTR:        %test = icmp eq i32 %t, 0
; INSTR-NEXT:   %1  = zext i1 %test to i64
; INSTR-NEXT:   call void @llvm.instrprof.increment.step(ptr @bar, i64 [[#]], i32 2, i32 1, i64 %1)
; INSTR-NEXT:   %res = select

; POST-INL-LABEL: yes:
; POST-INL-NEXT:   call void @llvm.instrprof.increment
; POST-INL:        call void @llvm.instrprof.increment.step
; POST-INL-NEXT:   %res.i = select

; POST-INL-LABEL: no:
; POST-INL-NEXT:   call void @llvm.instrprof.increment
; POST-INL-NEXT:   br label

; POST-INL-LABEL: exit:
; POST-INL-NEXT:   %res = phi i32 [ %res.i, %yes ], [ 1, %no ]

; FLATTEN-LABEL: yes:
; FLATTEN:          %res.i = select i1 %test.i, i32 %inc.i, i32 %dec.i, !prof ![[SELPROF:[0-9]+]]
; FLATTEN-LABEL: no:
;
; See the profile, in the "yes" case we set the step counter's value, in @bar, to 3. The total
; entry count of that BB is 4.
; ![[SELPROF]] = !{!"branch_weights", i32 3, i32 1}

;--- example.ll
define i32 @foo(i32 %t) !guid !0 {
  %test = icmp slt i32 %t, 0
  br i1 %test, label %yes, label %no
yes:
  %res1 = call i32 @bar(i32 %t) alwaysinline
  br label %exit
no:
  ; this will result in eliding the select in @bar, when inlined.
  %res2 = call i32 @bar(i32 0) alwaysinline
  br label %exit
exit:
  %res = phi i32 [%res1, %yes], [%res2, %no]
  ret i32 %res
}

define i32 @bar(i32 %t) !guid !1 {
  %inc = add i32 %t, 1
  %dec = sub i32 %t, 1
  %test = icmp eq i32 %t, 0
  %res = select i1 %test, i32 %inc, i32 %dec
  ret i32 %res
}

!0 = !{i64 1234}
!1 = !{i64 5678}

;--- profile.json
[{"Guid":1234, "Counters":[10, 4], "Callsites":[[{"Guid": 5678, "Counters":[4,3]}],[{"Guid": 5678, "Counters":[6,6]}]]}]
