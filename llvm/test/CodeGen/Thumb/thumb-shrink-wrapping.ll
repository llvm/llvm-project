; RUN: llc %s -o - -enable-shrink-wrap=true -ifcvt-fn-start=1 -ifcvt-fn-stop=0 -tail-dup-placement=0 -mtriple=thumb-macho \
; RUN:      | FileCheck %s --check-prefix=CHECK --check-prefix=ENABLE --check-prefix=ENABLE-V4T
; RUN: llc %s -o - -enable-shrink-wrap=true -ifcvt-fn-start=1 -ifcvt-fn-stop=0 -tail-dup-placement=0 -mtriple=thumbv5-macho \
; RUN:      | FileCheck %s --check-prefix=CHECK --check-prefix=ENABLE --check-prefix=ENABLE-V5T
; RUN: llc %s -o - -enable-shrink-wrap=false -ifcvt-fn-start=1 -ifcvt-fn-stop=0 -tail-dup-placement=0 -mtriple=thumb-macho \
; RUN:      | FileCheck %s --check-prefix=CHECK --check-prefix=DISABLE --check-prefix=DISABLE-V4T
; RUN: llc %s -o - -enable-shrink-wrap=false -ifcvt-fn-start=1 -ifcvt-fn-stop=0 -tail-dup-placement=0 -mtriple=thumbv5-macho \
; RUN:      | FileCheck %s --check-prefix=CHECK --check-prefix=DISABLE --check-prefix=DISABLE-V5T

;
; Note: Lots of tests use inline asm instead of regular calls.
; This allows to have a better control on what the allocation will do.
; Otherwise, we may have spill right in the entry block, defeating
; shrink-wrapping. Moreover, some of the inline asm statements (nop)
; are here to ensure that the related paths do not end up as critical
; edges.
; Also disable the late if-converter as it makes harder to reason on
; the diffs.
; Disable tail-duplication during placement, as v4t vs v5t get different
; results due to branches not being analyzable under v5

; Initial motivating example: Simple diamond with a call just on one side.
; CHECK-LABEL: foo:
;
; Compare the arguments and jump to exit.
; No prologue needed.
; ENABLE: cmp r0, r1
; ENABLE-NEXT: bge [[EXIT_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; CHECK: push {r7, lr}
; CHECK: sub sp, #8
;
; Compare the arguments and jump to exit.
; After the prologue is set.
; DISABLE: cmp r0, r1
; DISABLE-NEXT: bge [[EXIT_LABEL:LBB[0-9_]+]]
;
; Store %a in the alloca.
; CHECK: str r0, [sp, #4]
; Set the alloca address in the second argument.
; Set the first argument to zero.
; CHECK: movs r0, #0
; CHECK-NEXT: add r1, sp, #4
; CHECK-NEXT: bl
;
; With shrink-wrapping, epilogue is just after the call.
; ENABLE-NEXT: add sp, #8
; ENABLE-V5T-NEXT: pop {r7, pc}
; ENABLE-V4T-NEXT: pop {r7}
; ENABLE-V4T-NEXT: pop {r1}
; ENABLE-V4T-NEXT: mov lr, r1
;
; CHECK: [[EXIT_LABEL]]:
;
; Without shrink-wrapping, epilogue is in the exit block.
; Epilogue code. (What we pop does not matter.)
; DISABLE: add sp, #8
; DISABLE-V5T-NEXT: pop {r7, pc}
; DISABLE-V4T-NEXT: pop {r7}
; DISABLE-V4T-NEXT: pop {r1}
; DISABLE-V4T-NEXT: bx r1
;
; ENABLE-NEXT: bx lr
define i32 @foo(i32 %a, i32 %b) {
  %tmp = alloca i32, align 4
  %tmp2 = icmp slt i32 %a, %b
  br i1 %tmp2, label %true, label %false

true:
  store i32 %a, i32* %tmp, align 4
  %tmp4 = call i32 @doSomething(i32 0, i32* %tmp)
  br label %false

false:
  %tmp.0 = phi i32 [ %tmp4, %true ], [ %a, %0 ]
  ret i32 %tmp.0
}


; Same, but the final BB is non-trivial, so we don't duplicate the return inst.
; CHECK-LABEL: bar:
;
; With shrink-wrapping, epilogue is just after the call.
; CHECK: bl
; ENABLE-NEXT: add sp, #8
; ENABLE-NEXT: pop {r7}
; ENABLE-NEXT: pop {r0}
; ENABLE-NEXT: mov lr, r0
;
; CHECK: movs r0, #42
;
; Without shrink-wrapping, epilogue is in the exit block.
; Epilogue code. (What we pop does not matter.)
; DISABLE: add sp, #8
; DISABLE-V5T-NEXT: pop {r7, pc}
; DISABLE-V4T-NEXT: pop {r7}
; DISABLE-V4T-NEXT: pop {r1}
; DISABLE-V4T-NEXT: bx r1
;
; ENABLE-NEXT: bx lr
define i32 @bar(i32 %a, i32 %b) {
  %tmp = alloca i32, align 4
  %tmp2 = icmp slt i32 %a, %b
  br i1 %tmp2, label %true, label %false

true:
  store i32 %a, i32* %tmp, align 4
  %tmp4 = call i32 @doSomething(i32 0, i32* %tmp)
  br label %false

false:
  ret i32 42
}

; Function Attrs: optsize
declare i32 @doSomething(i32, i32*)


; Check that we do not perform the restore inside the loop whereas the save
; is outside.
; CHECK-LABEL: freqSaveAndRestoreOutsideLoop:
;
; Shrink-wrapping allows to skip the prologue in the else case.
; ENABLE: cmp r0, #0
; ENABLE-NEXT: beq [[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the CSR used in the inline asm: r4.
; CHECK: push {r4, lr}
;
; DISABLE: cmp r0, #0
; DISABLE-NEXT: beq [[ELSE_LABEL:LBB[0-9_]+]]
;
; SUM is in r0 because it is coalesced with the second
; argument on the else path.
; CHECK: movs [[SUM:r0]], #0
; CHECK-NEXT: movs [[IV:r[0-9]+]], #10
;
; Next BB.
; CHECK: [[LOOP:LBB[0-9_]+]]: @ %for.body
; CHECK: movs [[TMP:r[0-9]+]], #1
; CHECK: adds [[SUM]], [[TMP]], [[SUM]]
; CHECK-NEXT: subs [[IV]], [[IV]], #1
; CHECK-NEXT: bne [[LOOP]]
;
; Next BB.
; SUM << 3.
; CHECK: lsls [[SUM]], [[SUM]], #3
;
; Duplicated epilogue.
; DISABLE-V5T: pop {r4, pc}
; DISABLE-V4T: b [[END_LABEL:LBB[0-9_]+]]
;
; CHECK: [[ELSE_LABEL]]: @ %if.else
; Shift second argument by one and store into returned register.
; CHECK: lsls r0, r1, #1
; DISABLE-V5T-NEXT: pop {r4, pc}
; DISABLE-V4T-NEXT: [[END_LABEL]]: @ %if.end
; DISABLE-V4T-NEXT: pop {r4}
; DISABLE-V4T-NEXT: pop {r1}
; DISABLE-V4T-NEXT: bx r1
;
; ENABLE-V5T-NEXT: {{LBB[0-9_]+}}: @ %if.end
; ENABLE-NEXT: bx lr
define i32 @freqSaveAndRestoreOutsideLoop(i32 %cond, i32 %N) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %for.preheader

for.preheader:
  tail call void asm "nop", ""()
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ %inc, %for.body ], [ 0, %for.preheader ]
  %sum.04 = phi i32 [ %add, %for.body ], [ 0, %for.preheader ]
  %call = tail call i32 asm sideeffect "movs $0, #1", "=r,~{r4}"()
  %add = add nsw i32 %call, %sum.04
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %shl = shl i32 %add, 3
  br label %if.end

if.else:                                          ; preds = %entry
  %mul = shl nsw i32 %N, 1
  br label %if.end

if.end:                                           ; preds = %if.else, %for.end
  %sum.1 = phi i32 [ %shl, %for.end ], [ %mul, %if.else ]
  ret i32 %sum.1
}

declare i32 @something(...)

; Check that we do not perform the shrink-wrapping inside the loop even
; though that would be legal. The cost model must prevent that.
; CHECK-LABEL: freqSaveAndRestoreOutsideLoop2:
; Prologue code.
; Make sure we save the CSR used in the inline asm: r4.
; CHECK: push {r4
; This is the nop.
; CHECK: mov r8, r8
; CHECK: movs [[SUM:r0]], #0
; CHECK-NEXT: movs [[IV:r[0-9]+]], #10
; Next BB.
; CHECK: [[LOOP_LABEL:LBB[0-9_]+]]: @ %for.body
; CHECK: movs [[TMP:r[0-9]+]], #1
; CHECK: adds [[SUM]], [[TMP]], [[SUM]]
; CHECK-NEXT: subs [[IV]], [[IV]], #1
; CHECK-NEXT: bne [[LOOP_LABEL]]
; Next BB.
; CHECK: @ %for.exit
; This is the nop.
; CHECK: mov r8, r8
; CHECK: pop {r4
define i32 @freqSaveAndRestoreOutsideLoop2(i32 %cond) {
entry:
  br label %for.preheader

for.preheader:
  tail call void asm "nop", ""()
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.04 = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]
  %sum.03 = phi i32 [ 0, %for.preheader ], [ %add, %for.body ]
  %call = tail call i32 asm sideeffect "movs $0, #1", "=r,~{r4}"()
  %add = add nsw i32 %call, %sum.03
  %inc = add nuw nsw i32 %i.04, 1
  %exitcond = icmp eq i32 %inc, 10
  br i1 %exitcond, label %for.exit, label %for.body

for.exit:
  tail call void asm "nop", ""()
  br label %for.end

for.end:                                          ; preds = %for.body
  ret i32 %add
}

; Check with a more complex case that we do not have save within the loop and
; restore outside.
; CHECK-LABEL: loopInfoSaveOutsideLoop:
;
; ENABLE: cmp r0, #0
; ENABLE-NEXT: beq [[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the CSR used in the inline asm: r4.
; CHECK: push {r4, lr}
;
; DISABLE: cmp r0, #0
; DISABLE-NEXT: beq [[ELSE_LABEL:LBB[0-9_]+]]
;
; SUM is in r0 because it is coalesced with the second
; argument on the else path.
; CHECK: movs [[SUM:r0]], #0
; CHECK-NEXT: movs [[IV:r[0-9]+]], #10
;
; Next BB.
; CHECK: [[LOOP:LBB[0-9_]+]]: @ %for.body
; CHECK: movs [[TMP:r[0-9]+]], #1
; CHECK: adds [[SUM]], [[TMP]], [[SUM]]
; CHECK-NEXT: subs [[IV]], [[IV]], #1
; CHECK-NEXT: bne [[LOOP]]
;
; Next BB.
; SUM << 3.
; CHECK: lsls [[SUM]], [[SUM]], #3
; ENABLE-V5T-NEXT: pop {r4, pc}
; ENABLE-V4T-NEXT: pop {r4}
; ENABLE-V4T-NEXT: pop {r1}
; ENABLE-V4T-NEXT: bx r1
;
; Duplicated epilogue.
; DISABLE-V5T: pop {r4, pc}
; DISABLE-V4T: b [[END_LABEL:LBB[0-9_]+]]
;
; CHECK: [[ELSE_LABEL]]: @ %if.else
; Shift second argument by one and store into returned register.
; CHECK: lsls r0, r1, #1
; DISABLE-V5T-NEXT: pop {r4, pc}
; DISABLE-V4T-NEXT: [[END_LABEL]]: @ %if.end
; DISABLE-V4T-NEXT: pop {r4}
; DISABLE-V4T-NEXT: pop {r1}
; DISABLE-V4T-NEXT: bx r1
;
; ENABLE-V5T-NEXT: {{LBB[0-9_]+}}: @ %if.end
; ENABLE-NEXT: bx lr
define i32 @loopInfoSaveOutsideLoop(i32 %cond, i32 %N) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %for.preheader

for.preheader:
  tail call void asm "nop", ""()
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ %inc, %for.body ], [ 0, %for.preheader ]
  %sum.04 = phi i32 [ %add, %for.body ], [ 0, %for.preheader ]
  %call = tail call i32 asm sideeffect "movs $0, #1", "=r,~{r4}"()
  %add = add nsw i32 %call, %sum.04
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  tail call void asm "nop", "~{r4}"()
  %shl = shl i32 %add, 3
  br label %if.end

if.else:                                          ; preds = %entry
  %mul = shl nsw i32 %N, 1
  br label %if.end

if.end:                                           ; preds = %if.else, %for.end
  %sum.1 = phi i32 [ %shl, %for.end ], [ %mul, %if.else ]
  ret i32 %sum.1
}

declare void @somethingElse(...)

; Check with a more complex case that we do not have restore within the loop and
; save outside.
; CHECK-LABEL: loopInfoRestoreOutsideLoop:
;
; ENABLE: cmp r0, #0
; ENABLE-NEXT: beq [[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the CSR used in the inline asm: r4.
; CHECK: push {r4, lr}
;
; DISABLE-NEXT: cmp r0, #0
; DISABLE-NEXT: beq [[ELSE_LABEL:LBB[0-9_]+]]
;
; SUM is in r0 because it is coalesced with the second
; argument on the else path.
; CHECK: movs [[SUM:r0]], #0
; CHECK-NEXT: movs [[IV:r[0-9]+]], #10
;
; Next BB.
; CHECK: [[LOOP:LBB[0-9_]+]]: @ %for.body
; CHECK: movs [[TMP:r[0-9]+]], #1
; CHECK: adds [[SUM]], [[TMP]], [[SUM]]
; CHECK-NEXT: subs [[IV]], [[IV]], #1
; CHECK-NEXT: bne [[LOOP]]
;
; Next BB.
; SUM << 3.
; CHECK: lsls [[SUM]], [[SUM]], #3
; ENABLE-V5T-NEXT: pop {r4, pc}
; ENABLE-V4T-NEXT: pop {r4}
; ENABLE-V4T-NEXT: pop {r1}
; ENABLE-V4T-NEXT: bx r1
;
; Duplicated epilogue.
; DISABLE-V5T: pop {r4, pc}
; DISABLE-V4T: b [[END_LABEL:LBB[0-9_]+]]
;
; CHECK: [[ELSE_LABEL]]: @ %if.else
; Shift second argument by one and store into returned register.
; CHECK: lsls r0, r1, #1
; DISABLE-V5T-NEXT: pop {r4, pc}
; DISABLE-V4T-NEXT: [[END_LABEL]]: @ %if.end
; DISABLE-V4T-NEXT: pop {r4}
; DISABLE-V4T-NEXT: pop {r1}
; DISABLE-V4T-NEXT: bx r1
;
; ENABLE-V5T-NEXT: {{LBB[0-9_]+}}: @ %if.end
; ENABLE-NEXT: bx lr
define i32 @loopInfoRestoreOutsideLoop(i32 %cond, i32 %N) nounwind {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  tail call void asm "nop", "~{r4}"()
  br label %for.body

for.body:                                         ; preds = %for.body, %if.then
  %i.05 = phi i32 [ 0, %if.then ], [ %inc, %for.body ]
  %sum.04 = phi i32 [ 0, %if.then ], [ %add, %for.body ]
  %call = tail call i32 asm sideeffect "movs $0, #1", "=r,~{r4}"()
  %add = add nsw i32 %call, %sum.04
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %shl = shl i32 %add, 3
  br label %if.end

if.else:                                          ; preds = %entry
  %mul = shl nsw i32 %N, 1
  br label %if.end

if.end:                                           ; preds = %if.else, %for.end
  %sum.1 = phi i32 [ %shl, %for.end ], [ %mul, %if.else ]
  ret i32 %sum.1
}

; Check that we handle function with no frame information correctly.
; CHECK-LABEL: emptyFrame:
; CHECK: @ %entry
; CHECK-NEXT: movs r0, #0
; CHECK-NEXT: bx lr
define i32 @emptyFrame() {
entry:
  ret i32 0
}

; Check that we handle inline asm correctly.
; CHECK-LABEL: inlineAsm:
;
; ENABLE: cmp r0, #0
; ENABLE-NEXT: beq [[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the CSR used in the inline asm: r4.
; CHECK: push {r4, lr}
;
; DISABLE: cmp r0, #0
; DISABLE-NEXT: beq [[ELSE_LABEL:LBB[0-9_]+]]
;
; CHECK: movs [[IV:r[0-9]+]], #10
;
; Next BB.
; CHECK: [[LOOP:LBB[0-9_]+]]: @ %for.body
; CHECK: movs r4, #1
; CHECK: subs [[IV]], [[IV]], #1
; CHECK-NEXT: bne [[LOOP]]
;
; Next BB.
; CHECK: movs r0, #0
; ENABLE-V5T-NEXT: pop {r4, pc}
; ENABLE-V4T-NEXT: pop {r4}
; ENABLE-V4T-NEXT: pop {r1}
; ENABLE-V4T-NEXT: bx r1
;
; Duplicated epilogue.
; DISABLE-V5T-NEXT: pop {r4, pc}
; DISABLE-V4T-NEXT: b [[END_LABEL:LBB[0-9_]+]]
;
; CHECK: [[ELSE_LABEL]]: @ %if.else
; Shift second argument by one and store into returned register.
; CHECK: lsls r0, r1, #1
; DISABLE-V5T-NEXT: pop {r4, pc}
; DISABLE-V4T-NEXT: [[END_LABEL]]: @ %if.end
; DISABLE-V4T-NEXT: pop {r4}
; DISABLE-V4T-NEXT: pop {r1}
; DISABLE-V4T-NEXT: bx r1
;
; ENABLE-V5T-NEXT: {{LBB[0-9_]+}}: @ %if.end
; ENABLE-NEXT: bx lr
define i32 @inlineAsm(i32 %cond, i32 %N) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %for.preheader

for.preheader:
  tail call void asm "nop", ""()
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.03 = phi i32 [ %inc, %for.body ], [ 0, %for.preheader ]
  tail call void asm sideeffect "movs r4, #1", "~{r4}"()
  %inc = add nuw nsw i32 %i.03, 1
  %exitcond = icmp eq i32 %inc, 10
  br i1 %exitcond, label %for.exit, label %for.body

for.exit:
  tail call void asm "nop", ""()
  br label %if.end

if.else:                                          ; preds = %entry
  %mul = shl nsw i32 %N, 1
  br label %if.end

if.end:                                           ; preds = %for.body, %if.else
  %sum.0 = phi i32 [ %mul, %if.else ], [ 0, %for.exit ]
  ret i32 %sum.0
}

; Check that we handle calls to variadic functions correctly.
; CHECK-LABEL: callVariadicFunc:
;
; ENABLE: cmp r0, #0
; ENABLE-NEXT: beq [[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; CHECK: push {[[TMP:r[0-9]+]], lr}
; CHECK: sub sp, #16
;
; DISABLE: cmp r0, #0
; DISABLE-NEXT: beq [[ELSE_LABEL:LBB[0-9_]+]]
;
; Setup of the varags.
; CHECK: str r1, [sp]
; CHECK-NEXT: str r1, [sp, #4]
; CHECK-NEXT: str r1, [sp, #8]
; CHECK:      movs r0, r1
; CHECK-NEXT: movs r2, r1
; CHECK-NEXT: movs r3, r1
; CHECK-NEXT: bl
; CHECK-NEXT: lsls r0, r0, #3
;
; ENABLE-NEXT: add sp, #16
; ENABLE-V5T-NEXT: pop {[[TMP]], pc}
; ENABLE-V4T-NEXT: pop {[[TMP]]}
; ENABLE-V4T-NEXT: pop {r1}
; ENABLE-V4T-NEXT: bx r1
;
; Duplicated epilogue.
; DISABLE-V5T-NEXT: add sp, #16
; DISABLE-V5T-NEXT: pop {[[TMP]], pc}
; DISABLE-V4T-NEXT: b [[END_LABEL:LBB[0-9_]+]]
;
; CHECK: [[ELSE_LABEL]]: @ %if.else
; Shift second argument by one and store into returned register.
; CHECK: lsls r0, r1, #1
;
; Epilogue code.
; ENABLE-V5T-NEXT: {{LBB[0-9_]+}}: @ %if.end
; ENABLE-NEXT: bx lr
;
; DISABLE-V4T-NEXT: [[END_LABEL]]: @ %if.end
; DISABLE-NEXT: add sp, #16
; DISABLE-V5T-NEXT: pop {[[TMP]], pc}
; DISABLE-V4T-NEXT: pop {[[TMP]]}
; DISABLE-V4T-NEXT: pop {r1}
; DISABLE-V4T-NEXT: bx r1
define i32 @callVariadicFunc(i32 %cond, i32 %N) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %call = tail call i32 (i32, ...) @someVariadicFunc(i32 %N, i32 %N, i32 %N, i32 %N, i32 %N, i32 %N, i32 %N)
  %shl = shl i32 %call, 3
  br label %if.end

if.else:                                          ; preds = %entry
  %mul = shl nsw i32 %N, 1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %sum.0 = phi i32 [ %shl, %if.then ], [ %mul, %if.else ]
  ret i32 %sum.0
}

declare i32 @someVariadicFunc(i32, ...)

; Make sure we do not insert unreachable code after noreturn function.
; Although this is not incorrect to insert such code, it is useless
; and it hurts the binary size.
;
; CHECK-LABEL: noreturn:
; DISABLE: push
;
; CHECK: cmp r0, #0
; CHECK-NEXT: bne      [[ABORT:LBB[0-9_]+]]
;
; CHECK: movs r0, #42
;
; ENABLE-NEXT: bx lr
;
; DISABLE-NEXT: pop
;;
; CHECK: [[ABORT]]: @ %if.abort
;
; ENABLE: push
;
; CHECK: bl
; ENABLE-NOT: pop
define i32 @noreturn(i8 signext %bad_thing) {
entry:
  %tobool = icmp eq i8 %bad_thing, 0
  br i1 %tobool, label %if.end, label %if.abort

if.abort:
  %call = tail call i32 asm sideeffect "movs $0, #1", "=r,~{r4}"()
  tail call void @abort() #0
  unreachable

if.end:
  ret i32 42
}

declare void @abort() #0

define i32 @b_to_bx(i32 %value) {
; CHECK-LABEL: b_to_bx:
; DISABLE: push {r7, lr}
; CHECK: cmp r0, #49
; CHECK-NEXT: bgt [[ELSE_LABEL:LBB[0-9_]+]]
; ENABLE: push {r7, lr}

; CHECK: bl
; DISABLE-V5-NEXT: pop {r7, pc}
; DISABLE-V4T-NEXT: b [[END_LABEL:LBB[0-9_]+]]

; ENABLE-V5-NEXT: pop {r7, pc}
; ENABLE-V4-NEXT: pop {r7}
; ENABLE-V4-NEXT: pop {r1}
; ENABLE-V4-NEXT: bx r1

; CHECK: [[ELSE_LABEL]]: @ %if.else
; CHECK-NEXT: lsls r0, r1, #1
; DISABLE-V5-NEXT: pop {r7, pc}
; DISABLE-V4T-NEXT: [[END_LABEL]]: @ %if.end
; DISABLE-V4T-NEXT: pop {r7}
; DISABLE-V4T-NEXT: pop {r1}
; DISABLE-V4T-NEXT: bx r1

; ENABLE-V5T-NEXT: {{LBB[0-9_]+}}: @ %if.end
; ENABLE-NEXT: bx lr

entry:
  %cmp = icmp slt i32 %value, 50
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %div = sdiv i32 5000, %value
  br label %if.end

if.else:
  %mul = shl nsw i32 %value, 1
  br label %if.end

if.end:
  %value.addr.0 = phi i32 [ %div, %if.then ], [ %mul, %if.else ]
  ret i32 %value.addr.0
}

define i1 @beq_to_bx(i32* %y, i32 %head) {
; CHECK-LABEL: beq_to_bx:
; DISABLE: push {r4, lr}
; CHECK: cmp r2, #0
; CHECK-NEXT: beq [[EXIT_LABEL:LBB[0-9_]+]]
; ENABLE: push {r4, lr}

; CHECK: lsls    r4, r3, #30
; ENABLE-NEXT: ldr [[POP:r[4567]]], [sp, #4]
; ENABLE-NEXT: mov lr, [[POP]]
; ENABLE-NEXT: pop {[[POP]]}
; ENABLE-NEXT: add sp, #4
; CHECK-NEXT: bpl [[EXIT_LABEL]]

; CHECK: str r1, [r2]
; CHECK: str r3, [r2]
; CHECK-NEXT: movs r0, #0
; CHECK-NEXT: [[EXIT_LABEL]]: @ %cleanup
; ENABLE-NEXT: bx lr
; DISABLE-V5-NEXT: pop {r4, pc}
; DISABLE-V4T-NEXT: pop {r4}
; DISABLE-V4T-NEXT: pop {r1}
; DISABLE-V4T-NEXT: bx r1

entry:
  %cmp = icmp eq i32* %y, null
  br i1 %cmp, label %cleanup, label %if.end

if.end:
  %z = load i32, i32* %y, align 4
  %and = and i32 %z, 2
  %cmp2 = icmp eq i32 %and, 0
  br i1 %cmp2, label %cleanup, label %if.end4

if.end4:
  store i32 %head, i32* %y, align 4
  store volatile i32 %z, i32* %y, align 4
  br label %cleanup

cleanup:
  %retval.0 = phi i1 [ 0, %if.end4 ], [ 1, %entry ], [ 1, %if.end ]
  ret i1 %retval.0
}

attributes #0 = { noreturn nounwind }
