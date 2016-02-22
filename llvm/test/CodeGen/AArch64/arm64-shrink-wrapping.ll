; RUN: llc %s -o - -enable-shrink-wrap=true -disable-post-ra | FileCheck %s --check-prefix=CHECK --check-prefix=ENABLE
; RUN: llc %s -o - -enable-shrink-wrap=false -disable-post-ra | FileCheck %s --check-prefix=CHECK --check-prefix=DISABLE
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios"


; Initial motivating example: Simple diamond with a call just on one side.
; CHECK-LABEL: foo:
;
; Compare the arguments and jump to exit.
; No prologue needed.
; ENABLE: cmp w0, w1
; ENABLE-NEXT: b.ge [[EXIT_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; CHECK: stp [[SAVE_SP:x[0-9]+]], [[CSR:x[0-9]+]], [sp, #-16]!
; CHECK-NEXT: mov [[SAVE_SP]], sp
; CHECK-NEXT: sub sp, sp, #16
;
; Compare the arguments and jump to exit.
; After the prologue is set.
; DISABLE: cmp w0, w1
; DISABLE-NEXT: b.ge [[EXIT_LABEL:LBB[0-9_]+]]
;
; Store %a in the alloca.
; CHECK: stur w0, {{\[}}[[SAVE_SP]], #-4]
; Set the alloca address in the second argument.
; CHECK-NEXT: sub x1, [[SAVE_SP]], #4
; Set the first argument to zero.
; CHECK-NEXT: mov w0, wzr
; CHECK-NEXT: bl _doSomething
; 
; Without shrink-wrapping, epilogue is in the exit block.
; DISABLE: [[EXIT_LABEL]]:
; Epilogue code.
; CHECK-NEXT: mov sp, [[SAVE_SP]]
; CHECK-NEXT: ldp [[SAVE_SP]], [[CSR]], [sp], #16
;
; With shrink-wrapping, exit block is a simple return.
; ENABLE: [[EXIT_LABEL]]:
; CHECK-NEXT: ret
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

; Function Attrs: optsize
declare i32 @doSomething(i32, i32*)


; Check that we do not perform the restore inside the loop whereas the save
; is outside.
; CHECK-LABEL: freqSaveAndRestoreOutsideLoop:
;
; Shrink-wrapping allows to skip the prologue in the else case.
; ENABLE: cbz w0, [[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; CHECK: stp [[CSR1:x[0-9]+]], [[CSR2:x[0-9]+]], [sp, #-32]!
; CHECK-NEXT: stp [[CSR3:x[0-9]+]], [[CSR4:x[0-9]+]], [sp, #16]
; CHECK-NEXT: add [[NEW_SP:x[0-9]+]], sp, #16
;
; DISABLE: cbz w0, [[ELSE_LABEL:LBB[0-9_]+]]
;
; CHECK: mov [[SUM:w[0-9]+]], wzr
; CHECK-NEXT: movz [[IV:w[0-9]+]], #0xa
;
; Next BB.
; CHECK: [[LOOP:LBB[0-9_]+]]: ; %for.body
; CHECK: bl _something
; CHECK-NEXT: add [[SUM]], w0, [[SUM]]
; CHECK-NEXT: sub [[IV]], [[IV]], #1
; CHECK-NEXT: cbnz [[IV]], [[LOOP]]
;
; Next BB.
; Copy SUM into the returned register + << 3.
; CHECK: lsl w0, [[SUM]], #3
;
; Jump to epilogue.
; DISABLE: b [[EPILOG_BB:LBB[0-9_]+]]
;
; DISABLE: [[ELSE_LABEL]]: ; %if.else
; Shift second argument by one and store into returned register.
; DISABLE: lsl w0, w1, #1
; DISABLE: [[EPILOG_BB]]: ; %if.end
;
; Epilogue code.
; CHECK: ldp [[CSR3]], [[CSR4]], [sp, #16]
; CHECK-NEXT: ldp [[CSR1]], [[CSR2]], [sp], #32
; CHECK-NEXT: ret
;
; ENABLE: [[ELSE_LABEL]]: ; %if.else
; Shift second argument by one and store into returned register.
; ENABLE: lsl w0, w1, #1
; ENABLE: ret
define i32 @freqSaveAndRestoreOutsideLoop(i32 %cond, i32 %N) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %sum.04 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %call = tail call i32 bitcast (i32 (...)* @something to i32 ()*)()
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
; CHECK: stp [[CSR1:x[0-9]+]], [[CSR2:x[0-9]+]], [sp, #-32]!
; CHECK-NEXT: stp [[CSR3:x[0-9]+]], [[CSR4:x[0-9]+]], [sp, #16]
; CHECK-NEXT: add [[NEW_SP:x[0-9]+]], sp, #16
; CHECK: mov [[SUM:w[0-9]+]], wzr
; CHECK-NEXT: movz [[IV:w[0-9]+]], #0xa
; Next BB.
; CHECK: [[LOOP_LABEL:LBB[0-9_]+]]: ; %for.body
; CHECK: bl _something
; CHECK-NEXT: add [[SUM]], w0, [[SUM]]
; CHECK-NEXT: sub [[IV]], [[IV]], #1
; CHECK-NEXT: cbnz [[IV]], [[LOOP_LABEL]]
; Next BB.
; CHECK: ; %for.end
; CHECK: mov w0, [[SUM]]
; CHECK-NEXT: ldp [[CSR3]], [[CSR4]], [sp, #16]
; CHECK-NEXT: ldp [[CSR1]], [[CSR2]], [sp], #32
; CHECK-NEXT: ret
define i32 @freqSaveAndRestoreOutsideLoop2(i32 %cond) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.04 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum.03 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %call = tail call i32 bitcast (i32 (...)* @something to i32 ()*)()
  %add = add nsw i32 %call, %sum.03
  %inc = add nuw nsw i32 %i.04, 1
  %exitcond = icmp eq i32 %inc, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 %add
}

; Check with a more complex case that we do not have save within the loop and
; restore outside.
; CHECK-LABEL: loopInfoSaveOutsideLoop:
;
; ENABLE: cbz w0, [[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; CHECK: stp [[CSR1:x[0-9]+]], [[CSR2:x[0-9]+]], [sp, #-32]!
; CHECK-NEXT: stp [[CSR3:x[0-9]+]], [[CSR4:x[0-9]+]], [sp, #16]
; CHECK-NEXT: add [[NEW_SP:x[0-9]+]], sp, #16
;
; DISABLE: cbz w0, [[ELSE_LABEL:LBB[0-9_]+]]
;
; CHECK: mov [[SUM:w[0-9]+]], wzr
; CHECK-NEXT: movz [[IV:w[0-9]+]], #0xa
;
; CHECK: [[LOOP_LABEL:LBB[0-9_]+]]: ; %for.body
; CHECK: bl _something
; CHECK-NEXT: add [[SUM]], w0, [[SUM]]
; CHECK-NEXT: sub [[IV]], [[IV]], #1
; CHECK-NEXT: cbnz [[IV]], [[LOOP_LABEL]]
; Next BB.
; CHECK: bl _somethingElse
; CHECK-NEXT: lsl w0, [[SUM]], #3
;
; Jump to epilogue.
; DISABLE: b [[EPILOG_BB:LBB[0-9_]+]]
;
; DISABLE: [[ELSE_LABEL]]: ; %if.else
; Shift second argument by one and store into returned register.
; DISABLE: lsl w0, w1, #1
; DISABLE: [[EPILOG_BB]]: ; %if.end
; Epilogue code.
; CHECK-NEXT: ldp [[CSR3]], [[CSR4]], [sp, #16]
; CHECK-NEXT: ldp [[CSR1]], [[CSR2]], [sp], #32
; CHECK-NEXT: ret
;
; ENABLE: [[ELSE_LABEL]]: ; %if.else
; Shift second argument by one and store into returned register.
; ENABLE: lsl w0, w1, #1
; ENABLE: ret
define i32 @loopInfoSaveOutsideLoop(i32 %cond, i32 %N) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %sum.04 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %call = tail call i32 bitcast (i32 (...)* @something to i32 ()*)()
  %add = add nsw i32 %call, %sum.04
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  tail call void bitcast (void (...)* @somethingElse to void ()*)()
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
; ENABLE: cbz w0, [[ELSE_LABEL:LBB[0-9_]+]]
;
; CHECK: stp [[CSR1:x[0-9]+]], [[CSR2:x[0-9]+]], [sp, #-32]!
; CHECK-NEXT: stp [[CSR3:x[0-9]+]], [[CSR4:x[0-9]+]], [sp, #16]
; CHECK-NEXT: add [[NEW_SP:x[0-9]+]], sp, #16
;
; DISABLE: cbz w0, [[ELSE_LABEL:LBB[0-9_]+]]
;
; CHECK: bl _somethingElse
; CHECK-NEXT: mov [[SUM:w[0-9]+]], wzr
; CHECK-NEXT: movz [[IV:w[0-9]+]], #0xa
;
; CHECK: [[LOOP_LABEL:LBB[0-9_]+]]: ; %for.body
; CHECK: bl _something
; CHECK-NEXT: add [[SUM]], w0, [[SUM]]
; CHECK-NEXT: sub [[IV]], [[IV]], #1
; CHECK-NEXT: cbnz [[IV]], [[LOOP_LABEL]]
; Next BB.
; CHECK: lsl w0, [[SUM]], #3
;
; Jump to epilogue.
; DISABLE: b [[EPILOG_BB:LBB[0-9_]+]]
;
; DISABLE: [[ELSE_LABEL]]: ; %if.else
; Shift second argument by one and store into returned register.
; DISABLE: lsl w0, w1, #1
; DISABLE: [[EPILOG_BB]]: ; %if.end
; Epilogue code.
; CHECK: ldp [[CSR3]], [[CSR4]], [sp, #16]
; CHECK-NEXT: ldp [[CSR1]], [[CSR2]], [sp], #32
; CHECK-NEXT: ret
;
; ENABLE: [[ELSE_LABEL]]: ; %if.else
; Shift second argument by one and store into returned register.
; ENABLE: lsl w0, w1, #1
; ENABLE: ret
define i32 @loopInfoRestoreOutsideLoop(i32 %cond, i32 %N) #0 {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  tail call void bitcast (void (...)* @somethingElse to void ()*)()
  br label %for.body

for.body:                                         ; preds = %for.body, %if.then
  %i.05 = phi i32 [ 0, %if.then ], [ %inc, %for.body ]
  %sum.04 = phi i32 [ 0, %if.then ], [ %add, %for.body ]
  %call = tail call i32 bitcast (i32 (...)* @something to i32 ()*)()
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
; CHECK: ; %entry
; CHECK-NEXT: mov w0, wzr
; CHECK-NEXT: ret
define i32 @emptyFrame() {
entry:
  ret i32 0
}

; Check that we handle variadic function correctly.
; CHECK-LABEL: variadicFunc:
;
; ENABLE: cbz w0, [[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; CHECK: sub sp, sp, #16
; DISABLE: cbz w0, [[ELSE_LABEL:LBB[0-9_]+]]
;
; Sum is merged with the returned register.
; CHECK: mov [[SUM:w0]], wzr
; CHECK-NEXT: add [[VA_BASE:x[0-9]+]], sp, #16
; CHECK-NEXT: str [[VA_BASE]], [sp, #8]
; CHECK-NEXT: cmp w1, #1
; CHECK-NEXT: b.lt [[IFEND_LABEL:LBB[0-9_]+]]
;
; CHECK: [[LOOP_LABEL:LBB[0-9_]+]]: ; %for.body
; CHECK: ldr [[VA_ADDR:x[0-9]+]], [sp, #8]
; CHECK-NEXT: add [[NEXT_VA_ADDR:x[0-9]+]], [[VA_ADDR]], #8
; CHECK-NEXT: str [[NEXT_VA_ADDR]], [sp, #8]
; CHECK-NEXT: ldr [[VA_VAL:w[0-9]+]], {{\[}}[[VA_ADDR]]]
; CHECK-NEXT: add [[SUM]], [[SUM]], [[VA_VAL]]
; CHECK-NEXT: sub w1, w1, #1
; CHECK-NEXT: cbnz w1, [[LOOP_LABEL]]
;
; DISABLE-NEXT: b [[IFEND_LABEL]]
; DISABLE: [[ELSE_LABEL]]: ; %if.else
; DISABLE: lsl w0, w1, #1
;
; CHECK: [[IFEND_LABEL]]:
; Epilogue code.
; CHECK: add sp, sp, #16
; CHECK-NEXT: ret
;
; ENABLE: [[ELSE_LABEL]]: ; %if.else
; ENABLE: lsl w0, w1, #1
; ENABLE-NEXT: ret
define i32 @variadicFunc(i32 %cond, i32 %count, ...) #0 {
entry:
  %ap = alloca i8*, align 8
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %cmp6 = icmp sgt i32 %count, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %if.then, %for.body
  %i.08 = phi i32 [ %inc, %for.body ], [ 0, %if.then ]
  %sum.07 = phi i32 [ %add, %for.body ], [ 0, %if.then ]
  %0 = va_arg i8** %ap, i32
  %add = add nsw i32 %sum.07, %0
  %inc = add nuw nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %inc, %count
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %if.then
  %sum.0.lcssa = phi i32 [ 0, %if.then ], [ %add, %for.body ]
  call void @llvm.va_end(i8* %ap1)
  br label %if.end

if.else:                                          ; preds = %entry
  %mul = shl nsw i32 %count, 1
  br label %if.end

if.end:                                           ; preds = %if.else, %for.end
  %sum.1 = phi i32 [ %sum.0.lcssa, %for.end ], [ %mul, %if.else ]
  ret i32 %sum.1
}

declare void @llvm.va_start(i8*)

declare void @llvm.va_end(i8*)

; Check that we handle inline asm correctly.
; CHECK-LABEL: inlineAsm:
;
; ENABLE: cbz w0, [[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the CSR used in the inline asm: x19.
; CHECK: stp [[CSR1:x[0-9]+]], [[CSR2:x19]], [sp, #-16]!
;
; DISABLE: cbz w0, [[ELSE_LABEL:LBB[0-9_]+]]
;
; CHECK: movz [[IV:w[0-9]+]], #0xa
;
; CHECK: [[LOOP_LABEL:LBB[0-9_]+]]: ; %for.body
; Inline asm statement.
; CHECK: add x19, x19, #1
; CHECK: sub [[IV]], [[IV]], #1
; CHECK-NEXT: cbnz [[IV]], [[LOOP_LABEL]]
; Next BB.
; CHECK: mov w0, wzr
; Epilogue code.
; CHECK-NEXT: ldp [[CSR1]], [[CSR2]], [sp], #16
; CHECK-NEXT: ret
; Next BB.
; CHECK: [[ELSE_LABEL]]: ; %if.else
; CHECK-NEXT: lsl w0, w1, #1
; Epilogue code.
; DISABLE-NEXT: ldp [[CSR1]], [[CSR2]], [sp], #16
; CHECK-NEXT: ret
define i32 @inlineAsm(i32 %cond, i32 %N) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.03 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  tail call void asm sideeffect "add x19, x19, #1", "~{x19}"()
  %inc = add nuw nsw i32 %i.03, 1
  %exitcond = icmp eq i32 %inc, 10
  br i1 %exitcond, label %if.end, label %for.body

if.else:                                          ; preds = %entry
  %mul = shl nsw i32 %N, 1
  br label %if.end

if.end:                                           ; preds = %for.body, %if.else
  %sum.0 = phi i32 [ %mul, %if.else ], [ 0, %for.body ]
  ret i32 %sum.0
}

; Check that we handle calls to variadic functions correctly.
; CHECK-LABEL: callVariadicFunc:
;
; ENABLE: cbz w0, [[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; CHECK: stp [[CSR1:x[0-9]+]], [[CSR2:x[0-9]+]], [sp, #-16]!
; CHECK-NEXT: mov [[NEW_SP:x[0-9]+]], sp
; CHECK-NEXT: sub sp, sp, #48
;
; DISABLE: cbz w0, [[ELSE_LABEL:LBB[0-9_]+]]
; Setup of the varags.
; CHECK: stp x1, x1, [sp, #32]
; CHECK-NEXT: stp x1, x1, [sp, #16]
; CHECK-NEXT: stp x1, x1, [sp]
; CHECK-NEXT: mov w0, w1
; CHECK-NEXT: bl _someVariadicFunc
; CHECK-NEXT: lsl w0, w0, #3
;
; DISABLE: b [[IFEND_LABEL:LBB[0-9_]+]]
; DISABLE: [[ELSE_LABEL]]: ; %if.else
; DISABLE-NEXT: lsl w0, w1, #1
; DISABLE: [[IFEND_LABEL]]: ; %if.end
;
; Epilogue code.
; CHECK: mov sp, [[NEW_SP]]
; CHECK-NEXT: ldp [[CSR1]], [[CSR2]], [sp], #16
; CHECK-NEXT: ret
;
; ENABLE: [[ELSE_LABEL]]: ; %if.else
; ENABLE-NEXT: lsl w0, w1, #1
; ENABLE-NEXT: ret
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
; DISABLE: stp
;
; CHECK: and [[TEST:w[0-9]+]], w0, #0xff
; CHECK-NEXT: cbnz [[TEST]], [[ABORT:LBB[0-9_]+]]
;
; CHECK: movz w0, #0x2a
;
; DISABLE-NEXT: ldp
;
; CHECK-NEXT: ret
;
; CHECK: [[ABORT]]: ; %if.abort
;
; ENABLE: stp
;
; CHECK: bl _abort
; ENABLE-NOT: ldp
define i32 @noreturn(i8 signext %bad_thing) {
entry:
  %tobool = icmp eq i8 %bad_thing, 0
  br i1 %tobool, label %if.end, label %if.abort

if.abort:
  tail call void @abort() #0
  unreachable

if.end:
  ret i32 42
}

declare void @abort() #0

attributes #0 = { noreturn nounwind }

; Make sure that we handle infinite loops properly When checking that the Save
; and Restore blocks are control flow equivalent, the loop searches for the
; immediate (post) dominator for the (restore) save blocks. When either the Save
; or Restore block is located in an infinite loop the only immediate (post)
; dominator is itself. In this case, we cannot perform shrink wrapping, but we
; should return gracefully and continue compilation.
; The only condition for this test is the compilation finishes correctly.
;
; CHECK-LABEL: infiniteloop
; CHECK: ret
define void @infiniteloop() {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:
  %ptr = alloca i32, i32 4
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %sum.03 = phi i32 [ 0, %if.then ], [ %add, %for.body ]
  %call = tail call i32 bitcast (i32 (...)* @something to i32 ()*)()
  %add = add nsw i32 %call, %sum.03
  store i32 %add, i32* %ptr
  br label %for.body

if.end:
  ret void
}

; Another infinite loop test this time with a body bigger than just one block.
; CHECK-LABEL: infiniteloop2
; CHECK: ret
define void @infiniteloop2() {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:
  %ptr = alloca i32, i32 4
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %sum.03 = phi i32 [ 0, %if.then ], [ %add, %body1 ], [ 1, %body2]
  %call = tail call i32 asm "mov $0, #0", "=r,~{x19}"()
  %add = add nsw i32 %call, %sum.03
  store i32 %add, i32* %ptr
  br i1 undef, label %body1, label %body2

body1:
  tail call void asm sideeffect "nop", "~{x19}"()
  br label %for.body

body2:
  tail call void asm sideeffect "nop", "~{x19}"()
  br label %for.body

if.end:
  ret void
}

; Another infinite loop test this time with two nested infinite loop.
; CHECK-LABEL: infiniteloop3
; CHECK: ret
define void @infiniteloop3() {
entry:
  br i1 undef, label %loop2a, label %body

body:                                             ; preds = %entry
  br i1 undef, label %loop2a, label %end

loop1:                                            ; preds = %loop2a, %loop2b
  %var.phi = phi i32* [ %next.phi, %loop2b ], [ %var, %loop2a ]
  %next.phi = phi i32* [ %next.load, %loop2b ], [ %next.var, %loop2a ]
  %0 = icmp eq i32* %var, null
  %next.load = load i32*, i32** undef
  br i1 %0, label %loop2a, label %loop2b

loop2a:                                           ; preds = %loop1, %body, %entry
  %var = phi i32* [ null, %body ], [ null, %entry ], [ %next.phi, %loop1 ]
  %next.var = phi i32* [ undef, %body ], [ null, %entry ], [ %next.load, %loop1 ]
  br label %loop1

loop2b:                                           ; preds = %loop1
  %gep1 = bitcast i32* %var.phi to i32*
  %next.ptr = bitcast i32* %gep1 to i32**
  store i32* %next.phi, i32** %next.ptr
  br label %loop1

end:
  ret void
}

; Don't do shrink-wrapping when we need to re-align the stack pointer.
; See bug 26642.
; CHECK-LABEL: stack_realign:
; CHECK-NOT: lsl w[[LSL1:[0-9]+]], w0, w1
; CHECK-NOT: lsl w[[LSL2:[0-9]+]], w1, w0
; CHECK: stp x29, x30, [sp, #-16]!
; CHECK: mov x29, sp
; CHECK: sub x{{[0-9]+}}, sp, #16
; CHECK-DAG: lsl w[[LSL1:[0-9]+]], w0, w1
; CHECK-DAG: lsl w[[LSL2:[0-9]+]], w1, w0
; CHECK-DAG: str w[[LSL1]],
; CHECK-DAG: str w[[LSL2]],

define i32 @stack_realign(i32 %a, i32 %b, i32* %ptr1, i32* %ptr2) {
  %tmp = alloca i32, align 32
  %shl1 = shl i32 %a, %b
  %shl2 = shl i32 %b, %a
  %tmp2 = icmp slt i32 %a, %b
  br i1 %tmp2, label %true, label %false

true:
  store i32 %a, i32* %tmp, align 4
  %tmp4 = load i32, i32* %tmp
  br label %false

false:
  %tmp.0 = phi i32 [ %tmp4, %true ], [ %a, %0 ]
  store i32 %shl1, i32* %ptr1
  store i32 %shl2, i32* %ptr2
  ret i32 %tmp.0
}

; Re-aligned stack pointer with all caller-save regs live.  See bug
; 26642.  In this case we currently avoid shrink wrapping because
; ensuring we have a scratch register to re-align the stack pointer is
; too complicated.  Output should be the same for both enabled and
; disabled shrink wrapping.
; CHECK-LABEL: stack_realign2:
; CHECK: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #-{{[0-9]+}}]!
; CHECK: add x29, sp, #{{[0-9]+}}
; CHECK: lsl {{w[0-9]+}}, w0, w1

define void @stack_realign2(i32 %a, i32 %b, i32* %ptr1, i32* %ptr2, i32* %ptr3, i32* %ptr4, i32* %ptr5, i32* %ptr6) {
  %tmp = alloca i32, align 32
  %tmp1 = shl i32 %a, %b
  %tmp2 = shl i32 %b, %a
  %tmp3 = lshr i32 %a, %b
  %tmp4 = lshr i32 %b, %a
  %tmp5 = add i32 %b, %a
  %tmp6 = sub i32 %b, %a
  %tmp7 = add i32 %tmp1, %tmp2
  %tmp8 = sub i32 %tmp2, %tmp3
  %tmp9 = add i32 %tmp3, %tmp4
  %tmp10 = add i32 %tmp4, %tmp5
  %cmp = icmp slt i32 %a, %b
  br i1 %cmp, label %true, label %false

true:
  store i32 %a, i32* %tmp, align 4
  call void asm sideeffect "nop", "~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28}"() nounwind
  br label %false

false:
  store i32 %tmp1, i32* %ptr1, align 4
  store i32 %tmp2, i32* %ptr2, align 4
  store i32 %tmp3, i32* %ptr3, align 4
  store i32 %tmp4, i32* %ptr4, align 4
  store i32 %tmp5, i32* %ptr5, align 4
  store i32 %tmp6, i32* %ptr6, align 4
  %idx1 = getelementptr inbounds i32, i32* %ptr1, i64 1
  store i32 %a, i32* %idx1, align 4
  %idx2 = getelementptr inbounds i32, i32* %ptr1, i64 2
  store i32 %b, i32* %idx2, align 4
  %idx3 = getelementptr inbounds i32, i32* %ptr1, i64 3
  store i32 %tmp7, i32* %idx3, align 4
  %idx4 = getelementptr inbounds i32, i32* %ptr1, i64 4
  store i32 %tmp8, i32* %idx4, align 4
  %idx5 = getelementptr inbounds i32, i32* %ptr1, i64 5
  store i32 %tmp9, i32* %idx5, align 4
  %idx6 = getelementptr inbounds i32, i32* %ptr1, i64 6
  store i32 %tmp10, i32* %idx6, align 4

  ret void
}
