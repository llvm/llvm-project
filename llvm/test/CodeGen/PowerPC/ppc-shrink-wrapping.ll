; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=ENABLE
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu %s -o - -enable-shrink-wrap=false |  FileCheck %s --check-prefix=CHECK --check-prefix=DISABLE
;
; Note: Lots of tests use inline asm instead of regular calls.
; This allows to have a better control on what the allocation will do.
; Otherwise, we may have spill right in the entry block, defeating
; shrink-wrapping. Moreover, some of the inline asm statement (nop)
; are here to ensure that the related paths do not end up as critical
; edges.


; Initial motivating example: Simple diamond with a call just on one side.
; CHECK-LABEL: foo:
;
; Compare the arguments and return
; No prologue needed.
; ENABLE: cmpw 0, 3, 4
; ENABLE-NEXT: bgelr 0
;
; Prologue code.
;  At a minimum, we save/restore the link register. Other registers may be saved
;  as well. 
; CHECK: mflr 
;
; Compare the arguments and jump to exit.
; After the prologue is set.
; DISABLE: cmpw 0, 3, 4
; DISABLE-NEXT: bge 0, .[[EXIT_LABEL:LBB[0-9_]+]]
;
; Store %a on the stack
; CHECK: stw 3, {{[0-9]+([0-9]+)}}
; Set the alloca address in the second argument.
; CHECK-NEXT: addi 4, 1, {{[0-9]+}}
; Set the first argument to zero.
; CHECK-NEXT: li 3, 0
; CHECK-NEXT: bl doSomething
;
; With shrink-wrapping, epilogue is just after the call.
; Restore the link register and return.
; Note that there could be other epilog code before the link register is 
; restored but we will not check for it here.
; ENABLE: mtlr
; ENABLE-NEXT: blr
;
; DISABLE: [[EXIT_LABEL]]:
;
; Without shrink-wrapping, epilogue is in the exit block.
; Epilogue code. (What we pop does not matter.)
; DISABLE: mtlr {{[0-9]+}}
; DISABLE-NEXT: blr
;

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
; ENABLE: cmplwi 0, 3, 0
; ENABLE: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the link register
; CHECK: mflr {{[0-9]+}}
;
; DISABLE: cmplwi 0, 3, 0
; DISABLE: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; Loop preheader
; CHECK-DAG: li [[SUM:[0-9]+]], 0
; CHECK-DAG: li [[IV:[0-9]+]], 10
; 
; Loop body
; CHECK: .[[LOOP:LBB[0-9_]+]]: # %for.body
; CHECK: bl something
; CHECK-DAG: addi [[IV]], [[IV]], -1
; CHECK-DAG: add [[SUM]], 3, [[SUM]] 
; CHECK-NEXT: cmplwi [[IV]], 0
; CHECK-NEXT: bne 0, .[[LOOP]]
;
; Next BB.
; CHECK: slwi 3, [[SUM]], 3
;
; Jump to epilogue.
; DISABLE: b .[[EPILOG_BB:LBB[0-9_]+]]
;
; DISABLE: .[[ELSE_LABEL]]: # %if.else
; Shift second argument by one and store into returned register.
; DISABLE: slwi 3, 4, 1
; DISABLE: .[[EPILOG_BB]]: # %if.end
;
; Epilogue code.
; CHECK: mtlr {{[0-9]+}}
; CHECK-NEXT: blr
;
; ENABLE: .[[ELSE_LABEL]]: # %if.else
; Shift second argument by one and store into returned register.
; ENABLE: slwi 3, 4, 1
; ENABLE-NEXT: blr
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
; Make sure we save the link register before the call
; CHECK: mflr {{[0-9]+}}
;
; Loop preheader
; CHECK-DAG: li [[SUM:[0-9]+]], 0
; CHECK-DAG: li [[IV:[0-9]+]], 10
; 
; Loop body
; CHECK: .[[LOOP:LBB[0-9_]+]]: # %for.body
; CHECK: bl something
; CHECK-DAG: addi [[IV]], [[IV]], -1
; CHECK-DAG: add [[SUM]], 3, [[SUM]] 
; CHECK-NEXT: cmplwi [[IV]], 0
; CHECK-NEXT: bne 0, .[[LOOP]]
;
; Next BB
; CHECK: %for.exit
; CHECK: mtlr {{[0-9]+}}
; CHECK-NEXT: blr
define i32 @freqSaveAndRestoreOutsideLoop2(i32 %cond) {
entry:
  br label %for.preheader

for.preheader:
  tail call void asm "nop", ""()
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.04 = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]
  %sum.03 = phi i32 [ 0, %for.preheader ], [ %add, %for.body ]
  %call = tail call i32 bitcast (i32 (...)* @something to i32 ()*)()
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
; ENABLE: cmplwi 0, 3, 0
; ENABLE-NEXT: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the link register 
; CHECK: mflr {{[0-9]+}}
;
; DISABLE: cmplwi 0, 3, 0
; DISABLE-NEXT: std
; DISABLE-NEXT: std
; DISABLE-NEXT: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; Loop preheader
; CHECK-DAG: li [[SUM:[0-9]+]], 0
; CHECK-DAG: li [[IV:[0-9]+]], 10
; 
; Loop body
; CHECK: .[[LOOP:LBB[0-9_]+]]: # %for.body
; CHECK: bl something
; CHECK-DAG: addi [[IV]], [[IV]], -1
; CHECK-DAG: add [[SUM]], 3, [[SUM]] 
; CHECK-NEXT: cmplwi [[IV]], 0
; CHECK-NEXT: bne 0, .[[LOOP]]
; 
; Next BB
; CHECK: bl somethingElse 
; CHECK: slwi 3, [[SUM]], 3
;
; Jump to epilogue
; DISABLE: b .[[EPILOG_BB:LBB[0-9_]+]]
;
; DISABLE: .[[ELSE_LABEL]]: # %if.else
; Shift second argument by one and store into returned register.
; DISABLE: slwi 3, 4, 1
;
; DISABLE: .[[EPILOG_BB]]: # %if.end
; Epilog code
; CHECK: mtlr {{[0-9]+}}
; CHECK-NEXT: blr
; 
; ENABLE: .[[ELSE_LABEL]]: # %if.else
; Shift second argument by one and store into returned register.
; ENABLE: slwi 3, 4, 1
; ENABLE-NEXT: blr
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
; ENABLE: cmplwi 0, 3, 0
; ENABLE-NEXT: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the link register
; CHECK: mflr {{[0-9]+}}
;
; DISABLE: cmplwi 0, 3, 0
; DISABLE-NEXT: std
; DISABLE-NEXT: std
; DISABLE-NEXT: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; CHECK: bl somethingElse
;
; Loop preheader
; CHECK-DAG: li [[SUM:[0-9]+]], 0
; CHECK-DAG: li [[IV:[0-9]+]], 10
; 
; Loop body
; CHECK: .[[LOOP:LBB[0-9_]+]]: # %for.body
; CHECK: bl something
; CHECK-DAG: addi [[IV]], [[IV]], -1
; CHECK-DAG: add [[SUM]], 3, [[SUM]] 
; CHECK-NEXT: cmplwi [[IV]], 0
; CHECK-NEXT: bne 0, .[[LOOP]]
;
; Next BB. 
; slwi 3, [[SUM]], 3
;
; DISABLE: b .[[EPILOG_BB:LBB[0-9_]+]]
;
; DISABLE: .[[ELSE_LABEL]]: # %if.else
; Shift second argument by one and store into returned register.
; DISABLE: slwi 3, 4, 1
; DISABLE: .[[EPILOG_BB]]: # %if.end
;
; Epilogue code.
; CHECK: mtlr {{[0-9]+}}
; CHECK-NEXT: blr
;
; ENABLE: .[[ELSE_LABEL]]: # %if.else
; Shift second argument by one and store into returned register.
; ENABLE: slwi 3, 4, 1
; ENABLE-NEXT: blr
define i32 @loopInfoRestoreOutsideLoop(i32 %cond, i32 %N) nounwind {
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
; CHECK: # %entry
; CHECK-NEXT: li 3, 0
; CHECK-NEXT: blr
define i32 @emptyFrame() {
entry:
  ret i32 0
}


; Check that we handle inline asm correctly.
; CHECK-LABEL: inlineAsm:
;
; ENABLE: cmplwi 0, 3, 0
; ENABLE-NEXT: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the CSR used in the inline asm: r14
; ENABLE-DAG: li [[IV:[0-9]+]], 10
; ENABLE-DAG: std 14, -[[STACK_OFFSET:[0-9]+]](1) # 8-byte Folded Spill
;
; DISABLE: cmplwi 0, 3, 0
; DISABLE-NEXT: std 14, -[[STACK_OFFSET:[0-9]+]](1) # 8-byte Folded Spill
; DISABLE-NEXT: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
; DISABLE: li [[IV:[0-9]+]], 10
;
; CHECK: nop
; CHECK: mtctr [[IV]]
;
; CHECK: .[[LOOP_LABEL:LBB[0-9_]+]]: # %for.body
; Inline asm statement.
; CHECK: addi 14, 14, 1
; CHECK: bdnz .[[LOOP_LABEL]]
;
; Epilogue code.
; CHECK: li 3, 0
; CHECK-DAG: ld 14, -[[STACK_OFFSET]](1) # 8-byte Folded Reload
; CHECK: nop
; CHECK: blr
;
; CHECK: [[ELSE_LABEL]]
; CHECK-NEXT: slwi 3, 4, 1
; DISABLE: ld 14, -[[STACK_OFFSET]](1) # 8-byte Folded Reload
; CHECK-NEXT: blr
; 
define i32 @inlineAsm(i32 %cond, i32 %N) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %for.preheader

for.preheader:
  tail call void asm "nop", ""()
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.03 = phi i32 [ %inc, %for.body ], [ 0, %for.preheader ]
  tail call void asm "addi 14, 14, 1", "~{r14}"()
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
; ENABLE: cmplwi 0, 3, 0
; ENABLE-NEXT: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; CHECK: mflr {{[0-9]+}}
; 
; DISABLE: cmplwi 0, 3, 0
; DISABLE-NEXT: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; Setup of the varags.
; CHECK: mr 4, 3
; CHECK-NEXT: mr 5, 3
; CHECK-NEXT: mr 6, 3
; CHECK-NEXT: mr 7, 3
; CHECK-NEXT: mr 8, 3
; CHECK-NEXT: mr 9, 3
; CHECK-NEXT: bl someVariadicFunc
; CHECK: slwi 3, 3, 3
; DISABLE: b .[[EPILOGUE_BB:LBB[0-9_]+]]
;
; ENABLE: mtlr {{[0-9]+}}
; ENABLE-NEXT: blr
;
; CHECK: .[[ELSE_LABEL]]: # %if.else
; CHECK-NEXT: slwi 3, 4, 1
; 
; DISABLE: .[[EPILOGUE_BB]]: # %if.end
; DISABLE: mtlr
; CHECK: blr
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
; DISABLE: mflr {{[0-9]+}}
;
; CHECK: cmplwi 0, 3, 0
; CHECK-NEXT: bne{{[-]?}} 0, .[[ABORT:LBB[0-9_]+]]
;
; CHECK: li 3, 42
;
; DISABLE: mtlr {{[0-9]+}}
;
; CHECK-NEXT: blr
;
; CHECK: .[[ABORT]]: # %if.abort
;
; ENABLE: mflr {{[0-9]+}}
;
; CHECK: bl abort
; ENABLE-NOT: mtlr {{[0-9]+}}
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
; CHECK: blr
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
; CHECK: blr
define void @infiniteloop2() {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:
  %ptr = alloca i32, i32 4
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %sum.03 = phi i32 [ 0, %if.then ], [ %add, %body1 ], [ 1, %body2]
  %call = tail call i32 asm "mftb $0, 268", "=r,~{r14}"()
  %add = add nsw i32 %call, %sum.03
  store i32 %add, i32* %ptr
  br i1 undef, label %body1, label %body2

body1:
  tail call void asm sideeffect "nop", "~{r14}"()
  br label %for.body

body2:
  tail call void asm sideeffect "nop", "~{r14}"()
  br label %for.body

if.end:
  ret void
}

; Another infinite loop test this time with two nested infinite loop.
; CHECK-LABEL: infiniteloop3
; CHECK: Lfunc_begin[[FUNCNUM:[0-9]+]]
; CHECK: bclr
; CHECK: Lfunc_end[[FUNCNUM]]
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

@columns = external global [0 x i32], align 4
@lock = common global i32 0, align 4
@htindex = common global i32 0, align 4
@stride = common global i32 0, align 4
@ht = common global i32* null, align 8
@he = common global i8* null, align 8

; Test for a bug that was caused when save point was equal to restore point.
; Function Attrs: nounwind
; CHECK-LABEL: transpose
;
; Store of callee-save register saved by shrink wrapping
; FIXME: Test disabled: Improved scheduling needs no spills/reloads any longer!
; CHECKXX: std [[CSR:[0-9]+]], -[[STACK_OFFSET:[0-9]+]](1) # 8-byte Folded Spill
;
; Reload of callee-save register
; CHECKXX: ld [[CSR]], -[[STACK_OFFSET]](1) # 8-byte Folded Reload
;
; Ensure no subsequent uses of callee-save register before end of function
; CHECK-NOT: {{[a-z]+}} [[CSR]]
; CHECK: blr
define signext i32 @transpose() {
entry:
  %0 = load i32, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @columns, i64 0, i64 1), align 4
  %shl.i = shl i32 %0, 7
  %1 = load i32, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @columns, i64 0, i64 2), align 4
  %or.i = or i32 %shl.i, %1
  %shl1.i = shl i32 %or.i, 7
  %2 = load i32, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @columns, i64 0, i64 3), align 4
  %or2.i = or i32 %shl1.i, %2
  %3 = load i32, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @columns, i64 0, i64 7), align 4
  %shl3.i = shl i32 %3, 7
  %4 = load i32, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @columns, i64 0, i64 6), align 4
  %or4.i = or i32 %shl3.i, %4
  %shl5.i = shl i32 %or4.i, 7
  %5 = load i32, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @columns, i64 0, i64 5), align 4
  %or6.i = or i32 %shl5.i, %5
  %cmp.i = icmp ugt i32 %or2.i, %or6.i
  br i1 %cmp.i, label %cond.true.i, label %cond.false.i

cond.true.i:
  %shl7.i = shl i32 %or2.i, 7
  %6 = load i32, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @columns, i64 0, i64 4), align 4
  %or8.i = or i32 %6, %shl7.i
  %conv.i = zext i32 %or8.i to i64
  %shl9.i = shl nuw nsw i64 %conv.i, 21
  %conv10.i = zext i32 %or6.i to i64
  %or11.i = or i64 %shl9.i, %conv10.i
  br label %hash.exit

cond.false.i:
  %shl12.i = shl i32 %or6.i, 7
  %7 = load i32, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @columns, i64 0, i64 4), align 4
  %or13.i = or i32 %7, %shl12.i
  %conv14.i = zext i32 %or13.i to i64
  %shl15.i = shl nuw nsw i64 %conv14.i, 21
  %conv16.i = zext i32 %or2.i to i64
  %or17.i = or i64 %shl15.i, %conv16.i
  br label %hash.exit

hash.exit:
  %cond.i = phi i64 [ %or11.i, %cond.true.i ], [ %or17.i, %cond.false.i ]
  %shr.29.i = lshr i64 %cond.i, 17
  %conv18.i = trunc i64 %shr.29.i to i32
  store i32 %conv18.i, i32* @lock, align 4
  %rem.i = srem i64 %cond.i, 1050011
  %conv19.i = trunc i64 %rem.i to i32
  store i32 %conv19.i, i32* @htindex, align 4
  %rem20.i = urem i32 %conv18.i, 179
  %add.i = or i32 %rem20.i, 131072
  store i32 %add.i, i32* @stride, align 4
  %8 = load i32*, i32** @ht, align 8
  %arrayidx = getelementptr inbounds i32, i32* %8, i64 %rem.i
  %9 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp eq i32 %9, %conv18.i
  br i1 %cmp1, label %if.then, label %if.end

if.then:
  %idxprom.lcssa = phi i64 [ %rem.i, %hash.exit ], [ %idxprom.1, %if.end ], [ %idxprom.2, %if.end.1 ], [ %idxprom.3, %if.end.2 ], [ %idxprom.4, %if.end.3 ], [ %idxprom.5, %if.end.4 ], [ %idxprom.6, %if.end.5 ], [ %idxprom.7, %if.end.6 ]
  %10 = load i8*, i8** @he, align 8
  %arrayidx3 = getelementptr inbounds i8, i8* %10, i64 %idxprom.lcssa
  %11 = load i8, i8* %arrayidx3, align 1
  %conv = sext i8 %11 to i32
  br label %cleanup

if.end:
  %add = add nsw i32 %add.i, %conv19.i
  %cmp4 = icmp sgt i32 %add, 1050010
  %sub = add nsw i32 %add, -1050011
  %sub.add = select i1 %cmp4, i32 %sub, i32 %add
  %idxprom.1 = sext i32 %sub.add to i64
  %arrayidx.1 = getelementptr inbounds i32, i32* %8, i64 %idxprom.1
  %12 = load i32, i32* %arrayidx.1, align 4
  %cmp1.1 = icmp eq i32 %12, %conv18.i
  br i1 %cmp1.1, label %if.then, label %if.end.1

cleanup:
  %retval.0 = phi i32 [ %conv, %if.then ], [ -128, %if.end.6 ]
  ret i32 %retval.0

if.end.1:
  %add.1 = add nsw i32 %add.i, %sub.add
  %cmp4.1 = icmp sgt i32 %add.1, 1050010
  %sub.1 = add nsw i32 %add.1, -1050011
  %sub.add.1 = select i1 %cmp4.1, i32 %sub.1, i32 %add.1
  %idxprom.2 = sext i32 %sub.add.1 to i64
  %arrayidx.2 = getelementptr inbounds i32, i32* %8, i64 %idxprom.2
  %13 = load i32, i32* %arrayidx.2, align 4
  %cmp1.2 = icmp eq i32 %13, %conv18.i
  br i1 %cmp1.2, label %if.then, label %if.end.2

if.end.2:
  %add.2 = add nsw i32 %add.i, %sub.add.1
  %cmp4.2 = icmp sgt i32 %add.2, 1050010
  %sub.2 = add nsw i32 %add.2, -1050011
  %sub.add.2 = select i1 %cmp4.2, i32 %sub.2, i32 %add.2
  %idxprom.3 = sext i32 %sub.add.2 to i64
  %arrayidx.3 = getelementptr inbounds i32, i32* %8, i64 %idxprom.3
  %14 = load i32, i32* %arrayidx.3, align 4
  %cmp1.3 = icmp eq i32 %14, %conv18.i
  br i1 %cmp1.3, label %if.then, label %if.end.3

if.end.3:
  %add.3 = add nsw i32 %add.i, %sub.add.2
  %cmp4.3 = icmp sgt i32 %add.3, 1050010
  %sub.3 = add nsw i32 %add.3, -1050011
  %sub.add.3 = select i1 %cmp4.3, i32 %sub.3, i32 %add.3
  %idxprom.4 = sext i32 %sub.add.3 to i64
  %arrayidx.4 = getelementptr inbounds i32, i32* %8, i64 %idxprom.4
  %15 = load i32, i32* %arrayidx.4, align 4
  %cmp1.4 = icmp eq i32 %15, %conv18.i
  br i1 %cmp1.4, label %if.then, label %if.end.4

if.end.4:
  %add.4 = add nsw i32 %add.i, %sub.add.3
  %cmp4.4 = icmp sgt i32 %add.4, 1050010
  %sub.4 = add nsw i32 %add.4, -1050011
  %sub.add.4 = select i1 %cmp4.4, i32 %sub.4, i32 %add.4
  %idxprom.5 = sext i32 %sub.add.4 to i64
  %arrayidx.5 = getelementptr inbounds i32, i32* %8, i64 %idxprom.5
  %16 = load i32, i32* %arrayidx.5, align 4
  %cmp1.5 = icmp eq i32 %16, %conv18.i
  br i1 %cmp1.5, label %if.then, label %if.end.5

if.end.5:
  %add.5 = add nsw i32 %add.i, %sub.add.4
  %cmp4.5 = icmp sgt i32 %add.5, 1050010
  %sub.5 = add nsw i32 %add.5, -1050011
  %sub.add.5 = select i1 %cmp4.5, i32 %sub.5, i32 %add.5
  %idxprom.6 = sext i32 %sub.add.5 to i64
  %arrayidx.6 = getelementptr inbounds i32, i32* %8, i64 %idxprom.6
  %17 = load i32, i32* %arrayidx.6, align 4
  %cmp1.6 = icmp eq i32 %17, %conv18.i
  br i1 %cmp1.6, label %if.then, label %if.end.6

if.end.6:
  %add.6 = add nsw i32 %add.i, %sub.add.5
  %cmp4.6 = icmp sgt i32 %add.6, 1050010
  %sub.6 = add nsw i32 %add.6, -1050011
  %sub.add.6 = select i1 %cmp4.6, i32 %sub.6, i32 %add.6
  %idxprom.7 = sext i32 %sub.add.6 to i64
  %arrayidx.7 = getelementptr inbounds i32, i32* %8, i64 %idxprom.7
  %18 = load i32, i32* %arrayidx.7, align 4
  %cmp1.7 = icmp eq i32 %18, %conv18.i
  br i1 %cmp1.7, label %if.then, label %cleanup
}
