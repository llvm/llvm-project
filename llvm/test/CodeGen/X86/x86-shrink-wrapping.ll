; RUN: llc %s -o - -enable-shrink-wrap=true | FileCheck %s --check-prefix=CHECK --check-prefix=ENABLE
; RUN: llc %s -o - -enable-shrink-wrap=false | FileCheck %s --check-prefix=CHECK --check-prefix=DISABLE
;
; Note: Lots of tests use inline asm instead of regular calls.
; This allows to have a better control on what the allocation will do.
; Otherwise, we may have spill right in the entry block, defeating
; shrink-wrapping. Moreover, some of the inline asm statement (nop)
; are here to ensure that the related paths do not end up as critical
; edges.
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "x86_64-apple-macosx"


; Initial motivating example: Simple diamond with a call just on one side.
; CHECK-LABEL: foo:
;
; Compare the arguments and jump to exit.
; No prologue needed.
; ENABLE: movl %edi, [[ARG0CPY:%e[a-z]+]]
; ENABLE-NEXT: cmpl %esi, %edi
; ENABLE-NEXT: jge [[EXIT_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; (What we push does not matter. It should be some random sratch register.)
; CHECK: pushq
;
; Compare the arguments and jump to exit.
; After the prologue is set.
; DISABLE: movl %edi, [[ARG0CPY:%e[a-z]+]]
; DISABLE-NEXT: cmpl %esi, %edi
; DISABLE-NEXT: jge [[EXIT_LABEL:LBB[0-9_]+]]
;
; Store %a in the alloca.
; CHECK: movl [[ARG0CPY]], 4(%rsp)
; Set the alloca address in the second argument.
; CHECK-NEXT: leaq 4(%rsp), %rsi
; Set the first argument to zero.
; CHECK-NEXT: xorl %edi, %edi
; CHECK-NEXT: callq _doSomething
;
; With shrink-wrapping, epilogue is just after the call.
; ENABLE-NEXT: addq $8, %rsp
;
; CHECK: [[EXIT_LABEL]]:
;
; Without shrink-wrapping, epilogue is in the exit block.
; Epilogue code. (What we pop does not matter.)
; DISABLE-NEXT: popq
;
; CHECK-NEXT: retq
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
; ENABLE: testl %edi, %edi  
; ENABLE: je [[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the CSR used in the inline asm: rbx.
; CHECK: pushq %rbx
;
; DISABLE: testl %edi, %edi
; DISABLE: je [[ELSE_LABEL:LBB[0-9_]+]]
;
; SUM is in %esi because it is coalesced with the second
; argument on the else path.
; CHECK: xorl [[SUM:%esi]], [[SUM]]
; CHECK-NEXT: movl $10, [[IV:%e[a-z]+]]
;
; Next BB.
; CHECK: [[LOOP:LBB[0-9_]+]]: ## %for.body
; CHECK: movl $1, [[TMP:%e[a-z]+]]
; CHECK: addl [[TMP]], [[SUM]]
; CHECK-NEXT: decl [[IV]]
; CHECK-NEXT: jne [[LOOP]]
;
; Next BB.
; SUM << 3.
; CHECK: shll $3, [[SUM]]
;
; Jump to epilogue.
; DISABLE: jmp [[EPILOG_BB:LBB[0-9_]+]]
;
; DISABLE: [[ELSE_LABEL]]: ## %if.else
; Shift second argument by one and store into returned register.
; DISABLE: addl %esi, %esi
; DISABLE: [[EPILOG_BB]]: ## %if.end
;
; Epilogue code.
; CHECK-DAG: popq %rbx
; CHECK-DAG: movl %esi, %eax
; CHECK: retq
;
; ENABLE: [[ELSE_LABEL]]: ## %if.else
; Shift second argument by one and store into returned register.
; ENABLE: addl %esi, %esi
; ENABLE-NEXT: movl %esi, %eax
; ENABLE-NEXT: retq
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
  %call = tail call i32 asm "movl $$1, $0", "=r,~{ebx}"()
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
; Make sure we save the CSR used in the inline asm: rbx.
; CHECK: pushq %rbx
; CHECK: nop
; CHECK: xorl [[SUM:%e[a-z]+]], [[SUM]]
; CHECK-NEXT: movl $10, [[IV:%e[a-z]+]]
; Next BB.
; CHECK: [[LOOP_LABEL:LBB[0-9_]+]]: ## %for.body
; CHECK: movl $1, [[TMP:%e[a-z]+]]
; CHECK: addl [[TMP]], [[SUM]]
; CHECK-NEXT: decl [[IV]]
; CHECK-NEXT: jne [[LOOP_LABEL]]
; Next BB.
; CHECK: ## %for.exit
; CHECK: nop
; CHECK: popq %rbx
; CHECK-NEXT: retq
define i32 @freqSaveAndRestoreOutsideLoop2(i32 %cond) {
entry:
  br label %for.preheader

for.preheader:
  tail call void asm "nop", ""()
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.04 = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]
  %sum.03 = phi i32 [ 0, %for.preheader ], [ %add, %for.body ]
  %call = tail call i32 asm "movl $$1, $0", "=r,~{ebx}"()
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
; ENABLE: testl %edi, %edi
; ENABLE-NEXT: je [[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the CSR used in the inline asm: rbx.
; CHECK: pushq %rbx
;
; DISABLE: testl %edi, %edi
; DISABLE-NEXT: je [[ELSE_LABEL:LBB[0-9_]+]]
;
; CHECK: nop
; CHECK: xorl [[SUM:%esi]], [[SUM]]
; CHECK-NEXT: movl $10, [[IV:%e[a-z]+]]
;
; CHECK: [[LOOP_LABEL:LBB[0-9_]+]]: ## %for.body
; CHECK: movl $1, [[TMP:%e[a-z]+]]
; CHECK: addl [[TMP]], [[SUM]]
; CHECK-NEXT: decl [[IV]]
; CHECK-NEXT: jne [[LOOP_LABEL]]
; Next BB.
; CHECK: nop
; CHECK: shll $3, [[SUM]]
;
; DISABLE: jmp [[EPILOG_BB:LBB[0-9_]+]]
;
; DISABLE: [[ELSE_LABEL]]: ## %if.else
; Shift second argument by one and store into returned register.
; DISABLE: addl %esi, %esi
; DISABLE: [[EPILOG_BB]]: ## %if.end
;
; Epilogue code.
; CHECK-DAG: popq %rbx
; CHECK-DAG: movl %esi, %eax
; CHECK: retq
;
; ENABLE: [[ELSE_LABEL]]: ## %if.else
; Shift second argument by one and store into returned register.
; ENABLE: addl %esi, %esi
; ENABLE-NEXT: movl %esi, %eax
; ENABLE-NEXT: retq
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
  %call = tail call i32 asm "movl $$1, $0", "=r,~{ebx}"()
  %add = add nsw i32 %call, %sum.04
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  tail call void asm "nop", "~{ebx}"()
  %shl = shl i32 %add, 3
  br label %if.end

if.else:                                          ; preds = %entry
  %mul = shl nsw i32 %N, 1
  br label %if.end

if.end:                                           ; preds = %if.else, %for.end
  %sum.1 = phi i32 [ %shl, %for.end ], [ %mul, %if.else ]
  ret i32 %sum.1
}

; Check with a more complex case that we do not have restore within the loop and
; save outside.
; CHECK-LABEL: loopInfoRestoreOutsideLoop:
;
; ENABLE: testl %edi, %edi
; ENABLE-NEXT: je [[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the CSR used in the inline asm: rbx.
; CHECK: pushq %rbx
;
; DISABLE: testl %edi, %edi
; DISABLE-NEXT: je [[ELSE_LABEL:LBB[0-9_]+]]
;
; CHECK: nop
; CHECK: xorl [[SUM:%esi]], [[SUM]]
; CHECK-NEXT: movl $10, [[IV:%e[a-z]+]]
;
; CHECK: [[LOOP_LABEL:LBB[0-9_]+]]: ## %for.body
; CHECK: movl $1, [[TMP:%e[a-z]+]]
; CHECK: addl [[TMP]], [[SUM]]
; CHECK-NEXT: decl [[IV]]
; CHECK-NEXT: jne [[LOOP_LABEL]]
; Next BB.
; CHECK: shll $3, [[SUM]]
;
; DISABLE: jmp [[EPILOG_BB:LBB[0-9_]+]]
;
; DISABLE: [[ELSE_LABEL]]: ## %if.else

; Shift second argument by one and store into returned register.
; DISABLE: addl %esi, %esi
; DISABLE: [[EPILOG_BB]]: ## %if.end
;
; Epilogue code.
; CHECK-DAG: popq %rbx
; CHECK-DAG: movl %esi, %eax
; CHECK: retq
;
; ENABLE: [[ELSE_LABEL]]: ## %if.else
; Shift second argument by one and store into returned register.
; ENABLE: addl %esi, %esi
; ENABLE-NEXT: movl %esi, %eax
; ENABLE-NEXT: retq
define i32 @loopInfoRestoreOutsideLoop(i32 %cond, i32 %N) #0 {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  tail call void asm "nop", "~{ebx}"()
  br label %for.body

for.body:                                         ; preds = %for.body, %if.then
  %i.05 = phi i32 [ 0, %if.then ], [ %inc, %for.body ]
  %sum.04 = phi i32 [ 0, %if.then ], [ %add, %for.body ]
  %call = tail call i32 asm "movl $$1, $0", "=r,~{ebx}"()
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
; CHECK: ## %entry
; CHECK-NEXT: xorl %eax, %eax
; CHECK-NEXT: retq
define i32 @emptyFrame() {
entry:
  ret i32 0
}

; Check that we handle inline asm correctly.
; CHECK-LABEL: inlineAsm:
;
; ENABLE: testl %edi, %edi
; ENABLE-NEXT: je [[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the CSR used in the inline asm: rbx.
; CHECK: pushq %rbx
;
; DISABLE: testl %edi, %edi
; DISABLE-NEXT: je [[ELSE_LABEL:LBB[0-9_]+]]
;
; CHECK: nop
; CHECK: movl $10, [[IV:%e[a-z]+]]
;
; CHECK: [[LOOP_LABEL:LBB[0-9_]+]]: ## %for.body
; Inline asm statement.
; CHECK: addl $1, %ebx
; CHECK: decl [[IV]]
; CHECK-NEXT: jne [[LOOP_LABEL]]
; Next BB.
; CHECK: nop
; CHECK: xorl %esi, %esi
;
; DISABLE: jmp [[EPILOG_BB:LBB[0-9_]+]]
;
; DISABLE: [[ELSE_LABEL]]: ## %if.else
; Shift second argument by one and store into returned register.
; DISABLE: addl %esi, %esi
; DISABLE: [[EPILOG_BB]]: ## %if.end
;
; Epilogue code.
; CHECK-DAG: popq %rbx
; CHECK-DAG: movl %esi, %eax
; CHECK: retq
;
; ENABLE: [[ELSE_LABEL]]: ## %if.else
; Shift second argument by one and store into returned register.
; ENABLE: addl %esi, %esi
; ENABLE-NEXT: movl %esi, %eax
; ENABLE-NEXT: retq
define i32 @inlineAsm(i32 %cond, i32 %N) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %for.preheader

for.preheader:
  tail call void asm "nop", ""()
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.03 = phi i32 [ %inc, %for.body ], [ 0, %for.preheader ]
  tail call void asm "addl $$1, %ebx", "~{ebx}"()
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
; ENABLE: testl %edi, %edi
; ENABLE-NEXT: je [[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; CHECK: pushq
;
; DISABLE: testl %edi, %edi
; DISABLE-NEXT: je [[ELSE_LABEL:LBB[0-9_]+]]
;
; Setup of the varags.
; CHECK: movl %esi, (%rsp)
; CHECK-NEXT: xorl %eax, %eax
; CHECK-NEXT: %esi, %edi
; CHECK-NEXT: %esi, %edx
; CHECK-NEXT: %esi, %ecx
; CHECK-NEXT: %esi, %r8d
; CHECK-NEXT: %esi, %r9d
; CHECK-NEXT: callq _someVariadicFunc
; CHECK-NEXT: movl %eax, %esi
; CHECK-NEXT: shll $3, %esi
;
; ENABLE-NEXT: addq $8, %rsp
; ENABLE-NEXT: movl %esi, %eax
; ENABLE-NEXT: retq
;
; DISABLE: jmp [[IFEND_LABEL:LBB[0-9_]+]]
;
; CHECK: [[ELSE_LABEL]]: ## %if.else
; Shift second argument by one and store into returned register.
; CHECK: addl %esi, %esi
;
; DISABLE: [[IFEND_LABEL]]: ## %if.end
;
; Epilogue code.
; CHECK-NEXT: movl %esi, %eax
; DISABLE-NEXT: popq
; CHECK-NEXT: retq
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

; Check that we use LEA not to clobber EFLAGS.
%struct.temp_slot = type { %struct.temp_slot*, %struct.rtx_def*, %struct.rtx_def*, i32, i64, %union.tree_node*, %union.tree_node*, i8, i8, i32, i32, i64, i64 }
%union.tree_node = type { %struct.tree_decl }
%struct.tree_decl = type { %struct.tree_common, i8*, i32, i32, %union.tree_node*, i48, %union.anon, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %union.tree_node*, %struct.rtx_def*, %struct.rtx_def*, %union.anon.1, %union.tree_node*, %union.tree_node*, %union.tree_node*, i64, %struct.lang_decl* }
%struct.tree_common = type { %union.tree_node*, %union.tree_node*, i32 }
%union.anon = type { i64 }
%union.anon.1 = type { %struct.function* }
%struct.function = type { %struct.eh_status*, %struct.stmt_status*, %struct.expr_status*, %struct.emit_status*, %struct.varasm_status*, i8*, %union.tree_node*, %struct.function*, i32, i32, i32, i32, %struct.rtx_def*, %struct.ix86_args, %struct.rtx_def*, %struct.rtx_def*, i8*, %struct.initial_value_struct*, i32, %union.tree_node*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %union.tree_node*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, i64, %union.tree_node*, %union.tree_node*, %struct.rtx_def*, %struct.rtx_def*, i32, %struct.rtx_def**, %struct.temp_slot*, i32, i32, i32, %struct.var_refs_queue*, i32, i32, i8*, %union.tree_node*, %struct.rtx_def*, i32, i32, %struct.machine_function*, i32, i32, %struct.language_function*, %struct.rtx_def*, i24 }
%struct.eh_status = type opaque
%struct.stmt_status = type opaque
%struct.expr_status = type { i32, i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def* }
%struct.emit_status = type { i32, i32, %struct.rtx_def*, %struct.rtx_def*, %union.tree_node*, %struct.sequence_stack*, i32, i32, i8*, i32, i8*, %union.tree_node**, %struct.rtx_def** }
%struct.sequence_stack = type { %struct.rtx_def*, %struct.rtx_def*, %union.tree_node*, %struct.sequence_stack* }
%struct.varasm_status = type opaque
%struct.ix86_args = type { i32, i32, i32, i32, i32, i32, i32 }
%struct.initial_value_struct = type opaque
%struct.var_refs_queue = type { %struct.rtx_def*, i32, i32, %struct.var_refs_queue* }
%struct.machine_function = type opaque
%struct.language_function = type opaque
%struct.lang_decl = type opaque
%struct.rtx_def = type { i32, [1 x %union.rtunion_def] }
%union.rtunion_def = type { i64 }

declare hidden fastcc %struct.temp_slot* @find_temp_slot_from_address(%struct.rtx_def* readonly)

; CHECK-LABEL: useLEA:
; DISABLE: pushq 
;
; CHECK: testq   %rdi, %rdi
; CHECK-NEXT: je      [[CLEANUP:LBB[0-9_]+]]
;
; CHECK: movzwl  (%rdi), [[BF_LOAD:%e[a-z]+]]
; CHECK-NEXT: cmpl $66, [[BF_LOAD]]
; CHECK-NEXT: jne [[CLEANUP]]
;
; CHECK: movq 8(%rdi), %rdi
; CHECK-NEXT: movzwl (%rdi), %e[[BF_LOAD2:[a-z]+]]
; CHECK-NEXT: leal -54(%r[[BF_LOAD2]]), [[TMP:%e[a-z]+]]
; CHECK-NEXT: cmpl $14, [[TMP]]
; CHECK-NEXT: ja [[LOR_LHS_FALSE:LBB[0-9_]+]]
;
; CHECK: movl $24599, [[TMP2:%e[a-z]+]]
; CHECK-NEXT: btl [[TMP]], [[TMP2]]
; CHECK-NEXT: jae [[LOR_LHS_FALSE:LBB[0-9_]+]]
;
; CHECK: [[CLEANUP]]: ## %cleanup
; DISABLE: popq
; CHECK-NEXT: retq
;
; CHECK: [[LOR_LHS_FALSE]]: ## %lor.lhs.false
; CHECK: cmpl $134, %e[[BF_LOAD2]]
; CHECK-NEXT: je [[CLEANUP]]
;
; CHECK: cmpl $140, %e[[BF_LOAD2]]
; CHECK-NEXT: je [[CLEANUP]]
;
; ENABLE: pushq
; CHECK: callq _find_temp_slot_from_address
; CHECK-NEXT: testq   %rax, %rax
;
; The adjustment must use LEA here (or be moved above the test).
; ENABLE-NEXT: leaq 8(%rsp), %rsp
;
; CHECK-NEXT: je [[CLEANUP]]
;
; CHECK: movb $1, 57(%rax)
define void @useLEA(%struct.rtx_def* readonly %x) {
entry:
  %cmp = icmp eq %struct.rtx_def* %x, null
  br i1 %cmp, label %cleanup, label %if.end

if.end:                                           ; preds = %entry
  %tmp = getelementptr inbounds %struct.rtx_def, %struct.rtx_def* %x, i64 0, i32 0
  %bf.load = load i32, i32* %tmp, align 8
  %bf.clear = and i32 %bf.load, 65535
  %cmp1 = icmp eq i32 %bf.clear, 66
  br i1 %cmp1, label %lor.lhs.false, label %cleanup

lor.lhs.false:                                    ; preds = %if.end
  %arrayidx = getelementptr inbounds %struct.rtx_def, %struct.rtx_def* %x, i64 0, i32 1, i64 0
  %rtx = bitcast %union.rtunion_def* %arrayidx to %struct.rtx_def**
  %tmp1 = load %struct.rtx_def*, %struct.rtx_def** %rtx, align 8
  %tmp2 = getelementptr inbounds %struct.rtx_def, %struct.rtx_def* %tmp1, i64 0, i32 0
  %bf.load2 = load i32, i32* %tmp2, align 8
  %bf.clear3 = and i32 %bf.load2, 65535
  switch i32 %bf.clear3, label %if.end.55 [
    i32 67, label %cleanup
    i32 68, label %cleanup
    i32 54, label %cleanup
    i32 55, label %cleanup
    i32 58, label %cleanup
    i32 134, label %cleanup
    i32 56, label %cleanup
    i32 140, label %cleanup
  ]

if.end.55:                                        ; preds = %lor.lhs.false
  %call = tail call fastcc %struct.temp_slot* @find_temp_slot_from_address(%struct.rtx_def* %tmp1) #2
  %cmp59 = icmp eq %struct.temp_slot* %call, null
  br i1 %cmp59, label %cleanup, label %if.then.60

if.then.60:                                       ; preds = %if.end.55
  %addr_taken = getelementptr inbounds %struct.temp_slot, %struct.temp_slot* %call, i64 0, i32 8
  store i8 1, i8* %addr_taken, align 1
  br label %cleanup

cleanup:                                          ; preds = %if.then.60, %if.end.55, %lor.lhs.false, %lor.lhs.false, %lor.lhs.false, %lor.lhs.false, %lor.lhs.false, %lor.lhs.false, %lor.lhs.false, %lor.lhs.false, %if.end, %entry
  ret void
}

; Make sure we do not insert unreachable code after noreturn function.
; Although this is not incorrect to insert such code, it is useless
; and it hurts the binary size.
;
; CHECK-LABEL: noreturn:
; DISABLE: pushq
;
; CHECK: testb   %dil, %dil
; CHECK-NEXT: jne      [[ABORT:LBB[0-9_]+]]
;
; CHECK: movl $42, %eax
;
; DISABLE-NEXT: popq
;
; CHECK-NEXT: retq
;
; CHECK: [[ABORT]]: ## %if.abort
;
; ENABLE: pushq
;
; CHECK: callq _abort
; ENABLE-NOT: popq
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
; CHECK: retq
define void @infiniteloop() {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:
  %ptr = alloca i32, i32 4
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %sum.03 = phi i32 [ 0, %if.then ], [ %add, %for.body ]
  %call = tail call i32 asm "movl $$1, $0", "=r,~{ebx}"()
  %add = add nsw i32 %call, %sum.03
  store i32 %add, i32* %ptr
  br label %for.body

if.end:
  ret void
}

; Another infinite loop test this time with a body bigger than just one block.
; CHECK-LABEL: infiniteloop2
; CHECK: retq
define void @infiniteloop2() {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:
  %ptr = alloca i32, i32 4
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %sum.03 = phi i32 [ 0, %if.then ], [ %add, %body1 ], [ 1, %body2]
  %call = tail call i32 asm "movl $$1, $0", "=r,~{ebx}"()
  %add = add nsw i32 %call, %sum.03
  store i32 %add, i32* %ptr
  br i1 undef, label %body1, label %body2

body1:
  tail call void asm sideeffect "nop", "~{ebx}"()
  br label %for.body

body2:
  tail call void asm sideeffect "nop", "~{ebx}"()
  br label %for.body

if.end:
  ret void
}

; Another infinite loop test this time with two nested infinite loop.
; CHECK-LABEL: infiniteloop3
; CHECK: retq
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

; Check that we just don't bail out on RegMask.
; In this case, the RegMask does not touch a CSR so we are good to go!
; CHECK-LABEL: regmask:
;
; Compare the arguments and jump to exit.
; No prologue needed.
; ENABLE: cmpl %esi, %edi
; ENABLE-NEXT: jge [[EXIT_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; (What we push does not matter. It should be some random sratch register.)
; CHECK: pushq
;
; Compare the arguments and jump to exit.
; After the prologue is set.
; DISABLE: cmpl %esi, %edi
; DISABLE-NEXT: jge [[EXIT_LABEL:LBB[0-9_]+]]
;
; CHECK: nop
; Set the first argument to zero.
; CHECK: xorl %edi, %edi
; Set the second argument to addr.
; CHECK-NEXT: movq %rdx, %rsi
; CHECK-NEXT: callq _doSomething
; CHECK-NEXT: popq
; CHECK-NEXT: retq
;
; CHECK: [[EXIT_LABEL]]:
; Set the first argument to 6.
; CHECK-NEXT: movl $6, %edi
; Set the second argument to addr.
; CHECK-NEXT: movq %rdx, %rsi
;
; Without shrink-wrapping, we need to restore the stack before
; making the tail call.
; Epilogue code.
; DISABLE-NEXT: popq
;
; CHECK-NEXT: jmp _doSomething
define i32 @regmask(i32 %a, i32 %b, i32* %addr) {
  %tmp2 = icmp slt i32 %a, %b
  br i1 %tmp2, label %true, label %false

true:
  ; Clobber a CSR so that we check something on the regmask
  ; of the tail call.
  tail call void asm sideeffect "nop", "~{ebx}"()
  %tmp4 = call i32 @doSomething(i32 0, i32* %addr)
  br label %end

false:
  %tmp5 = tail call i32 @doSomething(i32 6, i32* %addr)
  br label %end

end:
  %tmp.0 = phi i32 [ %tmp4, %true ], [ %tmp5, %false ]
  ret i32 %tmp.0
}

@b = internal unnamed_addr global i1 false
@c = internal unnamed_addr global i8 0, align 1
@a = common global i32 0, align 4

; Make sure the prologue does not clobber the EFLAGS when
; it is live accross.
; PR25629.
; Note: The registers may change in the following patterns, but
; because they imply register hierarchy (e.g., eax, al) this is
; tricky to write robust patterns.
;
; CHECK-LABEL: useLEAForPrologue:
;
; Prologue is at the beginning of the function when shrink-wrapping
; is disabled.
; DISABLE: pushq
; The stack adjustment can use SUB instr because we do not need to
; preserve the EFLAGS at this point.
; DISABLE-NEXT: subq $16, %rsp
;
; Load the value of b.
; CHECK: movb _b(%rip), [[BOOL:%cl]]
; Create the zero value for the select assignment.
; CHECK-NEXT: xorl [[CMOVE_VAL:%eax]], [[CMOVE_VAL]]
; CHECK-NEXT: testb [[BOOL]], [[BOOL]]
; CHECK-NEXT: jne [[STOREC_LABEL:LBB[0-9_]+]]
;
; CHECK: movb $48, [[CMOVE_VAL:%al]]
;
; CHECK: [[STOREC_LABEL]]:
;
; ENABLE-NEXT: pushq
; For the stack adjustment, we need to preserve the EFLAGS.
; ENABLE-NEXT: leaq -16(%rsp), %rsp
;
; Technically, we should use CMOVE_VAL here or its subregister.
; CHECK-NEXT: movb %al, _c(%rip)
; testb set the EFLAGS read here.
; CHECK-NEXT: je [[VARFUNC_CALL:LBB[0-9_]+]]
;
; The code of the loop is not interesting.
; [...]
;
; CHECK: [[VARFUNC_CALL]]:
; Set the null parameter.
; CHECK-NEXT: xorl %edi, %edi
; CHECK-NEXT: callq _varfunc
;
; Set the return value.
; CHECK-NEXT: xorl %eax, %eax
;
; Epilogue code.
; CHECK-NEXT: addq $16, %rsp
; CHECK-NEXT: popq
; CHECK-NEXT: retq
define i32 @useLEAForPrologue(i32 %d, i32 %a, i8 %c) #3 {
entry:
  %tmp = alloca i3
  %.b = load i1, i1* @b, align 1
  %bool = select i1 %.b, i8 0, i8 48
  store i8 %bool, i8* @c, align 1
  br i1 %.b, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  tail call void asm sideeffect "nop", "~{ebx}"()
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %inc6 = phi i8 [ %c, %for.body.lr.ph ], [ %inc, %for.body ]
  %cond5 = phi i32 [ %a, %for.body.lr.ph ], [ %conv3, %for.body ]
  %cmp2 = icmp slt i32 %d, %cond5
  %conv3 = zext i1 %cmp2 to i32
  %inc = add i8 %inc6, 1
  %cmp = icmp slt i8 %inc, 45
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  store i32 %conv3, i32* @a, align 4
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  %call = tail call i32 (i8*) @varfunc(i8* null)
  ret i32 0
}

declare i32 @varfunc(i8* nocapture readonly)

@sum1 = external hidden thread_local global i32, align 4


; Function Attrs: nounwind
; Make sure the TLS call used to access @sum1 happens after the prologue
; and before the epilogue.
; TLS calls used to be wrongly model and shrink-wrapping would have inserted
; the prologue and epilogue just around the call to doSomething.
; PR25820.
;
; CHECK-LABEL: tlsCall:
; CHECK: pushq
; CHECK: testb $1, %dil
; CHECK: je [[ELSE_LABEL:LBB[0-9_]+]]
;
; master bb
; CHECK: movq _sum1@TLVP(%rip), %rdi
; CHECK-NEXT: callq *(%rdi)
; CHECK: jmp [[EXIT_LABEL:LBB[0-9_]+]]
;
; [[ELSE_LABEL]]:
; CHECK: callq _doSomething
;
; [[EXIT_LABEL]]:
; CHECK: popq
; CHECK-NEXT: retq
define i32 @tlsCall(i1 %bool1, i32 %arg, i32* readonly dereferenceable(4) %sum1) #3 {
entry:
  br i1 %bool1, label %master, label %else

master:
  %tmp1 = load i32, i32* %sum1, align 4
  store i32 %tmp1, i32* @sum1, align 4
  br label %exit

else:
  %call = call i32 @doSomething(i32 0, i32* null)
  br label %exit

exit:
  %res = phi i32 [ %arg, %master], [ %call, %else ]
  ret i32 %res
}

attributes #3 = { nounwind }

@irreducibleCFGa = common global i32 0, align 4
@irreducibleCFGf = common global i8 0, align 1
@irreducibleCFGb = common global i32 0, align 4

; Check that we do not run shrink-wrapping on irreducible CFGs until
; it is actually supported.
; At the moment, on those CFGs the loop information may be incorrect
; and since we use that information to do the placement, we may end up
; inserting the prologue/epilogue at incorrect places.
; PR25988.
;
; CHECK-LABEL: irreducibleCFG:
; CHECK: %entry
; Make sure the prologue happens in the entry block.
; CHECK-NEXT: pushq
; ...
; Make sure the epilogue happens in the exit block.
; CHECK-NOT: popq
; CHECK: popq
; CHECK-NEXT: popq
; CHECK-NEXT: retq
define i32 @irreducibleCFG() #4 {
entry:
  %i0 = load i32, i32* @irreducibleCFGa, align 4
  %.pr = load i8, i8* @irreducibleCFGf, align 1
  %bool = icmp eq i8 %.pr, 0
  br i1 %bool, label %split, label %preheader

preheader:
  br label %preheader

split:
  %i1 = load i32, i32* @irreducibleCFGb, align 4
  %tobool1.i = icmp ne i32 %i1, 0
  br i1 %tobool1.i, label %for.body4.i, label %for.cond8.i.preheader

for.body4.i:
  %call.i = tail call i32 (...) @something(i32 %i0)
  br label %for.cond8

for.cond8:
  %p1 = phi i32 [ %inc18.i, %for.inc ], [ 0, %for.body4.i ]
  %.pr1.pr = load i32, i32* @irreducibleCFGb, align 4
  br label %for.cond8.i.preheader

for.cond8.i.preheader:
  %.pr1 = phi i32 [ %.pr1.pr, %for.cond8 ], [ %i1, %split ]
  %p13 = phi i32 [ %p1, %for.cond8 ], [ 0, %split ]
  br label %for.inc

fn1.exit:
  ret i32 0

for.inc:
  %inc18.i = add nuw nsw i32 %p13, 1
  %cmp = icmp slt i32 %inc18.i, 7
  br i1 %cmp, label %for.cond8, label %fn1.exit
}

attributes #4 = { "no-frame-pointer-elim"="true" }

@x = external global i32, align 4
@y = external global i32, align 4

; The post-dominator tree does not include the branch containing the infinite
; loop, which can occur into a misplacement of the restore block, if we're
; looking for the nearest common post-dominator of an "unreachable" block.

; CHECK-LABEL: infiniteLoopNoSuccessor:
; CHECK: ## BB#0:
; Make sure the prologue happens in the entry block.
; CHECK-NEXT: pushq %rbp
; ...
; Make sure we don't shrink-wrap.
; CHECK: ## BB#1
; CHECK-NOT: pushq %rbp
; ...
; Make sure the epilogue happens in the exit block.
; CHECK: ## BB#5
; CHECK: popq %rbp
; CHECK-NEXT: retq
define void @infiniteLoopNoSuccessor() #5 {
  %1 = load i32, i32* @x, align 4
  %2 = icmp ne i32 %1, 0
  br i1 %2, label %3, label %4

; <label>:3:
  store i32 0, i32* @x, align 4
  br label %4

; <label>:4:
  call void (...) @somethingElse()
  %5 = load i32, i32* @y, align 4
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %10, label %7

; <label>:7:
  %8 = call i32 (...) @something()
  br label %9

; <label>:9:
  call void (...) @somethingElse()
  br label %9

; <label>:10:
  ret void
}

declare void @somethingElse(...)

attributes #5 = { nounwind  "no-frame-pointer-elim-non-leaf" }
