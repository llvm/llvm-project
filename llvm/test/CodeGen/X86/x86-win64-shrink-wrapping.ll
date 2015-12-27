; RUN: llc %s -o - -enable-shrink-wrap=true | FileCheck %s --check-prefix=CHECK --check-prefix=ENABLE
; RUN: llc %s -o - -enable-shrink-wrap=false | FileCheck %s --check-prefix=CHECK --check-prefix=DISABLE

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "x86_64--windows-gnu"

; The output of this function with or without shrink-wrapping
; shouldn't change.
; Indeed, the epilogue block would have been if.else, meaning
; after the pops, we will have additional instruction (jump, mov,
; etc.) prior to the return and this is forbidden for Win64.
; CHECK-LABEL: loopInfoSaveOutsideLoop:
; CHECK: push
; CHECK: push
; CHECK-NOT: popq
; CHECK: popq
; CHECK: popq
; CHECK-NOT: popq
; CHECK-NEXT: retq
define i32 @loopInfoSaveOutsideLoop(i32 %cond, i32 %N) #0 {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %for.preheader

for.preheader:                                    ; preds = %entry
  tail call void asm "nop", ""()
  br label %for.body

for.body:                                         ; preds = %for.body, %for.preheader
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

; When we can sink the epilogue of the function into an existing exit block,
; this is Ok for shrink-wrapping to kicks in.
; CHECK-LABEL: loopInfoSaveOutsideLoop2:
; ENABLE: testl %ecx, %ecx
; ENABLE-NEXT: je [[ELSE_LABEL:.LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the CSR used in the inline asm: rbx.
; CHECK: pushq %rbp
; CHECK: pushq %rbx
;
; DISABLE: testl %ecx, %ecx
; DISABLE-NEXT: je [[ELSE_LABEL:.LBB[0-9_]+]]
;
; CHECK: nop
; CHECK: xorl [[SUM:%eax]], [[SUM]]
; CHECK-NEXT: movl $10, [[IV:%e[a-z]+]]
;
; CHECK: [[LOOP_LABEL:.LBB[0-9_]+]]: # %for.body
; CHECK: movl $1, [[TMP:%e[a-z]+]]
; CHECK: addl [[TMP]], [[SUM]]
; CHECK-NEXT: decl [[IV]]
; CHECK-NEXT: jne [[LOOP_LABEL]]
; Next BB.
; CHECK: nop
; CHECK: shll $3, [[SUM]]
;
; DISABLE: jmp [[EPILOG_BB:.LBB[0-9_]+]]
;
; ENABLE-NEXT: popq %rbx
; ENABLE-NEXT: popq %rbp
; ENABLE-NEXT: retq
;
; CHECK: [[ELSE_LABEL]]: # %if.else
; Shift second argument by one and store into returned register.
; CHECK: addl %edx, %edx
; CHECK: movl %edx, %eax
;
; DISABLE: [[EPILOG_BB]]: # %if.end
; DISABLE-NEXT: popq %rbx
;
; CHECK: retq
;
define i32 @loopInfoSaveOutsideLoop2(i32 %cond, i32 %N) #0 {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %for.preheader

for.preheader:                                    ; preds = %entry
  tail call void asm "nop", ""()
  br label %for.body

for.body:                                         ; preds = %for.body, %for.preheader
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
  ret i32 %shl

if.else:                                          ; preds = %entry
  %mul = shl nsw i32 %N, 1
  br label %if.end

if.end:                                           ; preds = %if.else, %for.end
  ret i32 %mul
}

attributes #0 = { uwtable }
