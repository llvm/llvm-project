; RUN: llc -mtriple=x86_64-unknown-windows-msvc -mattr=+egpr -o - %s | FileCheck %s

; EGPR is enabled to verify V3 + EGPR compiles without errors.
; R16-R31 are caller-saved on Win64, so they won't appear in SEH push/pop.

; V3 uses a module-wide default via a file-level .seh_unwindversion 3 directive.
; Functions should NOT have per-function .seh_unwindversion or .seh_unwindv2start.

; Unlike V1/V2, there is a .seh_* directive *before* each real instruction in
; the prolog AND before each instruction in the epilog.

; CHECK:        .seh_unwindversion 3

define dso_local void @no_epilog() local_unnamed_addr {
entry:
  ret void
}
; CHECK-LABEL:  no_epilog:
; CHECK-NOT:    .seh_
; CHECK:        retq

define dso_local void @stack_alloc_no_pushes() local_unnamed_addr {
entry:
  call void @a()
  ret void
}
; CHECK-LABEL:  stack_alloc_no_pushes:
; CHECK:        .seh_proc stack_alloc_no_pushes
; CHECK-NOT:    .seh_unwindv2start
; CHECK:        .seh_stackalloc
; CHECK-NEXT:   subq
; CHECK:        .seh_endprologue
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   .seh_stackalloc
; CHECK-NEXT:   addq
; CHECK-NOT:    .seh_unwindv2start
; CHECK:        .seh_endepilogue
; CHECK-NEXT:   retq

define dso_local i32 @stack_alloc_and_pushes(i32 %x) local_unnamed_addr {
entry:
  %call = tail call i32 @c(i32 %x)
  %call1 = tail call i32 @c(i32 %x)
  %add = add nsw i32 %call1, %call
  %call2 = tail call i32 @c(i32 %x)
  %call3 = tail call i32 @c(i32 %call2)
  %add4 = add nsw i32 %add, %call3
  ret i32 %add4
}
; CHECK-LABEL:  stack_alloc_and_pushes:
; CHECK:        .seh_proc stack_alloc_and_pushes
; CHECK:        .seh_pushreg
; CHECK-NEXT:   pushq
; CHECK-NEXT:   .seh_pushreg
; CHECK-NEXT:   pushq
; CHECK-NEXT:   .seh_pushreg
; CHECK-NEXT:   pushq
; CHECK-NEXT:   .seh_stackalloc
; CHECK-NEXT:   subq
; CHECK-NEXT:   .seh_endprologue
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   .seh_stackalloc
; CHECK-NEXT:   addq
; CHECK-NEXT:   .seh_pushreg
; CHECK-NEXT:   popq
; CHECK-NEXT:   .seh_pushreg
; CHECK-NEXT:   popq
; CHECK-NEXT:   .seh_pushreg
; CHECK-NEXT:   popq
; CHECK-NOT:    .seh_unwindv2start
; CHECK-NEXT:   .seh_endepilogue

define dso_local i32 @multiple_epilogs(i32 %x) local_unnamed_addr {
entry:
  %call = tail call i32 @c(i32 noundef %x)
  %cmp = icmp sgt i32 %call, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %call1 = tail call i32 @c(i32 noundef %call)
  ret i32 %call1

if.else:
  %call2 = tail call i32 @b()
  ret i32 %call2
}
; CHECK-LABEL:  multiple_epilogs:
; CHECK:        .seh_proc multiple_epilogs
; CHECK:        .seh_stackalloc
; CHECK:        .seh_endprologue
; CHECK:        .seh_startepilogue
; CHECK-NOT:    .seh_unwindv2start
; CHECK:        .seh_endepilogue
; CHECK:        .seh_startepilogue
; CHECK-NOT:    .seh_unwindv2start
; CHECK:        .seh_endepilogue

; ---- Tail call: epilog terminates with jmp, not ret ----
define dso_local i32 @tail_call(i32 %x) local_unnamed_addr {
entry:
  %call = tail call i32 @c(i32 %x)
  %call1 = tail call i32 @c(i32 %call)
  ret i32 %call1
}
; CHECK-LABEL:  tail_call:
; CHECK:        .seh_proc tail_call
; CHECK:        .seh_stackalloc
; CHECK-NEXT:   subq
; CHECK:        .seh_endprologue
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   .seh_stackalloc
; CHECK-NEXT:   addq
; CHECK-NOT:    .seh_unwindv2start
; CHECK:        .seh_endepilogue
; CHECK-NEXT:   jmp

; ---- Dynamic stack alloc: setframe with rbp ----
define dso_local void @dynamic_stack_alloc(i32 %x) local_unnamed_addr {
entry:
  %y = alloca i32, i32 %x
  ret void
}
; CHECK-LABEL:  dynamic_stack_alloc:
; CHECK:        .seh_proc dynamic_stack_alloc
; CHECK:        .seh_pushreg %rbp
; CHECK-NEXT:   pushq   %rbp
; CHECK:        .seh_setframe %rbp, 0
; CHECK:        .seh_endprologue
; Epilogue must have SEH_SetFrame to undo the prolog setframe, even when
; LEAAmount == 0 (i.e., the epilogue uses MOV RSP, RBP instead of LEA).
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   .seh_setframe %rbp, 0
; CHECK-NEXT:   movq    %rbp, %rsp
; CHECK-NEXT:   .seh_pushreg %rbp
; CHECK-NEXT:   popq    %rbp
; CHECK-NOT:    .seh_unwindv2start
; CHECK:        .seh_endepilogue
; CHECK-NEXT:   retq

; ---- Large aligned alloc: setframe with non-zero offset ----
define dso_local void @large_aligned_alloc() align 16 {
  %1 = alloca [128 x i8], align 64
  ret void
}
; CHECK-LABEL:  large_aligned_alloc:
; CHECK:        .seh_proc large_aligned_alloc
; CHECK:        .seh_pushreg %rbp
; CHECK-NEXT:   pushq   %rbp
; CHECK:        .seh_stackalloc
; CHECK-NEXT:   subq
; CHECK:        .seh_setframe %rbp,
; CHECK:        .seh_endprologue
; CHECK:        .seh_startepilogue
; CHECK-NOT:    .seh_unwindv2start
; CHECK:        .seh_endepilogue
; CHECK-NEXT:   retq

; ---- No-return function: no epilog emitted ----
attributes #1 = { noreturn }
define dso_local void @no_return_func() local_unnamed_addr #1 {
entry:
  call void @d()
  unreachable
}
; CHECK-LABEL:  no_return_func:
; CHECK:        .seh_stackalloc
; CHECK-NEXT:   subq
; CHECK:        .seh_endprologue
; CHECK-NOT:    .seh_startepilogue
; CHECK-NOT:    .seh_unwindv2start
; CHECK:        int3
; CHECK-NEXT:   .seh_endproc

; ---- Exception handler with funclet ----
define dso_local i32 @has_funclet(i32 %x) local_unnamed_addr personality ptr @__C_specific_handler {
entry:
  %call = invoke i32 @c(i32 %x)
    to label %call.block.1 unwind label %cleanup

call.block.1:
  %call1 = invoke i32 @c(i32 %x)
    to label %call.block.2 unwind label %cleanup

call.block.2:
  %add = add nsw i32 %call1, %call
  %call2 = invoke i32 @c(i32 %x)
    to label %call.block.3 unwind label %cleanup

call.block.3:
  %call3 = invoke i32 @c(i32 %call2)
    to label %call.block.4 unwind label %cleanup

call.block.4:
  %add4 = add nsw i32 %add, %call3
  ret i32 %add4

cleanup:
  %cleanup_token = cleanuppad within none []
  call fastcc void @cleanup_helper(i32 %x) #18 [ "funclet"(token %cleanup_token) ]
  cleanupret from %cleanup_token unwind to caller
}

define internal fastcc void @cleanup_helper(i32 %x) local_unnamed_addr {
entry:
  %y = alloca i32, i32 %x
  ret void
}

; CHECK-LABEL:  has_funclet:
; CHECK:        .seh_proc has_funclet
; CHECK:        .seh_handler __C_specific_handler, @unwind
; CHECK:        .seh_pushreg %rbp
; CHECK-NEXT:   pushq   %rbp
; CHECK-NEXT:   .seh_pushreg %rsi
; CHECK-NEXT:   pushq   %rsi
; CHECK-NEXT:   .seh_pushreg %rdi
; CHECK-NEXT:   pushq   %rdi
; CHECK:        .seh_stackalloc 48
; CHECK-NEXT:   subq    $48, %rsp
; CHECK:        .seh_setframe %rbp, 48
; CHECK:        .seh_endprologue
; Epilog: no setframe, the stackalloc will take care of it
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   .seh_stackalloc 48
; CHECK-NEXT:   addq    $48, %rsp
; CHECK-NEXT:   .seh_pushreg %rdi
; CHECK-NEXT:   popq    %rdi
; CHECK-NEXT:   .seh_pushreg %rsi
; CHECK-NEXT:   popq    %rsi
; CHECK-NEXT:   .seh_pushreg %rbp
; CHECK-NEXT:   popq    %rbp
; CHECK-NEXT:   .seh_endepilogue
; CHECK-NEXT:   retq
; CHECK:        .seh_handlerdata
; CHECK:        .text
; CHECK-NEXT:   .seh_endproc

; The funclet has its own .seh_proc/.seh_endproc covering its code range.
; CHECK-LABEL:  "?dtor$5@?0?has_funclet@4HA":
; CHECK:        .seh_proc "?dtor$5@?0?has_funclet@4HA"
; CHECK:        .seh_pushreg %rbp
; CHECK-NEXT:   pushq   %rbp
; CHECK-NEXT:   .seh_pushreg %rsi
; CHECK-NEXT:   pushq   %rsi
; CHECK-NEXT:   .seh_pushreg %rdi
; CHECK-NEXT:   pushq   %rdi
; CHECK-NEXT:   .seh_stackalloc 32
; CHECK-NEXT:   subq    $32, %rsp
; CHECK:        .seh_endprologue
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   .seh_stackalloc 32
; CHECK-NEXT:   addq    $32, %rsp
; CHECK-NEXT:   .seh_pushreg %rdi
; CHECK-NEXT:   popq    %rdi
; CHECK-NEXT:   .seh_pushreg %rsi
; CHECK-NEXT:   popq    %rsi
; CHECK-NEXT:   .seh_pushreg %rbp
; CHECK-NEXT:   popq    %rbp
; CHECK-NEXT:   .seh_endepilogue
; CHECK-NEXT:   retq
; CHECK:        .seh_endproc

; The helper function also gets V3 treatment.
; CHECK-LABEL:  cleanup_helper:
; CHECK:        .seh_pushreg %rbp
; CHECK-NEXT:   pushq   %rbp
; CHECK:        .seh_setframe %rbp, 0
; CHECK:        .seh_endprologue
; CHECK:        .seh_startepilogue
; CHECK:        .seh_endepilogue
; CHECK-NEXT:   retq
; CHECK-NEXT:   .seh_endproc

declare void @a() local_unnamed_addr
declare i32 @b() local_unnamed_addr
declare i32 @c(i32) local_unnamed_addr
declare void @d() local_unnamed_addr #1
declare dso_local i32 @__C_specific_handler(...)
declare i64 @llvm.x86.flags.read.u64()

; ---- XMM callee-saved saves: SEH_SaveXMM before movaps in V3 ----
define dso_local void @xmm_saves() local_unnamed_addr {
entry:
  call void asm sideeffect "", "~{xmm6},~{xmm7}"()
  call void @a()
  ret void
}
; CHECK-LABEL:  xmm_saves:
; CHECK:        .seh_proc xmm_saves
; CHECK:        .seh_stackalloc
; CHECK-NEXT:   subq
; CHECK:        .seh_savexmm %xmm7,
; CHECK-NEXT:   movaps
; CHECK:        .seh_savexmm %xmm6,
; CHECK-NEXT:   movaps
; CHECK:        .seh_endprologue
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   .seh_savexmm %xmm6,
; CHECK-NEXT:   movaps
; CHECK-NEXT:   .seh_savexmm %xmm7,
; CHECK-NEXT:   movaps
; CHECK-NEXT:   .seh_stackalloc
; CHECK-NEXT:   addq
; CHECK-NEXT:   .seh_endepilogue
; CHECK-NEXT:   retq

; ---- XMM callee-saved with GPR pushes: both must appear in epilog ----
define dso_local void @xmm_and_gpr_saves(ptr %p) local_unnamed_addr {
entry:
  call void asm sideeffect "", "~{xmm6},~{rbx}"()
  call void @a()
  ret void
}
; CHECK-LABEL:  xmm_and_gpr_saves:
; CHECK:        .seh_proc xmm_and_gpr_saves
; CHECK:        .seh_pushreg %rbx
; CHECK-NEXT:   pushq   %rbx
; CHECK:        .seh_stackalloc
; CHECK-NEXT:   subq
; CHECK:        .seh_savexmm %xmm6,
; CHECK-NEXT:   movaps
; CHECK:        .seh_endprologue
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   .seh_savexmm %xmm6,
; CHECK-NEXT:   movaps
; CHECK-NEXT:   .seh_stackalloc
; CHECK-NEXT:   addq
; CHECK-NEXT:   .seh_pushreg %rbx
; CHECK-NEXT:   popq    %rbx
; CHECK-NEXT:   .seh_endepilogue
; CHECK-NEXT:   retq

; ---- Large stack frame: uses __chkstk, SEH_StackAlloc before the sequence ----
define dso_local void @large_stack_alloc() local_unnamed_addr {
entry:
  %buf = alloca [8192 x i8], align 1
  call void @a()
  ret void
}
; CHECK-LABEL:  large_stack_alloc:
; CHECK:        .seh_proc large_stack_alloc
; CHECK:        .seh_stackalloc
; For V3, the .seh_stackalloc must appear before the stack allocation sequence.
; The allocation sequence includes movabs/movl + call __chkstk_ms + sub rsp,rax
; or just sub rsp, imm. Just check it comes before .seh_endprologue.
; CHECK:        .seh_endprologue

; ---- Large frame with dynamic alloc: epilog LEA offset exceeds 240 ----
; When a function uses VLAs (dynamic alloca) and has a large fixed frame, the
; prolog emits .seh_stackalloc + .seh_setframe separately, but the epilog
; restores the stack with a single LEA whose offset exceeds the 240-byte
; setframe limit. The epilog must split this into separate .seh_setframe
; (with the original prolog offset) and .seh_stackalloc, both pointing at
; the same LEA instruction.
define dso_local void @large_dynalloc_frame(i32 %n) local_unnamed_addr {
entry:
  %buf = alloca [4096 x i8], align 16
  %dyn = alloca i32, i32 %n
  call void @a()
  ret void
}
; CHECK-LABEL:  large_dynalloc_frame:
; CHECK:        .seh_proc large_dynalloc_frame
; CHECK:        .seh_pushreg %rbp
; CHECK-NEXT:   pushq   %rbp
; CHECK:        .seh_stackalloc
; CHECK:        .seh_setframe %rbp, 128
; CHECK:        .seh_endprologue
; Epilog: the setframe and stackalloc are split (not a single setframe > 240).
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   .seh_setframe %rbp, 128
; CHECK-NEXT:   .seh_stackalloc
; CHECK-NEXT:   leaq
; CHECK-NEXT:   .seh_pushreg %rbp
; CHECK-NEXT:   popq    %rbp
; CHECK-NEXT:   .seh_endepilogue
; CHECK-NEXT:   retq

; ---- XMM saves with dynamic alloca: SEH_SaveXMM must include SEHFrameOffset ----
; When a function has HasFP (from dynamic alloca) and XMM callee-saved registers,
; the SEH_SaveXMM offset in the epilog must include SEHFrameOffset to match the
; prolog.
define dso_local void @xmm_with_dynalloc(i32 %n) local_unnamed_addr {
entry:
  call void asm sideeffect "", "~{xmm6},~{xmm7}"()
  %dyn = alloca i32, i32 %n
  call void @a()
  ret void
}
; CHECK-LABEL:  xmm_with_dynalloc:
; CHECK:        .seh_proc xmm_with_dynalloc
; CHECK:        .seh_pushreg %rbp
; CHECK-NEXT:   pushq   %rbp
; CHECK:        .seh_stackalloc
; CHECK:        .seh_setframe %rbp,
; CHECK:        .seh_savexmm %xmm7, [[XMM7OFF:[0-9]+]]
; CHECK:        .seh_savexmm %xmm6, [[XMM6OFF:[0-9]+]]
; CHECK:        .seh_endprologue
; Epilog: XMM offsets must match the prolog offsets (both include SEHFrameOffset).
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   .seh_savexmm %xmm6, [[XMM6OFF]]
; CHECK:        .seh_savexmm %xmm7, [[XMM7OFF]]
; CHECK:        .seh_setframe %rbp,
; CHECK:        .seh_stackalloc
; CHECK:        .seh_pushreg %rbp
; CHECK-NEXT:   popq    %rbp
; CHECK-NEXT:   .seh_endepilogue
; CHECK-NEXT:   retq

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"winx64-eh-unwind", i32 3}
