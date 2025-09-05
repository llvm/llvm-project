; RUN: llc -mtriple=x86_64-unknown-windows-msvc -o - %s | FileCheck %s

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
; CHECK:        .seh_unwindversion 2
; CHECK-NOT:    .seh_pushreg
; CHECK:        .seh_stackalloc
; CHECK:        .seh_endprologue
; CHECK-NOT:    .seh_endproc
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   addq
; CHECK-NEXT:   .seh_unwindv2start
; CHECK-NEXT:   .seh_endepilogue
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
; CHECK:        .seh_unwindversion 2
; CHECK:        .seh_pushreg %rsi
; CHECK:        .seh_pushreg %rdi
; CHECK:        .seh_pushreg %rbx
; CHECK:        .seh_stackalloc
; CHECK:        .seh_endprologue
; CHECK-NOT:    .seh_endproc
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   addq
; CHECK-NEXT:   .seh_unwindv2start
; CHECK-NEXT:   popq    %rbx
; CHECK-NEXT:   popq    %rdi
; CHECK-NEXT:   popq    %rsi
; CHECK-NEXT:   .seh_endepilogue
; CHECK-NEXT:   retq

define dso_local i32 @tail_call(i32 %x) local_unnamed_addr {
entry:
  %call = tail call i32 @c(i32 %x)
  %call1 = tail call i32 @c(i32 %call)
  ret i32 %call1
}
; CHECK-LABEL:  tail_call:
; CHECK:        .seh_unwindversion 2
; CHECK-NOT:    .seh_pushreg
; CHECK:        .seh_stackalloc
; CHECK:        .seh_endprologue
; CHECK-NOT:    .seh_endproc
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   addq
; CHECK-NEXT:   .seh_unwindv2start
; CHECK-NEXT:   .seh_endepilogue
; CHECK-NEXT:   jmp

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
; CHECK:        .seh_unwindversion 2
; CHECK-NOT:    .seh_pushreg
; CHECK:        .seh_stackalloc
; CHECK:        .seh_endprologue
; CHECK-NOT:    .seh_endproc
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   addq
; CHECK-NEXT:   .seh_unwindv2start
; CHECK-NEXT:   .seh_endepilogue
; CHECK-NEXT:   jmp
; CHECK-NOT:    .seh_endproc
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   addq
; CHECK-NEXT:   .seh_unwindv2start
; CHECK-NEXT:   .seh_endepilogue
; CHECK-NEXT:   jmp

define dso_local i32 @mismatched_terminators() local_unnamed_addr {
entry:
  %call = tail call i32 @b()
  %cmp = icmp sgt i32 %call, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %call1 = tail call i32 @b()
  ret i32 %call1

if.else:
  ret i32 %call
}
; CHECK-LABEL:  mismatched_terminators:
; CHECK:        .seh_unwindversion 2
; CHECK-NOT:    .seh_pushreg
; CHECK:        .seh_stackalloc
; CHECK:        .seh_endprologue
; CHECK-NOT:    .seh_endproc
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   addq
; CHECK-NEXT:   .seh_unwindv2start
; CHECK-NEXT:   .seh_endepilogue
; CHECK-NEXT:   jmp
; CHECK-NOT:    .seh_endproc
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   addq
; CHECK-NEXT:   .seh_unwindv2start
; CHECK-NEXT:   .seh_endepilogue
; CHECK-NEXT:   ret

define dso_local void @dynamic_stack_alloc(i32 %x) local_unnamed_addr {
entry:
  %y = alloca i32, i32 %x
  ret void
}
; CHECK-LABEL:  dynamic_stack_alloc:
; CHECK:        .seh_unwindversion 2
; CHECK:        .seh_pushreg %rbp
; CHECK:        .seh_setframe %rbp, 0
; CHECK:        .seh_endprologue
; CHECK-NOT:    .seh_endproc
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   movq    %rbp, %rsp
; CHECK-NEXT:   .seh_unwindv2start
; CHECK-NEXT:   popq    %rbp
; CHECK-NEXT:   .seh_endepilogue
; CHECK-NEXT:   retq
; CHECK-NEXT:   .seh_endproc

define dso_local void @large_aligned_alloc() align 16 {
  %1 = alloca [128 x i8], align 64
  ret void
}
; CHECK-LABEL:  large_aligned_alloc:
; CHECK:        .seh_unwindversion 2
; CHECK:        .seh_pushreg %rbp
; CHECK:        .seh_stackalloc 176
; CHECK:        .seh_setframe %rbp, 128
; CHECK:        .seh_endprologue
; CHECK-NOT:    .seh_endproc
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   leaq    48(%rbp), %rsp
; CHECK-NEXT:   .seh_unwindv2start
; CHECK-NEXT:   popq    %rbp
; CHECK-NEXT:   .seh_endepilogue
; CHECK-NEXT:   retq
; CHECK-NEXT:   .seh_endproc

define dso_local void @set_frame_only() local_unnamed_addr {
  tail call i64 @llvm.x86.flags.read.u64()
  ret void
}

; CHECK-LABEL:  set_frame_only:
; CHECK:        .seh_unwindversion 2
; CHECK:        .seh_pushreg %rbp
; CHECK:        .seh_setframe %rbp, 0
; CHECK:        .seh_endprologue
; CHECK-NOT:    .seh_endproc
; CHECK:        .seh_startepilogue
; CHECK-NEXT:   .seh_unwindv2start
; CHECK-NEXT:   popq    %rbp
; CHECK-NEXT:   .seh_endepilogue
; CHECK-NEXT:   retq
; CHECK-NEXT:   .seh_endproc

attributes #1 = { noreturn }
define dso_local void @no_return_func() local_unnamed_addr #1 {
entry:
  call void @d()
  unreachable
}
; CHECK-LABEL:  no_return_func:
; CHECK-NOT:    .seh_unwindversion 2
; CHECK:        .seh_stackalloc
; CHECK-NEXT:   .seh_endprologue
; CHECK-NOT:    .seh_startepilogue
; CHECK-NOT:    .seh_unwindv2start
; CHECK:        int3
; CHECK-NEXT:   .seh_endproc

declare i64 @llvm.x86.flags.read.u64()
declare void @a() local_unnamed_addr
declare i32 @b() local_unnamed_addr
declare i32 @c(i32) local_unnamed_addr
declare void @d() local_unnamed_addr #1

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"winx64-eh-unwindv2", i32 2}
