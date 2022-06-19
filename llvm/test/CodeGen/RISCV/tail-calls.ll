; RUN: llc -mtriple riscv32-unknown-linux-gnu -o - %s | FileCheck %s
; RUN: llc -mtriple riscv32-unknown-elf       -o - %s | FileCheck %s

; Perform tail call optimization for global address.
declare i32 @callee_tail(i32 %i)
define i32 @caller_tail(i32 %i) nounwind {
; CHECK-LABEL: caller_tail
; CHECK: tail callee_tail
entry:
  %r = tail call i32 @callee_tail(i32 %i)
  ret i32 %r
}

; Perform tail call optimization for external symbol.
@dest = global [2 x i8] zeroinitializer
declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i1)
define void @caller_extern(i8* %src) optsize {
entry:
; CHECK: caller_extern
; CHECK-NOT: call memcpy
; CHECK: tail memcpy
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @dest, i32 0, i32 0), i8* %src, i32 7, i1 false)
  ret void
}

; Perform tail call optimization for external symbol.
@dest_pgso = global [2 x i8] zeroinitializer
define void @caller_extern_pgso(i8* %src) !prof !14 {
entry:
; CHECK: caller_extern_pgso
; CHECK-NOT: call memcpy
; CHECK: tail memcpy
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @dest_pgso, i32 0, i32 0), i8* %src, i32 7, i1 false)
  ret void
}

; Perform indirect tail call optimization (for function pointer call).
declare void @callee_indirect1()
declare void @callee_indirect2()
define void @caller_indirect_tail(i32 %a) nounwind {
; CHECK-LABEL: caller_indirect_tail
; CHECK-NOT: call callee_indirect1
; CHECK-NOT: call callee_indirect2
; CHECK-NOT: tail callee_indirect1
; CHECK-NOT: tail callee_indirect2

; CHECK: lui a0, %hi(callee_indirect2)
; CHECK-NEXT: addi t1, a0, %lo(callee_indirect2)
; CHECK-NEXT: jr t1

; CHECK: lui a0, %hi(callee_indirect1)
; CHECK-NEXT: addi t1, a0, %lo(callee_indirect1)
; CHECK-NEXT: jr t1
entry:
  %tobool = icmp eq i32 %a, 0
  %callee = select i1 %tobool, void ()* @callee_indirect1, void ()* @callee_indirect2
  tail call void %callee()
  ret void
}

; Make sure we don't use t0 as the source for jr as that is a hint to pop the
; return address stack on some microarchitectures.
define i32 @caller_indirect_no_t0(i32 (i32, i32, i32, i32, i32, i32, i32)* %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7) {
; CHECK-LABEL: caller_indirect_no_t0:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mv t1, a0
; CHECK-NEXT:    mv a0, a1
; CHECK-NEXT:    mv a1, a2
; CHECK-NEXT:    mv a2, a3
; CHECK-NEXT:    mv a3, a4
; CHECK-NEXT:    mv a4, a5
; CHECK-NEXT:    mv a5, a6
; CHECK-NEXT:    mv a6, a7
; CHECK-NEXT:    jr t1
  %9 = tail call i32 %0(i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7)
  ret i32 %9
}

; Do not tail call optimize functions with varargs passed by stack.
declare i32 @callee_varargs(i32, ...)
define void @caller_varargs(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: caller_varargs
; CHECK-NOT: tail callee_varargs
; CHECK: call callee_varargs
entry:
  %call = tail call i32 (i32, ...) @callee_varargs(i32 %a, i32 %b, i32 %b, i32 %a, i32 %a, i32 %b, i32 %b, i32 %a, i32 %a)
  ret void
}

; Do not tail call optimize if stack is used to pass parameters.
declare i32 @callee_args(i32 %a, i32 %b, i32 %c, i32 %dd, i32 %e, i32 %ff, i32 %g, i32 %h, i32 %i, i32 %j, i32 %k, i32 %l, i32 %m, i32 %n)
define i32 @caller_args(i32 %a, i32 %b, i32 %c, i32 %dd, i32 %e, i32 %ff, i32 %g, i32 %h, i32 %i, i32 %j, i32 %k, i32 %l, i32 %m, i32 %n) nounwind {
; CHECK-LABEL: caller_args
; CHECK-NOT: tail callee_args
; CHECK: call callee_args
entry:
  %r = tail call i32 @callee_args(i32 %a, i32 %b, i32 %c, i32 %dd, i32 %e, i32 %ff, i32 %g, i32 %h, i32 %i, i32 %j, i32 %k, i32 %l, i32 %m, i32 %n)
  ret i32 %r
}

; Do not tail call optimize if parameters need to be passed indirectly.
declare i32 @callee_indirect_args(fp128 %a)
define void @caller_indirect_args() nounwind {
; CHECK-LABEL: caller_indirect_args
; CHECK-NOT: tail callee_indirect_args
; CHECK: call callee_indirect_args
entry:
  %call = tail call i32 @callee_indirect_args(fp128 0xL00000000000000003FFF000000000000)
  ret void
}

; Externally-defined functions with weak linkage should not be tail-called.
; The behaviour of branch instructions in this situation (as used for tail
; calls) is implementation-defined, so we cannot rely on the linker replacing
; the tail call with a return.
declare extern_weak void @callee_weak()
define void @caller_weak() nounwind {
; CHECK-LABEL: caller_weak
; CHECK-NOT: tail callee_weak
; CHECK: call callee_weak
entry:
  tail call void @callee_weak()
  ret void
}

; Exception-handling functions need a special set of instructions to indicate a
; return to the hardware. Tail-calling another function would probably break
; this.
declare void @callee_irq()
define void @caller_irq() #0 {
; CHECK-LABEL: caller_irq
; CHECK-NOT: tail callee_irq
; CHECK: call callee_irq
entry:
  tail call void @callee_irq()
  ret void
}
attributes #0 = { "interrupt"="machine" }

; Byval parameters hand the function a pointer directly into the stack area
; we want to reuse during a tail call. Do not tail call optimize functions with
; byval parameters.
declare i32 @callee_byval(i32** byval(i32*) %a)
define i32 @caller_byval() nounwind {
; CHECK-LABEL: caller_byval
; CHECK-NOT: tail callee_byval
; CHECK: call callee_byval
entry:
  %a = alloca i32*
  %r = tail call i32 @callee_byval(i32** byval(i32*) %a)
  ret i32 %r
}

; Do not tail call optimize if callee uses structret semantics.
%struct.A = type { i32 }
@a = global %struct.A zeroinitializer

declare void @callee_struct(%struct.A* sret(%struct.A) %a)
define void @caller_nostruct() nounwind {
; CHECK-LABEL: caller_nostruct
; CHECK-NOT: tail callee_struct
; CHECK: call callee_struct
entry:
  tail call void @callee_struct(%struct.A* sret(%struct.A) @a)
  ret void
}

; Do not tail call optimize if caller uses structret semantics.
declare void @callee_nostruct()
define void @caller_struct(%struct.A* sret(%struct.A) %a) nounwind {
; CHECK-LABEL: caller_struct
; CHECK-NOT: tail callee_nostruct
; CHECK: call callee_nostruct
entry:
  tail call void @callee_nostruct()
  ret void
}

; Do not tail call optimize if disabled.
define i32 @disable_tail_calls(i32 %i) nounwind "disable-tail-calls"="true" {
; CHECK-LABEL: disable_tail_calls:
; CHECK-NOT: tail callee_nostruct
; CHECK: call callee_tail
entry:
  %rv = tail call i32 @callee_tail(i32 %i)
  ret i32 %rv
}

; Duplicate returns to enable tail call optimizations.
declare i32 @test()
declare i32 @test1()
declare i32 @test2()
declare i32 @test3()
define i32 @duplicate_returns(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: duplicate_returns:
; CHECK:    tail test2
; CHECK:    tail test
; CHECK:    tail test1
; CHECK:    tail test3
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call = tail call i32 @test()
  br label %return

if.else:                                          ; preds = %entry
  %cmp1 = icmp eq i32 %b, 0
  br i1 %cmp1, label %if.then2, label %if.else4

if.then2:                                         ; preds = %if.else
  %call3 = tail call i32 @test1()
  br label %return

if.else4:                                         ; preds = %if.else
  %cmp5 = icmp sgt i32 %a, %b
  br i1 %cmp5, label %if.then6, label %if.else8

if.then6:                                         ; preds = %if.else4
  %call7 = tail call i32 @test2()
  br label %return

if.else8:                                         ; preds = %if.else4
  %call9 = tail call i32 @test3()
  br label %return

return:                                           ; preds = %if.else8, %if.then6, %if.then2, %if.then
  %retval = phi i32 [ %call, %if.then ], [ %call3, %if.then2 ], [ %call7, %if.then6 ], [ %call9, %if.else8 ]
  ret i32 %retval
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 0}
