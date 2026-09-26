; REQUIRES: x86-registered-target || aarch64-registered-target
; RUN: %if x86-registered-target %{ llc -mtriple=x86_64-pc-windows-msvc -x86-asm-syntax=intel -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,X86 %}
; RUN: %if aarch64-registered-target %{ llc -mtriple=aarch64-pc-windows-msvc -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,AARCH64 %}
; RUN: %if x86-registered-target %{ llc -mtriple=x86_64-pc-windows-msvc -phi-elim-split-all-critical-edges -stop-after=phi-node-elimination -verify-machineinstrs < %s | FileCheck %s --check-prefix=PHI %}

declare void @llvm.seh.try.begin()
declare void @llvm.seh.try.end()
declare i32 @__C_specific_handler(...)

define void @guarded_loop(ptr %address) personality ptr @__C_specific_handler {
entry:
  invoke void @llvm.seh.try.begin()
          to label %loop unwind label %dispatch

loop:
  store volatile i32 7, ptr %address
  br label %loop

dispatch:
  %switch = catchswitch within none [label %handler] unwind to caller

handler:
  %pad = catchpad within %switch [ptr null]
  catchret from %pad to label %exit

exit:
  ret void
}

; CHECK-LABEL: guarded_loop:
; CHECK: %loop
; CHECK: [[BEGIN:\.Ltmp[0-9]+]]:
; X86: mov dword ptr [{{[^]]+}}], 7
; AARCH64: str {{w[0-9]+}}, [{{[^]]+}}]
; CHECK: {{(jmp|b)}} {{\.LBB[0-9]+_[0-9]+}}
; CHECK-NEXT: [[END:\.LBB_END[0-9]+_[0-9]+]]:
; CHECK: [[HANDLER:\.LBB[0-9]+_[0-9]+]]: {{.*}}%handler
; CHECK: .Llsda_begin{{[0-9]+}}:
; CHECK-NEXT: {{\.(long|word)}} [[BEGIN]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[END]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} 1
; CHECK-NEXT: {{\.(long|word)}} [[HANDLER]]@IMGREL
; CHECK-NEXT: .Llsda_end{{[0-9]+}}:

declare void @no_return() noreturn
declare i32 @callee(ptr)

define void @guarded_return(ptr %address) personality ptr @__C_specific_handler {
entry:
  invoke void @llvm.seh.try.begin() to label %body unwind label %dispatch
body:
  store volatile i32 7, ptr %address
  ret void
dispatch:
  %switch = catchswitch within none [label %handler] unwind to caller
handler:
  %pad = catchpad within %switch [ptr null]
  catchret from %pad to label %fail
fail:
  call void @no_return()
  unreachable
}

; CHECK-LABEL: guarded_return:
; CHECK: %body
; CHECK: [[RETURN_BEGIN:\.Ltmp[0-9]+]]:
; X86: mov dword ptr [{{[^]]+}}], 7
; AARCH64: str {{w[0-9]+}}, [{{[^]]+}}]
; CHECK-NEXT: [[RETURN_END:\.Ltmp[0-9]+]]:
; CHECK-NEXT: {{(#|//)}}SEH_REGION_BARRIER
; CHECK-NEXT: .seh_startepilogue
; CHECK: ret
; CHECK: [[RETURN_HANDLER:\.LBB[0-9]+_[0-9]+]]: {{.*}}%handler
; CHECK: .Llsda_begin{{[0-9]+}}:
; CHECK-NEXT: {{\.(long|word)}} [[RETURN_BEGIN]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[RETURN_END]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} 1
; CHECK-NEXT: {{\.(long|word)}} [[RETURN_HANDLER]]@IMGREL
; CHECK-NEXT: .Llsda_end{{[0-9]+}}:

define i32 @split_region(ptr %address, i1 %condition) personality ptr @__C_specific_handler {
entry:
  %isnull = icmp eq ptr %address, null
  br i1 %isnull, label %exit, label %try.begin
try.begin:
  invoke void @llvm.seh.try.begin() to label %try.body unwind label %dispatch
try.body:
  %isnull2 = icmp eq ptr %address, null
  br i1 %isnull2, label %then, label %else
then:
  store volatile i32 0, ptr %address
  br label %join
else:
  %called = invoke i32 @callee(ptr %address) to label %cont unwind label %dispatch
cont:
  %value = load volatile i64, ptr %address
  %select = select i1 %condition, i64 %value, i64 0
  store volatile i64 %select, ptr %address
  br label %join
join:
  invoke void @llvm.seh.try.end() to label %exit unwind label %dispatch
dispatch:
  %switch = catchswitch within none [label %handler] unwind to caller
handler:
  %pad = catchpad within %switch [ptr null]
  catchret from %pad to label %exit
exit:
  %result = phi i32 [ 0, %join ], [ 1, %entry ], [ -1, %handler ]
  ret i32 %result
}

; X86-LABEL: split_region:
; X86: %try.body
; X86: [[SPLIT_BEGIN:\.Ltmp[0-9]+]]:
; X86: call callee
; X86: %cont
; X86: mov qword ptr [{{[^]]+}}], {{[a-z0-9]+}}
; X86: jmp
; X86-NEXT: [[SPLIT_END:\.LBB_END[0-9]+_[0-9]+]]:
; X86-NEXT: {{\.LBB[0-9]+_[0-9]+}}: {{.*}}%entry
; X86-NEXT: mov eax, 1
; X86: %then
; X86: [[RESUME:\.Ltmp[0-9]+]]:
; X86: mov dword ptr [{{[^]]+}}], 0
; X86: %join
; X86: xor eax, eax
; X86-NEXT: [[RESUME_END:\.Ltmp[0-9]+]]:
; X86: [[SPLIT_HANDLER:\.LBB[0-9]+_[0-9]+]]: {{.*}}%handler
; X86: .Llsda_begin{{[0-9]+}}:
; X86-NEXT: .long [[SPLIT_BEGIN]]@IMGREL
; X86-NEXT: .long [[SPLIT_END]]@IMGREL
; X86-NEXT: .long 1
; X86-NEXT: .long [[SPLIT_HANDLER]]@IMGREL
; X86-NEXT: .long [[RESUME]]@IMGREL
; X86-NEXT: .long [[RESUME_END]]@IMGREL
; X86-NEXT: .long 1
; X86-NEXT: .long [[SPLIT_HANDLER]]@IMGREL
; X86-NEXT: .Llsda_end{{[0-9]+}}:

define i32 @sunk_load_stays_outside(ptr %address, ptr %source, i1 %condition) personality ptr @__C_specific_handler {
entry:
  %value = load i32, ptr %source
  br i1 %condition, label %try.begin, label %exit
try.begin:
  invoke void @llvm.seh.try.begin() to label %body unwind label %dispatch
body:
  %sum = add i32 %value, 1
  store volatile i32 %sum, ptr %address
  %called = invoke i32 @callee(ptr %address) to label %end unwind label %dispatch
end:
  invoke void @llvm.seh.try.end() to label %exit unwind label %dispatch
dispatch:
  %switch = catchswitch within none [label %handler] unwind to caller
handler:
  %pad = catchpad within %switch [ptr null]
  catchret from %pad to label %exit
exit:
  %result = phi i32 [ 0, %entry ], [ 1, %end ], [ -1, %handler ]
  ret i32 %result
}

; CHECK-LABEL: sunk_load_stays_outside:
; X86: mov {{[a-z0-9]+}}, dword ptr [{{[a-z0-9]+}}]
; AARCH64: ldr {{w[0-9]+}}, [{{x[0-9]+}}]
; CHECK: {{(#|//)}}SEH_REGION_BARRIER
; CHECK-NEXT: [[SINK_BEGIN:\.Ltmp[0-9]+]]:
; CHECK: {{(call|bl)}} callee
; CHECK: .Llsda_begin{{[0-9]+}}:
; CHECK-NEXT: {{\.(long|word)}} [[SINK_BEGIN]]@IMGREL

define i32 @critical_region_edge(ptr %address, i1 %condition) personality ptr @__C_specific_handler {
entry:
  invoke void @llvm.seh.try.begin() to label %body unwind label %dispatch
body:
  %value = load volatile i32, ptr %address
  br i1 %condition, label %merge, label %other
other:
  %add = add i32 %value, 7
  br label %merge
merge:
  %merged = phi i32 [ %value, %body ], [ %add, %other ]
  invoke void @llvm.seh.try.end() to label %exit unwind label %dispatch
dispatch:
  %switch = catchswitch within none [label %handler] unwind to caller
handler:
  %pad = catchpad within %switch [ptr null]
  catchret from %pad to label %exit
exit:
  %result = phi i32 [ %merged, %merge ], [ -1, %handler ]
  ret i32 %result
}

; PHI-LABEL: name: critical_region_edge
; PHI: bb.1.body:
; PHI: bb.{{[0-9]+}}.body:
; PHI: SEH_REGION_BARRIER
; PHI-NEXT: EH_LABEL
; PHI: COPY
; PHI-NEXT: EH_LABEL
; PHI-NEXT: SEH_REGION_BARRIER

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"eh-asynch", i32 1}