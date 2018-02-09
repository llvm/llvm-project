; RUN: llc -verify-machineinstrs < %s 2>&1 | FileCheck %s

target triple = "x86_64-pc-linux-gnu"

declare void @"some_call"(i64 addrspace(1)*)
declare i64 addrspace(1)* @"some_other_call"(i64 addrspace(1)*)

declare i32 @"personality_function"()

define i64 addrspace(1)* @test_basic(i64 addrspace(1)* %obj,
                                     i64 addrspace(1)* %obj1)
gc "statepoint-example" personality i32 ()* @"personality_function" {
entry:
  ; CHECK: Ltmp{{[0-9]+}}:
  ; CHECK: callq some_call
  ; CHECK: Ltmp{{[0-9]+}}:
  %0 = invoke token (i64, i32, void (i64 addrspace(1)*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidp1i64f(i64 0, i32 0, void (i64 addrspace(1)*)* @some_call, i32 1, i32 0, i64 addrspace(1)* %obj, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i64 addrspace(1)* %obj, i64 addrspace(1)* %obj1)
          to label %invoke_safepoint_normal_dest unwind label %exceptional_return

invoke_safepoint_normal_dest:
  ; CHECK: movq
  %obj.relocated = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %0, i32 13, i32 13)
  %obj1.relocated = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %0, i32 14, i32 14)
  br label %normal_return

normal_return:
  ; CHECK: retq
  ret i64 addrspace(1)* %obj.relocated

exceptional_return:
  ; CHECK: Ltmp{{[0-9]+}}:
  ; CHECK: movq
  ; CHECK: retq
  %landing_pad = landingpad token
          cleanup
  %obj.relocated1 = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %landing_pad, i32 13, i32 13)
  %obj1.relocated1 = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %landing_pad, i32 14, i32 14)
  ret i64 addrspace(1)* %obj1.relocated1
}
; CHECK-LABEL: GCC_except_table{{[0-9]+}}:
; CHECK: .uleb128  .Ltmp{{[0-9]+}}-.Ltmp{{[0-9]+}}
; CHECK: .uleb128  .Ltmp{{[0-9]+}}-.Lfunc_begin{{[0-9]+}}
; CHECK: .byte  0
; CHECK: .p2align 4

define i64 addrspace(1)* @test_result(i64 addrspace(1)* %obj,
                                      i64 addrspace(1)* %obj1)
  gc "statepoint-example" personality i32 ()* @personality_function {
entry:
  ; CHECK: .Ltmp{{[0-9]+}}:
  ; CHECK: callq some_other_call
  ; CHECK: .Ltmp{{[0-9]+}}:
  %0 = invoke token (i64, i32, i64 addrspace(1)* (i64 addrspace(1)*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_p1i64p1i64f(i64 0, i32 0, i64 addrspace(1)* (i64 addrspace(1)*)* @some_other_call, i32 1, i32 0, i64 addrspace(1)* %obj, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i64 addrspace(1)* %obj, i64 addrspace(1)* %obj1)
          to label %normal_return unwind label %exceptional_return

normal_return:
  ; CHECK: popq 
  ; CHECK: retq
  %ret_val = call i64 addrspace(1)* @llvm.experimental.gc.result.p1i64(token %0)
  ret i64 addrspace(1)* %ret_val

exceptional_return:
  ; CHECK: .Ltmp{{[0-9]+}}:
  ; CHECK: movq
  %landing_pad = landingpad token
          cleanup
  %obj.relocated = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %landing_pad, i32 13, i32 13)
  ret i64 addrspace(1)* %obj.relocated
}
; CHECK-LABEL: GCC_except_table{{[0-9]+}}:
; CHECK: .uleb128 .Ltmp{{[0-9]+}}-.Ltmp{{[0-9]+}}
; CHECK: .uleb128 .Ltmp{{[0-9]+}}-.Lfunc_begin{{[0-9]+}}
; CHECK: .byte 0
; CHECK: .p2align 4

define i64 addrspace(1)* @test_same_val(i1 %cond, i64 addrspace(1)* %val1, i64 addrspace(1)* %val2, i64 addrspace(1)* %val3)
  gc "statepoint-example" personality i32 ()* @"personality_function" {
entry:
  br i1 %cond, label %left, label %right

left:
  ; CHECK-LABEL: %left
  ; CHECK: movq %rdx, 8(%rsp)
  ; CHECK: movq
  ; CHECK: callq some_call
  %sp1 = invoke token (i64, i32, void (i64 addrspace(1)*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidp1i64f(i64 0, i32 0, void (i64 addrspace(1)*)* @some_call, i32 1, i32 0, i64 addrspace(1)* %val1, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i64 addrspace(1)* %val1, i64 addrspace(1)* %val2)
           to label %left.relocs unwind label %exceptional_return.left

left.relocs:
  ; CHECK: movq (%rsp),
  ; CHECK: movq 8(%rsp), [[REGVAL2:%[a-z]+]]
  %val1.relocated = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %sp1, i32 13, i32 13)
  %val2.relocated_left = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %sp1, i32 14, i32 14)
  br label %normal_return

right:
  ; CHECK-LABEL: %right
  ; CHECK: movq %rdx, (%rsp)
  ; CHECK: movq
  ; CHECK: callq some_call
  %sp2 = invoke token (i64, i32, void (i64 addrspace(1)*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidp1i64f(i64 0, i32 0, void (i64 addrspace(1)*)* @some_call, i32 1, i32 0, i64 addrspace(1)* %val1, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i64 addrspace(1)* %val2, i64 addrspace(1)* %val3)
           to label %right.relocs unwind label %exceptional_return.right

right.relocs:
  ; CHECK: movq (%rsp), [[REGVAL2]]
  ; CHECK: movq
  %val2.relocated_right = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %sp2, i32 13, i32 13)
  %val3.relocated = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %sp2, i32 14, i32 14)
  br label %normal_return

normal_return:
  ; CHECK-LABEL: %normal_return
  ; CHECK: cmoveq {{.*}}[[REGVAL2]]{{.*}}
  ; CHECK: retq
  %a1 = phi i64 addrspace(1)* [%val1.relocated, %left.relocs], [%val3.relocated, %right.relocs]
  %a2 = phi i64 addrspace(1)* [%val2.relocated_left, %left.relocs], [%val2.relocated_right, %right.relocs]
  %ret = select i1 %cond, i64 addrspace(1)* %a1, i64 addrspace(1)* %a2
  ret i64 addrspace(1)* %ret

exceptional_return.left:
  %landing_pad = landingpad token
          cleanup
  %val.relocated2 = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %landing_pad, i32 13, i32 13)
  ret i64 addrspace(1)* %val.relocated2

exceptional_return.right:
  %landing_pad1 = landingpad token
          cleanup
  %val.relocated3 = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %landing_pad1, i32 13, i32 13)
  ret i64 addrspace(1)* %val.relocated3
}

define i64 addrspace(1)* @test_null_undef(i64 addrspace(1)* %val1)
       gc "statepoint-example" personality i32 ()* @"personality_function" {
; CHECK-LABEL: test_null_undef:
entry:
  ; CHECK: callq some_call
  %sp1 = invoke token (i64, i32, void (i64 addrspace(1)*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidp1i64f(i64 0, i32 0, void (i64 addrspace(1)*)* @some_call, i32 1, i32 0, i64 addrspace(1)* %val1, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i64 addrspace(1)* null, i64 addrspace(1)* undef)
           to label %normal_return unwind label %exceptional_return

normal_return:
  ; CHECK-LABEL: %normal_return
  ; CHECK: xorl %eax, %eax
  ; CHECK-NEXT: popq
  ; CHECK-NEXT: retq
  %null.relocated = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %sp1, i32 13, i32 13)
  %undef.relocated = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %sp1, i32 14, i32 14)
  ret i64 addrspace(1)* %null.relocated

exceptional_return:
  %landing_pad = landingpad token
          cleanup
  %null.relocated2 = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %landing_pad, i32 13, i32 13)
  %undef.relocated2 = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %landing_pad, i32 14, i32 14)
  ret i64 addrspace(1)* %null.relocated2
}

define i64 addrspace(1)* @test_alloca_and_const(i64 addrspace(1)* %val1)
       gc "statepoint-example" personality i32 ()* @"personality_function" {
; CHECK-LABEL: test_alloca_and_const:
entry:
  %a = alloca i32
  %aa = addrspacecast i32* %a to i32 addrspace(1)*
  %c = inttoptr i64 15 to i64 addrspace(1)*
  ; CHECK: callq
  %sp = invoke token (i64, i32, void (i64 addrspace(1)*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidp1i64f(i64 0, i32 0, void (i64 addrspace(1)*)* @some_call, i32 1, i32 0, i64 addrspace(1)* %val1, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i32 addrspace(1)* %aa, i64 addrspace(1)* %c)
           to label %normal_return unwind label %exceptional_return

normal_return:
  ; CHECK: leaq
  ; CHECK-NEXT: popq
  ; CHECK-NEXT: retq
  %aa.rel = call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %sp, i32 13, i32 13)
  %aa.converted = bitcast i32 addrspace(1)* %aa.rel to i64 addrspace(1)*
  ret i64 addrspace(1)* %aa.converted

exceptional_return:
  ; CHECK: movl	$15
  ; CHECK-NEXT: popq
  ; CHECK-NEXT: retq
  %landing_pad = landingpad token
          cleanup
  %aa.rel2 = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %landing_pad, i32 14, i32 14)
  ret i64 addrspace(1)* %aa.rel2
}

declare token @llvm.experimental.gc.statepoint.p0f_isVoidp1i64f(i64, i32, void (i64 addrspace(1)*)*, i32, i32, ...)
declare token @llvm.experimental.gc.statepoint.p0f_p1i64p1i64f(i64, i32, i64 addrspace(1)* (i64 addrspace(1)*)*, i32, i32, ...)

declare i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token, i32, i32)
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token, i32, i32)
declare i64 addrspace(1)* @llvm.experimental.gc.result.p1i64(token)
