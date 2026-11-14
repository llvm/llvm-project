; RUN: split-file %s %t
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs < %t/base.ll | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs -stop-after=kcfi < %t/base.ll | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs -verify-cfiinstrs -function-sections -basic-block-sections=all -basic-block-address-map -o %t.bbs.s < %t/base.ll
; RUN: FileCheck %s --check-prefix=BBS < %t.bbs.s
; RUN: llvm-mc -triple=x86_64-unknown-linux-gnu -filetype=obj -o /dev/null < %t.bbs.s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs -verify-cfiinstrs -function-sections -basic-block-sections=%t/prof -basic-block-address-map -o %t.prop.s 2> %t.prop.err < %t/base.ll
; RUN: FileCheck %s --check-prefix=PROP < %t.prop.s
; RUN: FileCheck %s --check-prefix=PROP-WARN < %t.prop.err
; RUN: llvm-mc -triple=x86_64-unknown-linux-gnu -filetype=obj -o /dev/null < %t.prop.s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -emulated-tls -verify-machineinstrs -verify-cfiinstrs -function-sections -basic-block-sections=all -o %t.emutls.s < %t/emutls.ll
; RUN: FileCheck %s --check-prefix=EMUTLS < %t.emutls.s
; RUN: llvm-mc -triple=x86_64-unknown-linux-gnu -filetype=obj -o /dev/null < %t.emutls.s

;; Indirect invokes move their KCFI traps out of line, keeping the invoke's
;; BeginLabel/EndLabel EH_LABEL pair and DW_CFA_GNU_args_size instruction with
;; the call in the split block so LSDA call-site ranges and FDEs stay valid
;; when basic block sections are enabled.

;--- base.ll
declare i32 @__gxx_personality_v0(...)
declare void @cleanup()
declare void @cleanup_8args(i64, i64, i64, i64, i64, i64, i64, i64)
declare void @side(i32)

define void @f1(ptr noundef %x) personality ptr @__gxx_personality_v0 {
; ASM-LABEL: f1:
; ASM:         movl $4282621618, %r10d # imm = 0xFF439EB2
; ASM-NEXT:    addl -4(%rdi), %r10d
; ASM-NEXT:    jne [[TRAP:.LBB[0-9_]+]]
; ASM-NEXT:  # %bb.{{[0-9]+}}:
; ASM-NEXT:  .Ltmp{{[0-9]+}}: # EH_LABEL
; ASM-NEXT:    callq *%rdi
; ASM-NEXT:  .Ltmp{{[0-9]+}}: # EH_LABEL
; ASM:       [[TRAP]]:
; ASM-NEXT:    .cfi_def_cfa_offset 16
; ASM-NEXT:  .Ltmp{{[0-9]+}}:
; ASM-NEXT:    ud2

; MIR-LABEL: name: f1
; MIR:       bb.0 (%ir-block.0):
; MIR-NEXT:    successors: %bb.4(0x80000000), %bb.3(0x00000000)
; MIR:         KCFI_CMP $rdi, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; MIR-NEXT:    JCC_1 %bb.3, 5, implicit $eflags
; MIR:       bb.4 (%ir-block.0):
; MIR-NEXT:    successors: %bb.1(0x7ffff800), %bb.2(0x00000800)
; MIR-NEXT:    liveins: {{.*}}$rdi
; MIR:         EH_LABEL
; MIR-NEXT:    CALL64r killed $rdi, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp
; MIR-NEXT:    EH_LABEL
; MIR:       bb.3 (%ir-block.0):
; MIR-NEXT:    nomerge KCFI_TRAP

; BBS-LABEL: f1:
; BBS:         addl -4(%rdi), %r10d
; BBS-NEXT:    jne [[TRAP:f1\.__part\.[0-9]+]]
; BBS-NEXT:    jmp [[CALL:f1\.__part\.[0-9]+]]
; BBS:       [[CALL]]:
; BBS:       [[BEGIN:\.Ltmp[0-9]+]]: # EH_LABEL
; BBS-NEXT:    callq *%rdi
; BBS-NEXT:  .LBB{{[0-9]+}}_1_CS0:
; BBS-NEXT:  [[END:\.Ltmp[0-9]+]]: # EH_LABEL
; BBS:       [[TRAP]]:
; BBS:         ud2
; BBS:       [[ERR:f1\.__part\.[0-9]+]]: # %err
; BBS:       [[LPAD:\.Ltmp[0-9]+]]: # EH_LABEL
; BBS:       .section .gcc_except_table.f1
; BBS:         .uleb128 [[BEGIN]]-[[CALL]]
; BBS-NEXT:    .uleb128 [[END]]-[[BEGIN]]
; BBS-NEXT:    .uleb128 [[LPAD]]-[[ERR]]
  invoke void %x() [ "kcfi"(i32 12345678) ]
    to label %cont
    unwind label %err
cont:
  ret void
err:
  %exn = landingpad { ptr, i32 }
          cleanup
  call void @cleanup()
  resume { ptr, i32 } %exn
}

;; Stack-pushed call arguments carry DW_CFA_GNU_args_size (.cfi_escape 0x2e, 0x10)
;; into the split invoke block's FDE when basic block sections are enabled.
define void @f2(ptr noundef %x) personality ptr @__gxx_personality_v0 {
; BBS-LABEL: f2:
; BBS:         pushq $8
; BBS-NEXT:    .cfi_adjust_cfa_offset 8
; BBS-NEXT:    pushq $7
; BBS-NEXT:    .cfi_adjust_cfa_offset 8
; BBS-NEXT:    movl $4282621618, %r10d # imm = 0xFF439EB2
; BBS-NEXT:    addl -4([[REG:%r[a-z0-9]+]]), %r10d
; BBS-NEXT:    jne [[TRAP:f2\.__part\.[0-9]+]]
; BBS-NEXT:    jmp [[CALL:f2\.__part\.[0-9]+]]
; BBS:       [[CALL]]:
; BBS-NEXT:    .cfi_startproc
; BBS:         .cfi_def_cfa %rsp, 32
; BBS:       [[BEGIN:\.Ltmp[0-9]+]]: # EH_LABEL
; BBS-NEXT:    .cfi_escape 0x2e, 0x10
; BBS-NEXT:    callq *[[REG]]
; BBS-NEXT:  .LBB{{[0-9]+}}_1_CS0:
; BBS-NEXT:    addq $16, %rsp
; BBS-NEXT:    .cfi_adjust_cfa_offset -16
; BBS-NEXT:  [[END:\.Ltmp[0-9]+]]: # EH_LABEL
; BBS:       [[TRAP]]:
; BBS-NEXT:    .cfi_startproc
; BBS:         .cfi_def_cfa %rsp, 32
; BBS:         ud2
; BBS:       .section .gcc_except_table.f2
; BBS:         .uleb128 [[BEGIN]]-[[CALL]]
; BBS-NEXT:    .uleb128 [[END]]-[[BEGIN]]
  invoke void %x(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8) [ "kcfi"(i32 12345678) ]
    to label %cont
    unwind label %err
cont:
  ret void
err:
  %exn = landingpad { ptr, i32 }
          cleanup
  call void @cleanup()
  resume { ptr, i32 } %exn
}

;; Checked calls inside a landing pad keep inline traps so the landing pad
;; EH_LABEL stays at the entry of its section without extra splitting.
define void @f3(ptr noundef %x) personality ptr @__gxx_personality_v0 {
; BBS-LABEL: f3:
; BBS:       [[BEGIN:\.Ltmp[0-9]+]]: # EH_LABEL
; BBS:         callq cleanup_8args@PLT
; BBS:       [[END:\.Ltmp[0-9]+]]: # EH_LABEL
; BBS:       [[ERR:f3\.__part\.[0-9]+]]: # %err
; BBS:         nop
; BBS-NEXT:  [[LPAD:\.Ltmp[0-9]+]]: # EH_LABEL
; BBS:         addl -4([[REG:%r[a-z0-9]+]]), %r10d
; BBS-NEXT:    je [[PASS:\.Ltmp[0-9]+]]
; BBS:         ud2
; BBS:       [[PASS]]:
; BBS-NEXT:    callq *[[REG]]
; BBS:       .section .gcc_except_table.f3
; BBS:         .uleb128 [[BEGIN]]-.Lfunc_begin
; BBS-NEXT:    .uleb128 [[END]]-[[BEGIN]]
; BBS-NEXT:    .uleb128 [[LPAD]]-[[ERR]]
  invoke void @cleanup_8args(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
    to label %cont
    unwind label %err
cont:
  ret void
err:
  %exn = landingpad { ptr, i32 }
          cleanup
  call void %x() [ "kcfi"(i32 12345678) ]
  resume { ptr, i32 } %exn
}

;; Checked call followed by checked invoke in the same block, exercised with a
;; Propeller profile that inserts code prefetch hints/targets on the split
;; invoke block and attempts path cloning of the KCFI_CMP block (which
;; BasicBlockPathCloning rejects because KCFI_CMP is marked isNotDuplicable).
define void @f4(i1 %cond, ptr noundef %x, ptr noundef %y) personality ptr @__gxx_personality_v0 {
; MIR-LABEL: name: f4
; MIR:       bb.3.join:
; MIR-NEXT:    successors: %bb.9(0x80000000), %bb.8(0x00000000)
; MIR:         CALL64pcrel32 target-flags(x86-plt) @side
; MIR-NEXT:    KCFI_CMP $r14, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; MIR-NEXT:    JCC_1 %bb.8, 5, implicit $eflags
; MIR:       bb.9.join:
; MIR-NEXT:    successors: %bb.7(0x80000000), %bb.6(0x00000000)
; MIR:         CALL64r killed $r14, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp
; MIR-NEXT:    KCFI_CMP $rbx, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; MIR-NEXT:    JCC_1 %bb.6, 5, implicit $eflags
; MIR:       bb.7.join:
; MIR-NEXT:    successors: %bb.4(0x7ffff800), %bb.5(0x00000800)
; MIR:         EH_LABEL
; MIR-NEXT:    CALL64r killed $rbx, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp
; MIR-NEXT:    EH_LABEL

; PROP-LABEL: f4:
; PROP:       [[FNBEGIN:\.Lfunc_begin[0-9]+]]:
; PROP:       [[JOIN:\.LBB[0-9_]+]]: # %join
; PROP:         addl -4(%r14), %r10d
; PROP-NEXT:    jne f4.cold
; PROP:         callq *%r14
; PROP:         addl -4(%rbx), %r10d
; PROP-NEXT:    jne .LBB{{[0-9_]+}}
; PROP:       [[BEGIN:\.Ltmp[0-9]+]]: # EH_LABEL
; PROP-NEXT:    callq *%rbx
; PROP:         .globl __llvm_prefetch_target_f4_8_1
; PROP-NEXT:  __llvm_prefetch_target_f4_8_1:
; PROP-NEXT:    prefetchit1 __llvm_prefetch_target_side_0_0(%rip)
; PROP:       [[END:\.Ltmp[0-9]+]]: # EH_LABEL
; PROP:         retq
; PROP:       [[RIGHT:f4\.__part\.[0-9]+]]: # %right
; PROP:         movl $2, %edi
; PROP-NEXT:    jmp [[JOIN]]
; PROP:       f4.cold: # %join
; PROP:       .Ltmp{{[0-9]+}}:
; PROP-NEXT:    ud2
; PROP:       .LBB{{[0-9_]+}}: # %join
; PROP-NEXT:  .Ltmp{{[0-9]+}}:
; PROP-NEXT:    ud2
; PROP:       [[LPAD:\.Ltmp[0-9]+]]: # EH_LABEL
; PROP:       .section .gcc_except_table.f4
; PROP:         .uleb128 [[BEGIN]]-[[FNBEGIN]] # >> Call Site 2 <<
; PROP-NEXT:    .uleb128 [[END]]-[[BEGIN]]
; PROP-NEXT:    .uleb128 [[LPAD]]-f4.cold

; PROP-WARN: warning: block #6 has non-duplicable instructions in function f4
entry:
  br i1 %cond, label %left, label %right
left:
  call void @side(i32 1)
  br label %join
right:
  call void @side(i32 2)
  br label %join
join:
  call void %x() [ "kcfi"(i32 12345678) ]
  invoke void %y() [ "kcfi"(i32 12345678) ]
    to label %cont
    unwind label %err
cont:
  ret void
err:
  %exn = landingpad { ptr, i32 }
          cleanup
  call void @cleanup()
  resume { ptr, i32 } %exn
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 8, !"kcfi-out-of-line-traps", i32 1}

;--- prof
v1
f f4
p 1 6 10
c 0 1 6 10 8 4
c 2
t 8,1
i 8,1 side,0,0

;--- emutls.ll
;; An invoke whose argument lowering inserts a call (such as emulated TLS)
;; between BeginLabel and the indirect call keeps an inline trap so
;; BeginLabel is not stranded across a KCFI block split.
@tls_var = thread_local global i32 0

declare i32 @__gxx_personality_v0(...)
declare void @cleanup()

define void @f5(ptr noundef %x) personality ptr @__gxx_personality_v0 {
; EMUTLS-LABEL: f5:
; EMUTLS:       [[FNBEGIN:\.Lfunc_begin[0-9]+]]:
; EMUTLS:       [[BEGIN:\.Ltmp[0-9]+]]: # EH_LABEL
; EMUTLS:         callq __emutls_get_address@PLT
; EMUTLS:         addl -4(%rbx), %r10d
; EMUTLS-NEXT:    je [[PASS:\.Ltmp[0-9]+]]
; EMUTLS-NEXT:  .Ltmp{{[0-9]+}}:
; EMUTLS-NEXT:    ud2
; EMUTLS:       [[PASS]]:
; EMUTLS-NEXT:    callq *%rbx
; EMUTLS-NEXT:  [[END:\.Ltmp[0-9]+]]: # EH_LABEL
; EMUTLS:       [[ERR:f5\.__part\.[0-9]+]]: # %err
; EMUTLS:       [[LPAD:\.Ltmp[0-9]+]]: # EH_LABEL
; EMUTLS:       .section .gcc_except_table.f5
; EMUTLS:         .uleb128 [[BEGIN]]-[[FNBEGIN]]
; EMUTLS-NEXT:    .uleb128 [[END]]-[[BEGIN]]
; EMUTLS-NEXT:    .uleb128 [[LPAD]]-[[ERR]]
  invoke void %x(ptr @tls_var) [ "kcfi"(i32 12345678) ]
    to label %cont
    unwind label %err
cont:
  ret void
err:
  %exn = landingpad { ptr, i32 }
          cleanup
  call void @cleanup()
  resume { ptr, i32 } %exn
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 8, !"kcfi-out-of-line-traps", i32 1}
