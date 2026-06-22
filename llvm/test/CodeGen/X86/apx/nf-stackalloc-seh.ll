; Verify how APX NF (no-flags) ADD/SUB are used for the stack-pointer
; adjustment in the Windows x64 prologue/epilogue, and how that interacts with
; the unwind information version.
;
; The Windows prologue unwinder is data-driven: it walks the unwind codes and
; never disassembles the prologue, so an EVEX NF {nf} sub of RSP in the
; prologue is always safe (v1, v2 and v3 all emit it). The v1/v2 epilogue
; unwinder, however, disassembles the epilogue to recognize the canonical
; "add/lea rsp; pop...; ret" sequence, and does not understand EVEX NF add/sub
; -- so NF must NOT be used in a v1/v2 epilogue. Unwind v3 encodes epilog
; operations declaratively (no disassembly), so {nf} add of RSP is allowed in a
; v3 epilogue. The v1/v2 epilogue restriction only applies when the function
; actually emits unwind info; a nounwind function has no unwind table entry, so
; its epilogue is never disassembled and {nf} add is allowed there too.
;
; The unwind version is selected by a module-wide flag, so each case lives in
; its own split-file section.
;
; RUN: split-file %s %t
; RUN: llc < %t/v1.ll -mtriple=x86_64-windows-msvc -mattr=+nf | FileCheck %s --check-prefix=V1
; RUN: llc < %t/v2.ll -mtriple=x86_64-windows-msvc -mattr=+nf | FileCheck %s --check-prefix=V2
; RUN: llc < %t/v3.ll -mtriple=x86_64-windows-msvc -mattr=+nf | FileCheck %s --check-prefix=V3
; RUN: llc < %t/nounwind.ll -mtriple=x86_64-windows-msvc -mattr=+nf | FileCheck %s --check-prefix=NOUNWIND
;
; The NF stack-adjust opcodes are 64-bit (SUB64ri32_NF/ADD64ri32_NF), so for
; the x32 ABI -- where the stack pointer is the 32-bit ESP -- NF must not be
; used; the sized non-NF SUB32ri/ADD32ri are emitted instead.
; RUN: llc < %t/nounwind.ll -mtriple=x86_64-linux-gnux32 -mattr=+nf -verify-machineinstrs | FileCheck %s --check-prefix=X32

;--- v1.ll
declare void @callee(ptr)

define void @f() {
; Prologue: NF is always safe on Windows regardless of unwind version, because
; the prologue unwinder does not disassemble. The stack is allocated with a
; {nf} subq, with the matching .seh_stackalloc.
;
; V1-LABEL: f:
; V1:         {nf} subq ${{[0-9]+}}, %rsp
; V1:         .seh_stackalloc
; V1:         .seh_endprologue
;
; Epilogue under v1/v2: the OS unwinder disassembles the epilogue, so the stack
; deallocation must be a legacy-encoded addq, NOT {nf} addq.
; V1:         .seh_startepilogue
; V1-NOT:     {nf} addq {{.*}}, %rsp
; V1:         addq ${{[0-9]+}}, %rsp
; V1:         .seh_endepilogue
; V1:         retq
entry:
  %p = alloca [64 x i8], align 16
  call void @callee(ptr %p)
  ret void
}

;--- v2.ll
declare void @callee(ptr)

define void @f() {
; Like v1, the v2 epilogue unwinder disassembles the epilogue (and looks for
; the .seh_unwindv2start marker), so NF is used in the prologue but NOT in the
; epilogue.
; V2-LABEL: f:
; V2:         .seh_unwindversion 2
; V2:         {nf} subq ${{[0-9]+}}, %rsp
; V2:         .seh_stackalloc
; V2:         .seh_endprologue
;
; V2:         .seh_startepilogue
; V2-NOT:     {nf} addq {{.*}}, %rsp
; V2:         addq ${{[0-9]+}}, %rsp
; V2:         .seh_endepilogue
; V2:         retq
entry:
  %p = alloca [64 x i8], align 16
  call void @callee(ptr %p)
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"winx64-eh-unwind", i32 2}

;--- v3.ll
declare void @callee(ptr)

define void @f() {
; In v3 mode the SEH directive is emitted *before* its instruction.
; V3:         .seh_unwindversion 3
; V3-LABEL: f:
; V3:         .seh_stackalloc
; V3:         {nf} subq ${{[0-9]+}}, %rsp
; V3:         .seh_endprologue
;
; Epilogue under v3: epilog ops are encoded declaratively, so {nf} addq is
; allowed for the stack deallocation.
; V3:         .seh_startepilogue
; V3:         .seh_stackalloc
; V3:         {nf} addq ${{[0-9]+}}, %rsp
; V3:         .seh_endepilogue
; V3:         retq
entry:
  %p = alloca [64 x i8], align 16
  call void @callee(ptr %p)
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"winx64-eh-unwind", i32 3}

;--- nounwind.ll
declare void @callee(ptr)

; A nounwind function emits no SEH unwind info, so the OS never disassembles
; its epilogue. NF is therefore allowed in both the prologue and the epilogue
; even under the default (v1) unwind mode, and no .seh_ directives are emitted.
define void @f() nounwind {
; NOUNWIND-LABEL: f:
; NOUNWIND-NOT:    .seh_
; NOUNWIND:        {nf} subq ${{[0-9]+}}, %rsp
; NOUNWIND:        {nf} addq ${{[0-9]+}}, %rsp
; NOUNWIND:        retq
; NOUNWIND-NOT:    .seh_
;
; For the x32 ABI the stack pointer is ESP, so the 64-bit NF opcodes can't be
; used: plain SUB32ri/ADD32ri on %esp are emitted, with no {nf} prefix.
; X32-LABEL: f:
; X32-NOT:        {nf}
; X32:            subl ${{[0-9]+}}, %esp
; X32:            addl ${{[0-9]+}}, %esp
; X32:            retq
; X32-NOT:        {nf}
entry:
  %p = alloca [64 x i8], align 16
  call void @callee(ptr %p)
  ret void
}
