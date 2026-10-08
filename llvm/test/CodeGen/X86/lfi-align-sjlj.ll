; RUN: split-file %s %t
; RUN: llc -mtriple=x86_64_lfi -exception-model=sjlj < %t/sjlj.ll | FileCheck -check-prefix=SJLJ %s
; RUN: llc -mtriple=x86_64_lfi < %t/dwarf.ll | FileCheck -check-prefix=DWARF %s

; The SJLJ dispatch block jumps indirectly to the landing pad through a jump
; table, so the landing pad is bundle aligned as a jump table target. Other
; exception models have no dispatch block, and the landing pad is only
; aligned because it is an EH pad.

;--- sjlj.ll
; FIXME: SJLJ lowering still requires -exception-model=sjlj

declare void @may_throw()
declare i32 @__gxx_personality_sj0(...)

define void @invoke_aligns_landing_pad() personality ptr @__gxx_personality_sj0 {
; SJLJ-LABEL: invoke_aligns_landing_pad:
; SJLJ:         callq may_throw@PLT
; SJLJ:         .p2align 5
; SJLJ-NEXT:  .LBB0_3:
; SJLJ:         jmpq *(%rcx,%rax,8)
; SJLJ-NEXT:    .p2align 5
; SJLJ-NEXT:  .LBB0_2:
; SJLJ-NEXT:  .Ltmp2:
; SJLJ:       .LJTI0_0:
; SJLJ-NEXT:    .quad .LBB0_2
entry:
  invoke void @may_throw() to label %cont unwind label %lpad

cont:
  ret void

lpad:
  %l = landingpad { ptr, i32 } catch ptr null
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"sjlj"}

;--- dwarf.ll

declare void @may_throw()
declare i32 @__gxx_personality_v0(...)

define void @invoke_aligns_landing_pad() personality ptr @__gxx_personality_v0 {
; DWARF-LABEL: invoke_aligns_landing_pad:
; DWARF:         callq may_throw@PLT
; DWARF:         .p2align 5
; DWARF-NEXT:  .LBB0_2:
; DWARF:       .Ltmp2:
; DWARF-NOT:     jmpq *
; DWARF-NOT:   .LJTI
entry:
  invoke void @may_throw() to label %cont unwind label %lpad

cont:
  ret void

lpad:
  %l = landingpad { ptr, i32 } catch ptr null
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"dwarf"}
