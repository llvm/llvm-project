; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK-BTF %s
; RUN: llc -mtriple=bpfel -mcpu=v3 < %s | FileCheck -check-prefixes=CHECK %s

; BPFTargetMachine Options.NoTrapAfterNoreturn has been set to true,
; so in below code, 'unreachable' will become a noop and
; llvm.trap() will become 'call __bpf_trap' after selectiondag.
define dso_local void @foo(i32 noundef %0) {
  tail call void @llvm.trap()
  unreachable
}

; CHECK:      .Lfunc_begin0:
; CHECK-NEXT:   .cfi_startproc
; CHECK-NEXT: # %bb.0:
; CHECK-NEXT:    call __bpf_trap
; CHECK-NEXT:    exit
; CHECK-NEXT: .Lfunc_end0:

; CHECK-BTF: [1] FUNC_PROTO '(anon)' ret_type_id=0 vlen=0
; CHECK-BTF: [2] FUNC '__bpf_trap' type_id=1 linkage=extern
; CHECK-BTF: [3] DATASEC '.ksyms' size=0 vlen=1
; CHECK-BTF:  type_id=2 offset=0 size=0

declare void @llvm.trap() #1

attributes #1 = {noreturn}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "test_trap.c", directory: "/some/dir")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
