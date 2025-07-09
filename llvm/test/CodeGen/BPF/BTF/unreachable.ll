; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK-BTF %s
; RUN: llc -mtriple=bpfel -mcpu=v3 < %s | FileCheck -check-prefixes=CHECK %s

define void @foo() {
entry:
  tail call void @bar()
  unreachable
}

; CHECK:      foo:
; CHECK-NEXT: .Lfunc_begin0:
; CHECK-NEXT:   .cfi_startproc
; CHECK-NEXT: # %bb.0:
; CHECK-NEXT:    call bar
; CHECK-NEXT:    call __bpf_trap
; CHECK-NEXT:    exit
; CHECK-NEXT: .Lfunc_end0:

define void @buz() #0 {
entry:
  tail call void asm sideeffect "r0 = r1; exit;", ""()
  unreachable
}

; CHECK: buz:
; CHECK-NEXT: .Lfunc_begin1:
; CHECK-NEXT:   .cfi_startproc
; CHECK-NEXT: # %bb.0:
; CHECK-NEXT:   #APP
; CHECK-NEXT:   r0 = r1
; CHECK-NEXT:   exit
; CHECK-EMPTY:
; CHECK-NEXT:   #NO_APP
; CHECK-NEXT: .Lfunc_end1:

; CHECK-BTF: [1] FUNC_PROTO '(anon)' ret_type_id=0 vlen=0
; CHECK-BTF: [2] FUNC '__bpf_trap' type_id=1 linkage=extern
; CHECK-BTF: [3] DATASEC '.ksyms' size=0 vlen=1
; CHECK-BTF:  type_id=2 offset=0 size=0

declare dso_local void @bar()

attributes #0 = { naked }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/some/dir")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
