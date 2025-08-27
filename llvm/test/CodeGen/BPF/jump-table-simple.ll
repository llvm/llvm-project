; Checks generated using command:
;
;    llvm/utils/update_test_body.py llvm/test/CodeGen/BPF/jump-table-simple.ll
;
; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llc -O2 -bpf-min-jump-table-entries=1 -mtriple=bpfel -mcpu=v4 < test.ll | FileCheck %s
;
; Check general program structure generated for a jump table

.ifdef GEN
;--- test.ll
define i64 @foo(i64 %v) {
entry:
  switch i64 %v, label %sw.default [
    i64 0, label %sw.epilog
    i64 1, label %sw.bb1
    i64 2, label %sw.bb1
    i64 3, label %sw.bb2
  ]

sw.bb1:
  br label %sw.epilog

sw.bb2:
  br label %sw.epilog

sw.default:
  br label %sw.epilog

sw.epilog:
  %ret = phi i64 [ 42, %sw.default ], [ 3, %sw.bb1 ], [ 5, %sw.bb2 ], [ 7, %entry ]
  ret i64 %ret
}

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang some version"}

;--- gen
echo ""
echo "; Generated checks follow"
echo ";"
llc -O2 -bpf-min-jump-table-entries=1 -mtriple=bpfel -mcpu=v4 < test.ll \
  | awk '/# -- End function/ {p=0} /@function/ {p=1} p {print "; CHECK" ": " $0}'
.endif

; Generated checks follow
;
; CHECK: 	.type	foo,@function
; CHECK: foo:                                    # @foo
; CHECK: 	.cfi_startproc
; CHECK: # %bb.0:                                # %entry
; CHECK: 	if r1 > 3 goto LBB0_5
; CHECK: # %bb.1:                                # %entry
; CHECK: .LBPF.JX.0.0:
; CHECK: 	.reloc 0, FK_SecRel_8, BPF.JT.0.0
; CHECK: 	gotox r1
; CHECK: LBB0_3:                                 # %sw.bb1
; CHECK: 	r0 = 3
; CHECK: 	goto LBB0_6
; CHECK: LBB0_2:
; CHECK: 	r0 = 7
; CHECK: 	goto LBB0_6
; CHECK: LBB0_4:                                 # %sw.bb2
; CHECK: 	r0 = 5
; CHECK: 	goto LBB0_6
; CHECK: LBB0_5:                                 # %sw.default
; CHECK: 	r0 = 42
; CHECK: LBB0_6:                                 # %sw.epilog
; CHECK: 	exit
; CHECK: .Lfunc_end0:
; CHECK: 	.size	foo, .Lfunc_end0-foo
; CHECK: 	.cfi_endproc
; CHECK: 	.section	.jumptables,"",@progbits
; CHECK: .L0_0_set_2 = ((LBB0_2-.LBPF.JX.0.0)>>3)-1
; CHECK: .L0_0_set_3 = ((LBB0_3-.LBPF.JX.0.0)>>3)-1
; CHECK: .L0_0_set_4 = ((LBB0_4-.LBPF.JX.0.0)>>3)-1
; CHECK: BPF.JT.0.0:
; CHECK: 	.long	.L0_0_set_2
; CHECK: 	.long	.L0_0_set_3
; CHECK: 	.long	.L0_0_set_3
; CHECK: 	.long	.L0_0_set_4
; CHECK: 	.size	BPF.JT.0.0, 16
