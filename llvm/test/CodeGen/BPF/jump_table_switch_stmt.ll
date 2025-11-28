; Checks generated using command:
;    llvm/utils/update_test_body.py llvm/test/CodeGen/BPF/jump_table_switch_stmt.ll

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llc -march=bpf -mcpu=v4 -bpf-min-jump-table-entries=3 < test.ll | FileCheck %s
;
; Source code:
;   int ret_user;
;   int foo(int a)
;   {
;      switch (a) {
;      case 1: ret_user = 18; break;
;      case 20: ret_user = 6; break;
;      case 30: ret_user = 2; break;
;      default: break;
;      }
;      return 0;
;   }
;
; Compilation Flags:
;   clang --target=bpf -mcpu=v4 -O2 -emit-llvm -S test.c

.ifdef GEN
;--- test.ll
@ret_user = dso_local local_unnamed_addr global i32 0, align 4

define dso_local noundef i32 @foo(i32 noundef %a) local_unnamed_addr {
entry:
  switch i32 %a, label %sw.epilog [
    i32 1, label %sw.epilog.sink.split
    i32 20, label %sw.bb1
    i32 30, label %sw.bb2
  ]

sw.bb1:                                           ; preds = %entry
  br label %sw.epilog.sink.split

sw.bb2:                                           ; preds = %entry
  br label %sw.epilog.sink.split

sw.epilog.sink.split:                             ; preds = %entry, %sw.bb1, %sw.bb2
  %.sink = phi i32 [ 2, %sw.bb2 ], [ 6, %sw.bb1 ], [ 18, %entry ]
  store i32 %.sink, ptr @ret_user, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.epilog.sink.split, %entry
  ret i32 0
}

;--- gen
echo ""
echo "; Generated checks follow"
echo ";"
llc -march=bpf -mcpu=v4 -bpf-min-jump-table-entries=3 < test.ll \
  | awk '/# -- End function/ {p=0} /@function/ {p=1} p {print "; CHECK" ": " $0}'

.endif

; Generated checks follow
;
; CHECK: 	.type	foo,@function
; CHECK: foo:                                    # @foo
; CHECK: .Lfoo$local:
; CHECK: 	.type	.Lfoo$local,@function
; CHECK: 	.cfi_startproc
; CHECK: # %bb.0:                                # %entry
; CHECK:                                         # kill: def $w1 killed $w1 def $r1
; CHECK: 	w1 += -1
; CHECK: 	if w1 > 29 goto LBB0_5
; CHECK: # %bb.1:                                # %entry
; CHECK: 	w2 = 18
; CHECK: 	r1 <<= 3
; CHECK: 	r3 = BPF.JT.0.0 ll
; CHECK: 	r4 = BPF.JT.0.0 ll
; CHECK: 	r4 += r1
; CHECK: 	r1 = *(u64 *)(r4 + 0)
; CHECK: 	r3 += r1
; CHECK: 	gotox r3
; CHECK: LBB0_2:                                 # %sw.bb1
; CHECK: 	w2 = 6
; CHECK: 	goto LBB0_4
; CHECK: LBB0_3:                                 # %sw.bb2
; CHECK: 	w2 = 2
; CHECK: LBB0_4:                                 # %sw.epilog.sink.split
; CHECK: 	r1 = ret_user ll
; CHECK: 	*(u32 *)(r1 + 0) = w2
; CHECK: LBB0_5:                                 # %sw.epilog
; CHECK: 	w0 = 0
; CHECK: 	exit
; CHECK: .Lfunc_end0:
; CHECK: 	.size	foo, .Lfunc_end0-foo
; CHECK: 	.size	.Lfoo$local, .Lfunc_end0-foo
; CHECK: 	.cfi_endproc
; CHECK: 	.section	.jumptables,"",@progbits
; CHECK: BPF.JT.0.0:
; CHECK: 	.quad	LBB0_4-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_2-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_5-.text
; CHECK: 	.quad	LBB0_3-.text
; CHECK: 	.size	BPF.JT.0.0, 240
