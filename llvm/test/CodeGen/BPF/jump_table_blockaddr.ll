; Checks generated using command:
;    llvm/utils/update_test_body.py llvm/test/CodeGen/BPF/jump_table_blockaddr.ll

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llc -march=bpf -mcpu=v4 < test.ll | FileCheck %s
;
; Source code:
;    int bar(int a) {
;       __label__ l1, l2;
;       void * volatile tgt;
;       int ret = 0;
;       if (a)
;         tgt = &&l1; // synthetic jump table generated here
;       else
;         tgt = &&l2; // another synthetic jump table
;       goto *tgt;
;   l1: ret += 1;
;   l2: ret += 2;
;       return ret;
;     }
;
; Compilation Flags:
;   clang --target=bpf -mcpu=v4 -O2 -emit-llvm -S test.c

.ifdef GEN
;--- test.ll
define dso_local range(i32 2, 4) i32 @bar(i32 noundef %a) local_unnamed_addr{
entry:
  %tgt = alloca ptr, align 8
  %tobool.not = icmp eq i32 %a, 0
  %. = select i1 %tobool.not, ptr blockaddress(@bar, %l2), ptr blockaddress(@bar, %l1)
  store volatile ptr %., ptr %tgt, align 8
  %tgt.0.tgt.0.tgt.0.tgt.0. = load volatile ptr, ptr %tgt, align 8
  indirectbr ptr %tgt.0.tgt.0.tgt.0.tgt.0., [label %l1, label %l2]

l1:                                               ; preds = %entry
  br label %l2

l2:                                               ; preds = %l1, %entry
  %ret.0 = phi i32 [ 3, %l1 ], [ 2, %entry ]
  ret i32 %ret.0
}

;--- gen
echo ""
echo "; Generated checks follow"
echo ";"
llc -march=bpf -mcpu=v4 < test.ll \
  | awk '/# -- End function/ {p=0} /@function/ {p=1} p {print "; CHECK" ": " $0}'

.endif

; Generated checks follow
;
; CHECK: 	.type	bar,@function
; CHECK: bar:                                    # @bar
; CHECK: .Lbar$local:
; CHECK: 	.type	.Lbar$local,@function
; CHECK: 	.cfi_startproc
; CHECK: # %bb.0:                                # %entry
; CHECK: 	r2 = BPF.JT.0.0 ll
; CHECK: 	r2 = *(u64 *)(r2 + 0)
; CHECK: 	r3 = BPF.JT.0.1 ll
; CHECK: 	r3 = *(u64 *)(r3 + 0)
; CHECK: 	if w1 == 0 goto LBB0_2
; CHECK: # %bb.1:                                # %entry
; CHECK: 	r3 = r2
; CHECK: LBB0_2:                                 # %entry
; CHECK: 	*(u64 *)(r10 - 8) = r3
; CHECK: 	r1 = *(u64 *)(r10 - 8)
; CHECK: 	gotox r1
; CHECK: .Ltmp0:                                 # Block address taken
; CHECK: LBB0_3:                                 # %l1
; CHECK: 	w0 = 3
; CHECK: 	goto LBB0_5
; CHECK: .Ltmp1:                                 # Block address taken
; CHECK: LBB0_4:                                 # %l2
; CHECK: 	w0 = 2
; CHECK: LBB0_5:                                 # %.split
; CHECK: 	exit
; CHECK: .Lfunc_end0:
; CHECK: 	.size	bar, .Lfunc_end0-bar
; CHECK: 	.size	.Lbar$local, .Lfunc_end0-bar
; CHECK: 	.cfi_endproc
; CHECK: 	.section	.jumptables,"",@progbits
; CHECK: BPF.JT.0.0:
; CHECK: 	.quad	LBB0_3-.text
; CHECK: 	.size	BPF.JT.0.0, 8
; CHECK: BPF.JT.0.1:
; CHECK: 	.quad	LBB0_4-.text
; CHECK: 	.size	BPF.JT.0.1, 8
