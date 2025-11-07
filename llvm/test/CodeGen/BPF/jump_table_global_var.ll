; Checks generated using command:
;    llvm/utils/update_test_body.py llvm/test/CodeGen/BPF/jump_table_global_var.ll

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llc -march=bpf -mcpu=v4 < test.ll | FileCheck %s
;
; Source code:
;   int foo(unsigned a) {
;     __label__ l1, l2;
;     void *jt1[] = {[0]=&&l1, [1]=&&l2};
;     int ret = 0;
;
;     goto *jt1[a % 2];
;     l1: ret += 1;
;     l2: ret += 3;
;     return ret;
;   }
;
; Compilation Flags:
;   clang --target=bpf -mcpu=v4 -O2 -emit-llvm -S test.c

.ifdef GEN
;--- test.ll
@__const.foo.jt1 = private unnamed_addr constant [2 x ptr] [ptr blockaddress(@foo, %l1), ptr blockaddress(@foo, %l2)], align 8

define dso_local range(i32 3, 5) i32 @foo(i32 noundef %a) local_unnamed_addr {
entry:
  %rem = and i32 %a, 1
  %idxprom = zext nneg i32 %rem to i64
  %arrayidx = getelementptr inbounds nuw [2 x ptr], ptr @__const.foo.jt1, i64 0, i64 %idxprom
  %0 = load ptr, ptr %arrayidx, align 8
  indirectbr ptr %0, [label %l1, label %l2]

l1:                                               ; preds = %entry
  br label %l2

l2:                                               ; preds = %l1, %entry
  %ret.0 = phi i32 [ 4, %l1 ], [ 3, %entry ]
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
; CHECK: 	.type	foo,@function
; CHECK: foo:                                    # @foo
; CHECK: .Lfoo$local:
; CHECK: 	.type	.Lfoo$local,@function
; CHECK: 	.cfi_startproc
; CHECK: # %bb.0:                                # %entry
; CHECK:                                         # kill: def $w1 killed $w1 def $r1
; CHECK: 	w1 &= 1
; CHECK: 	r1 <<= 3
; CHECK: 	r2 = BPF.JT.0.0 ll
; CHECK: 	r2 += r1
; CHECK: 	r1 = *(u64 *)(r2 + 0)
; CHECK: 	gotox r1
; CHECK: .Ltmp0:                                 # Block address taken
; CHECK: LBB0_1:                                 # %l1
; CHECK: 	w0 = 4
; CHECK: 	goto LBB0_3
; CHECK: .Ltmp1:                                 # Block address taken
; CHECK: LBB0_2:                                 # %l2
; CHECK: 	w0 = 3
; CHECK: LBB0_3:                                 # %.split
; CHECK: 	exit
; CHECK: .Lfunc_end0:
; CHECK: 	.size	foo, .Lfunc_end0-foo
; CHECK: 	.size	.Lfoo$local, .Lfunc_end0-foo
; CHECK: 	.cfi_endproc
; CHECK: 	.section	.jumptables,"",@progbits
; CHECK: BPF.JT.0.0:
; CHECK: 	.quad	LBB0_1-.text
; CHECK: 	.quad	LBB0_2-.text
; CHECK: 	.size	BPF.JT.0.0, 16
