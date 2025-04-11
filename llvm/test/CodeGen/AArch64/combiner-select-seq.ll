; RUN: llc < %s -march=aarch64 -simplify-mir -stop-after=amdgpu-isel | FileCheck %s --check-prefix=PRE
; RUN: llc < %s -march=aarch64 -simplify-mir -stop-after=amdgpu-isel --combiner-select-seq -combiner-select-seq-min-cost-benefit=5 | FileCheck %s --check-prefix=PST

; Description:
; ------------
; Given:
;   struct A {
;     char junk[53];
;     int a0;
;     int a1;
;     int a2;
;     int a3;
;     int a4;
;     int a5;
;     int a6;
;     int a7;
;   };
;   extern int sum(struct A *pa);
;   extern int access(void *p);
;   int sum(struct A *pa) {
;     int s = 0;
;     s += access(pa ? (void *) &pa->a0 : (void *) 91);
;     s += access(pa ? (void *) &pa->a1 : (void *) 81);
;     s += access(pa ? (void *) &pa->a2 : (void *) 71);
;     s += access(pa ? (void *) &pa->a3 : (void *) 61);
;     s += access(pa ? (void *) &pa->a4 : (void *) 51);
;     s += access(pa ? (void *) &pa->a5 : (void *) 41);
;     s += access(pa ? (void *) &pa->a6 : (void *) 31);
;     s += access(pa ? (void *) &pa->a7 : (void *) 21);
;     return s;
;   }
; Compiled into LLVM IR, thus: -O3 --target=aarch64 -emit-llvm -S
; Do we, under --combiner-select-seq, identify the setcc:
;   Setcc: t5: i1 = setcc t2, Constant:i64<0>, seteq:ch
; along with its select users:
;   User:
;   0 t9: i64 = select t5, Constant:i64<91>, t7
;     t5: i1 = setcc t2, Constant:i64<0>, seteq:ch
;     t8: i64 = Constant<91>
;     t7: i64 = add nuw t2, Constant:i64<56>
;   1 t26: i64 = select t5, Constant:i64<81>, t24
;     t5: i1 = setcc t2, Constant:i64<0>, seteq:ch
;     t25: i64 = Constant<81>
;     t24: i64 = add nuw t2, Constant:i64<60>
;   2 t37: i64 = select t5, Constant:i64<71>, t35
;     t5: i1 = setcc t2, Constant:i64<0>, seteq:ch
;     t36: i64 = Constant<71>
;     t35: i64 = add nuw t2, Constant:i64<64>
;   3 t48: i64 = select t5, Constant:i64<61>, t46
;     t5: i1 = setcc t2, Constant:i64<0>, seteq:ch
;     t47: i64 = Constant<61>
;     t46: i64 = add nuw t2, Constant:i64<68>
;   4 t59: i64 = select t5, Constant:i64<51>, t57
;     t5: i1 = setcc t2, Constant:i64<0>, seteq:ch
;     t58: i64 = Constant<51>
;     t57: i64 = add nuw t2, Constant:i64<72>
;   5 t70: i64 = select t5, Constant:i64<41>, t68
;     t5: i1 = setcc t2, Constant:i64<0>, seteq:ch
;     t69: i64 = Constant<41>
;     t68: i64 = add nuw t2, Constant:i64<76>
;   6 t81: i64 = select t5, Constant:i64<31>, t79
;     t5: i1 = setcc t2, Constant:i64<0>, seteq:ch
;     t80: i64 = Constant<31>
;     t79: i64 = add nuw t2, Constant:i64<80>
;   7 t92: i64 = select t5, Constant:i64<21>, t90
;     t5: i1 = setcc t2, Constant:i64<0>, seteq:ch
;     t91: i64 = Constant<21>
;     t90: i64 = add nuw t2, Constant:i64<84>
; defining the two sequences:
;   t-seq: ( 0 + 91), ( 0 + 81), ..., (0  + 31), (0  + 21)
;   f-seq: (t2 + 56), (t2 + 60), ..., (t2 + 80), (t2 + 84)
; and derive and inject the sequence selectors:
;   BaseReg: t104: i64 = select t5, Constant:i64<0>, t2
;   InitialVal: t105: i64 = select t5, Constant:i64<91>, Constant:i64<56>
;   Delta: t106: i64 = select t5, Constant:i64<-10>, Constant:i64<4>
; that define:
;   AccessAddr:
;   0 t107: i64 = add nuw t104, t105
;   1 t108: i64 = add nuw t107, t106
;   2 t109: i64 = add nuw t108, t106
;   3 t110: i64 = add nuw t109, t106
;   4 t111: i64 = add nuw t110, t106
;   5 t112: i64 = add nuw t111, t106
;   6 t113: i64 = add nuw t112, t106
;   7 t114: i64 = add nuw t113, t106
; and, do we rewrite the DAG, eliminating the selects and the base-displacement
; add's so that instead, the AccessAddr's are now used?

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64"

; Function Attrs: nounwind uwtable
define dso_local i32 @sum(ptr noundef %pa) local_unnamed_addr #0 {
entry:
  %tobool.not = icmp eq ptr %pa, null
  %a0 = getelementptr inbounds nuw i8, ptr %pa, i64 56
  %cond = select i1 %tobool.not, ptr inttoptr (i64 91 to ptr), ptr %a0
  %call = tail call i32 @access(ptr noundef nonnull %cond) #2
  %a1 = getelementptr inbounds nuw i8, ptr %pa, i64 60
  %cond5 = select i1 %tobool.not, ptr inttoptr (i64 81 to ptr), ptr %a1
  %call6 = tail call i32 @access(ptr noundef nonnull %cond5) #2
  %add7 = add nsw i32 %call6, %call
  %a2 = getelementptr inbounds nuw i8, ptr %pa, i64 64
  %cond12 = select i1 %tobool.not, ptr inttoptr (i64 71 to ptr), ptr %a2
  %call13 = tail call i32 @access(ptr noundef nonnull %cond12) #2
  %add14 = add nsw i32 %add7, %call13
  %a3 = getelementptr inbounds nuw i8, ptr %pa, i64 68
  %cond19 = select i1 %tobool.not, ptr inttoptr (i64 61 to ptr), ptr %a3
  %call20 = tail call i32 @access(ptr noundef nonnull %cond19) #2
  %add21 = add nsw i32 %add14, %call20
  %a4 = getelementptr inbounds nuw i8, ptr %pa, i64 72
  %cond26 = select i1 %tobool.not, ptr inttoptr (i64 51 to ptr), ptr %a4
  %call27 = tail call i32 @access(ptr noundef nonnull %cond26) #2
  %add28 = add nsw i32 %add21, %call27
  %a5 = getelementptr inbounds nuw i8, ptr %pa, i64 76
  %cond33 = select i1 %tobool.not, ptr inttoptr (i64 41 to ptr), ptr %a5
  %call34 = tail call i32 @access(ptr noundef nonnull %cond33) #2
  %add35 = add nsw i32 %add28, %call34
  %a6 = getelementptr inbounds nuw i8, ptr %pa, i64 80
  %cond40 = select i1 %tobool.not, ptr inttoptr (i64 31 to ptr), ptr %a6
  %call41 = tail call i32 @access(ptr noundef nonnull %cond40) #2
  %add42 = add nsw i32 %add35, %call41
  %a7 = getelementptr inbounds nuw i8, ptr %pa, i64 84
  %cond47 = select i1 %tobool.not, ptr inttoptr (i64 21 to ptr), ptr %a7
  %call48 = tail call i32 @access(ptr noundef nonnull %cond47) #2
  %add49 = add nsw i32 %add42, %call48
  ret i32 %add49
}

declare dso_local i32 @access(ptr noundef) local_unnamed_addr #1

; PRE-LABEL:   name: sum
; PRE:  early-clobber $sp = frame-setup STRXpre killed $lr, $sp, -80 :: (store (s64) into %stack.8)
; PRE:  frame-setup STPXi killed $x26, killed $x25, $sp, 2 :: (store (s64) into %stack.7), (store (s64) into %stack.6)
; PRE:  frame-setup STPXi killed $x24, killed $x23, $sp, 4 :: (store (s64) into %stack.5), (store (s64) into %stack.4)
; PRE:  frame-setup STPXi killed $x22, killed $x21, $sp, 6 :: (store (s64) into %stack.3), (store (s64) into %stack.2)
; PRE:  frame-setup STPXi killed $x20, killed $x19, $sp, 8 :: (store (s64) into %stack.1), (store (s64) into %stack.0)
; PRE:  frame-setup CFI_INSTRUCTION def_cfa_offset 80
; PRE:  frame-setup CFI_INSTRUCTION offset $w19, -8
; PRE:  frame-setup CFI_INSTRUCTION offset $w20, -16
; PRE:  frame-setup CFI_INSTRUCTION offset $w21, -24
; PRE:  frame-setup CFI_INSTRUCTION offset $w22, -32
; PRE:  frame-setup CFI_INSTRUCTION offset $w23, -40
; PRE:  frame-setup CFI_INSTRUCTION offset $w24, -48
; PRE:  frame-setup CFI_INSTRUCTION offset $w25, -56
; PRE:  frame-setup CFI_INSTRUCTION offset $w26, -64
; PRE:  frame-setup CFI_INSTRUCTION offset $w30, -80
; PRE:  renamable $w8 = MOVZWi 91, 0, implicit-def $x8
; PRE:  renamable $x9 = nuw ADDXri renamable $x0, 56, 0
; PRE:  dead $xzr = SUBSXri renamable $x0, 0, 0, implicit-def $nzcv
; PRE:  renamable $x8 = CSELXr killed renamable $x8, killed renamable $x9, 0, implicit $nzcv
; PRE:  renamable $x9 = nuw ADDXri renamable $x0, 60, 0
; PRE:  renamable $w10 = MOVZWi 81, 0, implicit-def $x10
; PRE:  renamable $x11 = nuw ADDXri renamable $x0, 64, 0
; PRE:  renamable $w12 = MOVZWi 71, 0, implicit-def $x12
; PRE:  renamable $x19 = CSELXr killed renamable $x10, killed renamable $x9, 0, implicit $nzcv
; PRE:  renamable $x20 = CSELXr killed renamable $x12, killed renamable $x11, 0, implicit $nzcv
; PRE:  renamable $x9 = nuw ADDXri renamable $x0, 68, 0
; PRE:  renamable $w10 = MOVZWi 61, 0, implicit-def $x10
; PRE:  renamable $x11 = nuw ADDXri renamable $x0, 72, 0
; PRE:  renamable $w12 = MOVZWi 51, 0, implicit-def $x12
; PRE:  renamable $x21 = CSELXr killed renamable $x10, killed renamable $x9, 0, implicit $nzcv
; PRE:  renamable $x22 = CSELXr killed renamable $x12, killed renamable $x11, 0, implicit $nzcv
; PRE:  renamable $x9 = nuw ADDXri renamable $x0, 76, 0
; PRE:  renamable $w10 = MOVZWi 41, 0, implicit-def $x10
; PRE:  renamable $x11 = nuw ADDXri renamable $x0, 80, 0
; PRE:  renamable $w12 = MOVZWi 31, 0, implicit-def $x12
; PRE:  renamable $x13 = nuw ADDXri killed renamable $x0, 84, 0
; PRE:  renamable $w14 = MOVZWi 21, 0, implicit-def $x14
; PRE:  $x0 = ORRXrs $xzr, killed $x8, 0
; PRE:  renamable $x23 = CSELXr killed renamable $x10, killed renamable $x9, 0, implicit $nzcv
; PRE:  renamable $x24 = CSELXr killed renamable $x12, killed renamable $x11, 0, implicit $nzcv
; PRE:  renamable $x25 = CSELXr killed renamable $x14, killed renamable $x13, 0, implicit killed $nzcv

; PST-LABEL:   name: sum
; PST:  early-clobber $sp = frame-setup STPXpre killed $lr, killed $x21, $sp, -4 :: (store (s64) into %stack.3), (store (s64) into %stack.2)
; PST:  frame-setup STPXi killed $x20, killed $x19, $sp, 2 :: (store (s64) into %stack.1), (store (s64) into %stack.0)
; PST:  frame-setup CFI_INSTRUCTION def_cfa_offset 32
; PST:  frame-setup CFI_INSTRUCTION offset $w19, -8
; PST:  frame-setup CFI_INSTRUCTION offset $w20, -16
; PST:  frame-setup CFI_INSTRUCTION offset $w21, -24
; PST:  frame-setup CFI_INSTRUCTION offset $w30, -32
; PST:  dead $xzr = SUBSXri renamable $x0, 0, 0, implicit-def $nzcv
; PST:  renamable $w9 = MOVZWi 56, 0, implicit-def $x9
; PST:  renamable $w10 = MOVZWi 91, 0, implicit-def $x10
; PST:  renamable $x9 = CSELXr killed renamable $x10, killed renamable $x9, 0, implicit $nzcv
; PST:  renamable $x10 = CSELXr $xzr, killed renamable $x0, 0, implicit $nzcv
; PST:  renamable $w8 = MOVZWi 4, 0, implicit-def $x8
; PST:  $x19 = ADDXrs killed renamable $x10, killed renamable $x9, 0
; PST:  renamable $x11 = MOVNXi 9, 0
; PST:  $x0 = ORRXrs $xzr, $x19, 0
; PST:  renamable $x21 = CSELXr killed renamable $x11, killed renamable $x8, 0, implicit killed $nzcv
; PST:  BL @access, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit killed $x0, implicit-def $sp, implicit-def $w0
; PST:  $x19 = ADDXrs killed renamable $x19, renamable $x21, 0
; PST:  $w20 = ORRWrs $wzr, killed $w0, 0
; PST:  $x0 = ORRXrs $xzr, $x19, 0
; PST:  BL @access, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit killed $x0, implicit-def $sp, implicit-def $w0
; PST:  $x19 = ADDXrs killed renamable $x19, renamable $x21, 0
; PST:  $w20 = ADDWrs killed renamable $w0, killed renamable $w20, 0
; PST:  $x0 = ORRXrs $xzr, $x19, 0
; PST:  BL @access, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit killed $x0, implicit-def $sp, implicit-def $w0
; PST:  $x19 = ADDXrs killed renamable $x19, renamable $x21, 0
; PST:  $w20 = ADDWrs killed renamable $w20, killed renamable $w0, 0
; PST:  $x0 = ORRXrs $xzr, $x19, 0
; PST:  BL @access, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit killed $x0, implicit-def $sp, implicit-def $w0
; PST:  $x19 = ADDXrs killed renamable $x19, renamable $x21, 0
; PST:  $w20 = ADDWrs killed renamable $w20, killed renamable $w0, 0
; PST:  $x0 = ORRXrs $xzr, $x19, 0
; PST:  BL @access, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit killed $x0, implicit-def $sp, implicit-def $w0
; PST:  $x19 = ADDXrs killed renamable $x19, renamable $x21, 0
; PST:  $w20 = ADDWrs killed renamable $w20, killed renamable $w0, 0
; PST:  $x0 = ORRXrs $xzr, $x19, 0
; PST:  BL @access, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit killed $x0, implicit-def $sp, implicit-def $w0
; PST:  $x19 = ADDXrs killed renamable $x19, renamable $x21, 0
; PST:  $w20 = ADDWrs killed renamable $w20, killed renamable $w0, 0
; PST:  $x0 = ORRXrs $xzr, $x19, 0
; PST:  BL @access, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit killed $x0, implicit-def $sp, implicit-def $w0
; PST:  $w20 = ADDWrs killed renamable $w20, killed renamable $w0, 0
; PST:  $x0 = ADDXrs killed renamable $x19, killed renamable $x21, 0
; PST:  BL @access, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit killed $x0, implicit-def $sp, implicit-def $w0
; PST:  $w0 = ADDWrs killed renamable $w20, killed renamable $w0, 0
; PST:  $x20, $x19 = frame-destroy LDPXi $sp, 2 :: (load (s64) from %stack.1), (load (s64) from %stack.0)
; PST:  early-clobber $sp, $lr, $x21 = frame-destroy LDPXpost $sp, 4 :: (load (s64) from %stack.3), (load (s64) from %stack.2)
; PST:  RET undef $lr, implicit killed $w0
