; RUN: llc < %s -march=aarch64 -simplify-mir -stop-after=aarch64-isel -combiner-select-seq-min-cost-benefit=5 | FileCheck %s --check-prefix=PST

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
; Do we identify the setcc:
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

; PST-LABEL: name:            sum
; PST:     %0:gpr64common = COPY $x0
; PST-NEXT:     %1:gpr64 = SUBSXri %0, 0, 0, implicit-def $nzcv
; PST-NEXT:     %2:gpr32 = MOVi32imm 4
; PST-NEXT:     %3:gpr64 = SUBREG_TO_REG 0, killed %2, %subreg.sub_32
; PST-NEXT:     %4:gpr64 = MOVi64imm -10
; PST-NEXT:     %5:gpr64 = CSELXr killed %4, killed %3, 0, implicit $nzcv
; PST-NEXT:     %6:gpr32 = MOVi32imm 56
; PST-NEXT:     %7:gpr64 = SUBREG_TO_REG 0, killed %6, %subreg.sub_32
; PST-NEXT:     %8:gpr32 = MOVi32imm 91
; PST-NEXT:     %9:gpr64 = SUBREG_TO_REG 0, killed %8, %subreg.sub_32
; PST-NEXT:     %10:gpr64 = CSELXr killed %9, killed %7, 0, implicit $nzcv
; PST-NEXT:     %11:gpr64 = COPY $xzr
; PST-NEXT:     %12:gpr64 = CSELXr %11, %0, 0, implicit $nzcv
; PST-NEXT:     %13:gpr64 = nuw ADDXrr killed %12, killed %10
; PST-NEXT:     ADJCALLSTACKDOWN 0, 0, implicit-def dead $sp, implicit $sp
; PST-NEXT:     $x0 = COPY %13
; PST-NEXT:     BL @access, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit $x0, implicit-def $sp, implicit-def $w0
; PST-NEXT:     ADJCALLSTACKUP 0, 0, implicit-def dead $sp, implicit $sp
; PST-NEXT:     %14:gpr32 = COPY $w0
; PST-NEXT:     %15:gpr64 = nuw ADDXrr %13, %5
; PST-NEXT:     ADJCALLSTACKDOWN 0, 0, implicit-def dead $sp, implicit $sp
; PST-NEXT:     $x0 = COPY %15
; PST-NEXT:     BL @access, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit $x0, implicit-def $sp, implicit-def $w0
; PST-NEXT:     ADJCALLSTACKUP 0, 0, implicit-def dead $sp, implicit $sp
; PST-NEXT:     %16:gpr32 = COPY $w0
; PST-NEXT:     %17:gpr32 = nsw ADDWrr %16, %14
; PST-NEXT:     %18:gpr64 = nuw ADDXrr %15, %5
; PST-NEXT:     ADJCALLSTACKDOWN 0, 0, implicit-def dead $sp, implicit $sp
; PST-NEXT:     $x0 = COPY %18
; PST-NEXT:     BL @access, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit $x0, implicit-def $sp, implicit-def $w0
; PST-NEXT:     ADJCALLSTACKUP 0, 0, implicit-def dead $sp, implicit $sp
; PST-NEXT:     %19:gpr32 = COPY $w0
; PST-NEXT:     %20:gpr32 = nsw ADDWrr killed %17, %19
; PST-NEXT:     %21:gpr64 = nuw ADDXrr %18, %5
; PST-NEXT:     ADJCALLSTACKDOWN 0, 0, implicit-def dead $sp, implicit $sp
; PST-NEXT:     $x0 = COPY %21
; PST-NEXT:     BL @access, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit $x0, implicit-def $sp, implicit-def $w0
; PST-NEXT:     ADJCALLSTACKUP 0, 0, implicit-def dead $sp, implicit $sp
; PST-NEXT:     %22:gpr32 = COPY $w0
; PST-NEXT:     %23:gpr32 = nsw ADDWrr killed %20, %22
; PST-NEXT:     %24:gpr64 = nuw ADDXrr %21, %5
; PST-NEXT:     ADJCALLSTACKDOWN 0, 0, implicit-def dead $sp, implicit $sp
; PST-NEXT:     $x0 = COPY %24
; PST-NEXT:     BL @access, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit $x0, implicit-def $sp, implicit-def $w0
; PST-NEXT:     ADJCALLSTACKUP 0, 0, implicit-def dead $sp, implicit $sp
; PST-NEXT:     %25:gpr32 = COPY $w0
; PST-NEXT:     %26:gpr32 = nsw ADDWrr killed %23, %25
; PST-NEXT:     %27:gpr64 = nuw ADDXrr %24, %5
; PST-NEXT:     ADJCALLSTACKDOWN 0, 0, implicit-def dead $sp, implicit $sp
; PST-NEXT:     $x0 = COPY %27
; PST-NEXT:     BL @access, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit $x0, implicit-def $sp, implicit-def $w0
; PST-NEXT:     ADJCALLSTACKUP 0, 0, implicit-def dead $sp, implicit $sp
; PST-NEXT:     %28:gpr32 = COPY $w0
; PST-NEXT:     %29:gpr32 = nsw ADDWrr killed %26, %28
; PST-NEXT:     %30:gpr64 = nuw ADDXrr %27, %5
; PST-NEXT:     ADJCALLSTACKDOWN 0, 0, implicit-def dead $sp, implicit $sp
; PST-NEXT:     $x0 = COPY %30
; PST-NEXT:     BL @access, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit $x0, implicit-def $sp, implicit-def $w0
; PST-NEXT:     ADJCALLSTACKUP 0, 0, implicit-def dead $sp, implicit $sp
; PST-NEXT:     %31:gpr32 = COPY $w0
; PST-NEXT:     %32:gpr32 = nsw ADDWrr killed %29, %31
; PST-NEXT:     %33:gpr64 = nuw ADDXrr %30, %5
; PST-NEXT:     ADJCALLSTACKDOWN 0, 0, implicit-def dead $sp, implicit $sp
; PST-NEXT:     $x0 = COPY %33
; PST-NEXT:     BL @access, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit $x0, implicit-def $sp, implicit-def $w0
; PST-NEXT:     ADJCALLSTACKUP 0, 0, implicit-def dead $sp, implicit $sp
; PST-NEXT:     %34:gpr32 = COPY $w0
; PST-NEXT:     %35:gpr32 = nsw ADDWrr killed %32, %34
; PST-NEXT:     $w0 = COPY %35
; PST-NEXT:     RET_ReallyLR implicit $w0
; PST-NEXT: ...
