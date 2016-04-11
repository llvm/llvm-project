; RUN: llc -march=mipsel -disable-mips-delay-filler -relocation-model=pic < %s | \
; RUN:  FileCheck %s -check-prefix=O32
; RUN: llc -march=mipsel -mips-load-target-from-got=false \
; RUN:  -disable-mips-delay-filler -relocation-model=pic < %s | FileCheck %s -check-prefix=O32-LOADTGT

@gd1 = common global double 0.000000e+00, align 8
@gd2 = common global double 0.000000e+00, align 8

; O32-LABEL: caller3:
; O32-DAG:   lw $25, %call16(callee3)
; O32-DAG:   move $gp
; O32:       jalr $25
; O32-NOT:   move $gp
; O32:       lw $25, %call16(callee3)
; O32-NOT:   move $gp
; O32:       jalr $25
; O32-NOT:   move $gp
; O32:       lw $25, %call16(callee3)
; O32-NOT:   move $gp
; O32:       jalr $25

; O32-LOADTGT-LABEL: caller3:
; O32-LOADTGT-DAG:   lw $25, %call16(callee3)
; O32-LOADTGT-DAG:   move $gp
; O32-LOADTGT:       jalr $25
; O32-LOADTGT-NOT:   move $gp
; O32-LOADTGT:       move $25
; O32-LOADTGT-NOT:   move $gp
; O32-LOADTGT:       jalr $25
; O32-LOADTGT-NOT:   move $gp
; O32-LOADTGT:       move $25
; O32-LOADTGT-NOT:   move $gp
; O32-LOADTGT:       jalr $25

define void @caller3(i32 %n) {
entry:
  tail call void @callee3()
  tail call void @callee3()
  %tobool1 = icmp eq i32 %n, 0
  br i1 %tobool1, label %while.end, label %while.body

while.body:
  %n.addr.02 = phi i32 [ %dec, %while.body ], [ %n, %entry ]
  %dec = add nsw i32 %n.addr.02, -1
  tail call void @callee3()
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:
  ret void
}

declare void @callee3()

; O32-LABEL: caller4:
; O32-DAG:   lw $25, %call16(ceil)
; O32-DAG:   move $gp
; O32:       jalr $25
; O32-NOT:   move $gp
; O32:       lw $25, %call16(ceil)
; O32-NOT:   move $gp
; O32:       jalr $25
; O32-NOT:   move $gp
; O32:       lw $25, %call16(ceil)
; O32-NOT:   move $gp
; O32:       jalr $25

; O32-LOADTGT-LABEL: caller4:
; O32-LOADTGT-DAG:   lw $25, %call16(ceil)
; O32-LOADTGT-DAG:   move $gp
; O32-LOADTGT:       jalr $25
; O32-LOADTGT-NOT:   move $gp
; O32-LOADTGT:       move $25
; O32-LOADTGT-NOT:   move $gp
; O32-LOADTGT:       jalr $25
; O32-LOADTGT-NOT:   move $gp
; O32-LOADTGT:       move $25
; O32-LOADTGT-NOT:   move $gp
; O32-LOADTGT:       jalr $25

define void @caller4(double %d) {
entry:
  %call = tail call double @ceil(double %d)
  %call1 = tail call double @ceil(double %call)
  store double %call1, double* @gd2, align 8
  %call2 = tail call double @ceil(double %call1)
  store double %call2, double* @gd1, align 8
  ret void
}

declare double @ceil(double)
