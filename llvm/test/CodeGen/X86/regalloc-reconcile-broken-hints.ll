; RUN: llc < %s -o - -mtriple=x86_64-apple-macosx | FileCheck %s
; Test case for the recoloring of broken hints.
; This is tricky to have something reasonably small to kick this optimization since
; it requires that spliting and spilling occur.
; The bottom line is that this test case is fragile.
; This was reduced from the make_list function from the llvm-testsuite:
; SingleSource/Benchmarks/McGill/chomp.c

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct._list = type { i32*, %struct._list* }

@ncol = external global i32, align 4
@nrow = external global i32, align 4

declare noalias i32* @copy_data()

declare noalias i8* @malloc(i64)

declare i32 @get_value()

declare i32 @in_wanted(i32* nocapture readonly)

declare noalias i32* @make_data()

; CHECK-LABEL: make_list:
; Function prologue.
; CHECK: pushq
; CHECK: subq ${{[0-9]+}}, %rsp
; Move the first argument (%data) into a temporary register.
; It will not survive the call to malloc otherwise.
; CHECK: movq %rdi, [[ARG1:%r[0-9a-z]+]]
; CHECK: callq _malloc
; Compute %data - 1 as used for load in land.rhs.i (via the variable  %indvars.iv.next.i).
; CHECK: addq $-4, [[ARG1]]
; We use to produce a useless copy here and move %data in another temporary register. 
; CHECK-NOT: movq [[ARG1]]
; End of the first basic block.
; CHECK: .p2align
; Now check that %data is used in an address computation.
; CHECK: leaq ([[ARG1]]
define %struct._list* @make_list(i32* nocapture readonly %data, i32* nocapture %value, i32* nocapture %all) {
entry:
  %call = tail call i8* @malloc(i64 16)
  %next = getelementptr inbounds i8, i8* %call, i64 8
  %tmp = bitcast i8* %next to %struct._list**
  %tmp2 = bitcast i8* %call to %struct._list*
  %.pre78 = load i32, i32* @ncol, align 4
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc32, %entry
  %tmp4 = phi i32 [ %.pre78, %entry ], [ 0, %for.inc32 ]
  %current.077 = phi %struct._list* [ %tmp2, %entry ], [ %current.1.lcssa, %for.inc32 ]
  %cmp270 = icmp eq i32 %tmp4, 0
  br i1 %cmp270, label %for.inc32, label %for.body3

for.body3:                                        ; preds = %if.end31, %for.cond1.preheader
  %current.173 = phi %struct._list* [ %current.2, %if.end31 ], [ %current.077, %for.cond1.preheader ]
  %row.172 = phi i32 [ %row.3, %if.end31 ], [ 0, %for.cond1.preheader ]
  %col.071 = phi i32 [ %inc, %if.end31 ], [ 0, %for.cond1.preheader ]
  %call4 = tail call i32* @make_data()
  %tmp5 = load i32, i32* @ncol, align 4
  %tobool14.i = icmp eq i32 %tmp5, 0
  br i1 %tobool14.i, label %while.cond.i, label %while.body.lr.ph.i

while.body.lr.ph.i:                               ; preds = %for.body3
  %tmp6 = sext i32 %tmp5 to i64
  br label %while.body.i

while.body.i:                                     ; preds = %while.body.i, %while.body.lr.ph.i
  %indvars.iv.i = phi i64 [ %tmp6, %while.body.lr.ph.i ], [ %indvars.iv.next.i, %while.body.i ]
  %indvars.iv.next.i = add nsw i64 %indvars.iv.i, -1
  %tmp9 = trunc i64 %indvars.iv.next.i to i32
  %tobool.i = icmp eq i32 %tmp9, 0
  br i1 %tobool.i, label %while.cond.i, label %while.body.i

while.cond.i:                                     ; preds = %land.rhs.i, %while.body.i, %for.body3
  %indvars.iv.i64 = phi i64 [ %indvars.iv.next.i65, %land.rhs.i ], [ 0, %for.body3 ], [ %tmp6, %while.body.i ]
  %indvars.iv.next.i65 = add nsw i64 %indvars.iv.i64, -1
  %tmp10 = trunc i64 %indvars.iv.i64 to i32
  %tobool.i66 = icmp eq i32 %tmp10, 0
  br i1 %tobool.i66, label %if.else, label %land.rhs.i

land.rhs.i:                                       ; preds = %while.cond.i
  %arrayidx.i67 = getelementptr inbounds i32, i32* %call4, i64 %indvars.iv.next.i65
  %tmp11 = load i32, i32* %arrayidx.i67, align 4
  %arrayidx2.i68 = getelementptr inbounds i32, i32* %data, i64 %indvars.iv.next.i65
  %tmp12 = load i32, i32* %arrayidx2.i68, align 4
  %cmp.i69 = icmp eq i32 %tmp11, %tmp12
  br i1 %cmp.i69, label %while.cond.i, label %equal_data.exit

equal_data.exit:                                  ; preds = %land.rhs.i
  %cmp3.i = icmp slt i32 %tmp10, 1
  br i1 %cmp3.i, label %if.else, label %if.then

if.then:                                          ; preds = %equal_data.exit
  %next7 = getelementptr inbounds %struct._list, %struct._list* %current.173, i64 0, i32 1
  %tmp14 = load %struct._list*, %struct._list** %next7, align 8
  %next12 = getelementptr inbounds %struct._list, %struct._list* %tmp14, i64 0, i32 1
  store %struct._list* null, %struct._list** %next12, align 8
  %tmp15 = load %struct._list*, %struct._list** %next7, align 8
  %tmp16 = load i32, i32* %value, align 4
  %cmp14 = icmp eq i32 %tmp16, 1
  %.tmp16 = select i1 %cmp14, i32 0, i32 %tmp16
  %tmp18 = load i32, i32* %all, align 4
  %tmp19 = or i32 %tmp18, %.tmp16
  %tmp20 = icmp eq i32 %tmp19, 0
  br i1 %tmp20, label %if.then19, label %if.end31

if.then19:                                        ; preds = %if.then
  %call21 = tail call i32 @in_wanted(i32* %call4)
  br label %if.end31

if.else:                                          ; preds = %equal_data.exit, %while.cond.i
  %cmp26 = icmp eq i32 %col.071, 0
  %.row.172 = select i1 %cmp26, i32 0, i32 %row.172
  %sub30 = add nsw i32 %tmp5, -1
  br label %if.end31

if.end31:                                         ; preds = %if.else, %if.then19, %if.then
  %col.1 = phi i32 [ %sub30, %if.else ], [ 0, %if.then ], [ 0, %if.then19 ]
  %row.3 = phi i32 [ %.row.172, %if.else ], [ %row.172, %if.then ], [ 0, %if.then19 ]
  %current.2 = phi %struct._list* [ %current.173, %if.else ], [ %tmp15, %if.then ], [ %tmp15, %if.then19 ]
  %inc = add nsw i32 %col.1, 1
  %tmp25 = load i32, i32* @ncol, align 4
  %cmp2 = icmp eq i32 %inc, %tmp25
  br i1 %cmp2, label %for.cond1.for.inc32_crit_edge, label %for.body3

for.cond1.for.inc32_crit_edge:                    ; preds = %if.end31
  %.pre79 = load i32, i32* @nrow, align 4
  br label %for.inc32

for.inc32:                                        ; preds = %for.cond1.for.inc32_crit_edge, %for.cond1.preheader
  %tmp26 = phi i32 [ %.pre79, %for.cond1.for.inc32_crit_edge ], [ 0, %for.cond1.preheader ]
  %current.1.lcssa = phi %struct._list* [ %current.2, %for.cond1.for.inc32_crit_edge ], [ %current.077, %for.cond1.preheader ]
  %row.1.lcssa = phi i32 [ %row.3, %for.cond1.for.inc32_crit_edge ], [ 0, %for.cond1.preheader ]
  %inc33 = add nsw i32 %row.1.lcssa, 1
  %cmp = icmp eq i32 %inc33, %tmp26
  br i1 %cmp, label %for.end34, label %for.cond1.preheader

for.end34:                                        ; preds = %for.inc32
  %.pre = load %struct._list*, %struct._list** %tmp, align 8
  ret %struct._list* %.pre
}
