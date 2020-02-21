
; RUN: llc -mtriple=i686-linux -pre-RA-sched=source < %s | FileCheck %s
; RUN: opt -disable-output -debugify < %s

; This was derived from the Linux kernel. The __builtin_expect was ignored
; which pushed the hot block "if.else" out of the critical path choosing
; instead the cold block "if.then23". The cold block should be moved towards
; the bottom.

; CHECK-LABEL: test1:
; CHECK:       %for.inc
; CHECK:       %if.end18
; CHECK:       %if.else
; CHECK:       %if.end.i.i
; CHECK:       %if.end8.i.i
; CHECK:       %if.then23
; CHECK:       ret

%struct.hlist_bl_node = type { %struct.hlist_bl_node*, %struct.hlist_bl_node** }
%struct.dentry = type { i32, %struct.inode, %struct.hlist_bl_node, %struct.dentry*, %struct.inode, %struct.inode*, [32 x i8], %struct.inode, %struct.dentry_operations* }
%struct.inode = type { i32 }
%struct.dentry_operations = type { i32 (%struct.dentry*, i32)*, i32 (%struct.dentry*, i32)*, i32 (%struct.dentry*, %struct.inode*)*, i32 (%struct.dentry*, i32, i8*)* }
%struct.anon.2 = type { i32, i32 }

define %struct.dentry* @test1(%struct.dentry* readonly %parent, i8* %name, i32* nocapture %seqp, i64 %param1) {
entry:
  %tobool135 = icmp eq i64 %param1, 0
  br i1 %tobool135, label %cleanup63, label %do.body4.lr.ph

do.body4.lr.ph:                                   ; preds = %entry
  %d_op = getelementptr inbounds %struct.dentry, %struct.dentry* %parent, i64 0, i32 8
  %shr = lshr i64 %param1, 32
  %conv49 = trunc i64 %shr to i32
  br label %do.body4

do.body4:                                         ; preds = %for.inc, %do.body4.lr.ph
  %node.0.in136 = phi i64 [ %param1, %do.body4.lr.ph ], [ %tmp35, %for.inc ]
  %node.0 = inttoptr i64 %node.0.in136 to %struct.hlist_bl_node*
  %add.ptr = getelementptr %struct.hlist_bl_node, %struct.hlist_bl_node* %node.0, i64 -1, i32 1
  %tmp6 = bitcast %struct.hlist_bl_node*** %add.ptr to %struct.dentry*
  %tmp7 = getelementptr inbounds %struct.dentry, %struct.dentry* %tmp6, i64 0, i32 1, i32 0
  %tmp8 = load volatile i32, i32* %tmp7, align 4
  call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  %d_parent = getelementptr inbounds %struct.hlist_bl_node**, %struct.hlist_bl_node*** %add.ptr, i64 3
  %tmp9 = bitcast %struct.hlist_bl_node*** %d_parent to %struct.dentry**
  %tmp10 = load %struct.dentry*, %struct.dentry** %tmp9, align 8
  %cmp133 = icmp eq %struct.dentry* %tmp10, %parent
  br i1 %cmp133, label %if.end14.lr.ph, label %for.inc

if.end14.lr.ph:                                   ; preds = %do.body4
  %tmp11 = getelementptr inbounds %struct.hlist_bl_node**, %struct.hlist_bl_node*** %add.ptr, i64 2
  %d_name43 = getelementptr inbounds %struct.hlist_bl_node**, %struct.hlist_bl_node*** %add.ptr, i64 4
  %hash = bitcast %struct.hlist_bl_node*** %d_name43 to i32*
  %tmp12 = bitcast %struct.hlist_bl_node*** %d_name43 to %struct.anon.2*
  %len = getelementptr inbounds %struct.anon.2, %struct.anon.2* %tmp12, i64 0, i32 1
  %name31 = getelementptr inbounds %struct.hlist_bl_node**, %struct.hlist_bl_node*** %add.ptr, i64 5
  %tmp13 = bitcast %struct.hlist_bl_node*** %name31 to i8**
  br label %if.end14

if.end14:                                         ; preds = %cleanup, %if.end14.lr.ph
  %and.i100134.in = phi i32 [ %tmp8, %if.end14.lr.ph ], [ undef, %cleanup ]
  %and.i100134 = and i32 %and.i100134.in, -2
  %tmp14 = load %struct.hlist_bl_node**, %struct.hlist_bl_node*** %tmp11, align 8
  %tobool.i.i = icmp eq %struct.hlist_bl_node** %tmp14, null
  br i1 %tobool.i.i, label %for.inc, label %if.end18

if.end18:                                         ; preds = %if.end14
  %tmp15 = load i32, i32* %seqp, align 8
  %tmp16 = and i32 %tmp15, 2
  %tobool22 = icmp eq i32 %tmp16, 0
  br i1 %tobool22, label %if.else, label %if.then23, !prof !0, !misexpect !1

if.then23:                                        ; preds = %if.end18
  %tmp17 = load i32, i32* %hash, align 8
  %cmp25 = icmp eq i32 %tmp17, 42
  br i1 %cmp25, label %if.end28, label %for.inc

if.end28:                                         ; preds = %if.then23
  %tmp18 = load i32, i32* %len, align 4
  %tmp19 = load i8*, i8** %tmp13, align 8
  call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  %tmp20 = load i32, i32* %tmp7, align 4
  %cmp.i.i101 = icmp eq i32 %tmp20, %and.i100134
  br i1 %cmp.i.i101, label %if.end36, label %cleanup

if.end36:                                         ; preds = %if.end28
  %tmp21 = load %struct.dentry_operations*, %struct.dentry_operations** %d_op, align 8
  %d_compare = getelementptr inbounds %struct.dentry_operations, %struct.dentry_operations* %tmp21, i64 0, i32 3
  %tmp22 = load i32 (%struct.dentry*, i32, i8*)*, i32 (%struct.dentry*, i32, i8*)** %d_compare, align 8
  %call37 = call i32 %tmp22(%struct.dentry* %tmp6, i32 %tmp18, i8* %name)
  %cmp38 = icmp eq i32 %call37, 0
  br i1 %cmp38, label %cleanup56, label %for.inc

cleanup:                                          ; preds = %if.end28
  %tmp24 = load %struct.dentry*, %struct.dentry** %tmp9, align 8
  %cmp = icmp eq %struct.dentry* null, %parent
  br i1 %cmp, label %if.end14, label %for.inc

if.else:                                          ; preds = %if.end18
  %hash_len44 = bitcast %struct.hlist_bl_node*** %d_name43 to i64*
  %tmp25 = load i64, i64* %hash_len44, align 8
  %cmp45 = icmp eq i64 %tmp25, %param1
  br i1 %cmp45, label %if.end48, label %for.inc

if.end48:                                         ; preds = %if.else
  %tmp26 = bitcast %struct.hlist_bl_node*** %name31 to i64*
  %tmp27 = load volatile i64, i64* %tmp26, align 8
  %tmp28 = inttoptr i64 %tmp27 to i8*
  br label %for.cond.i.i

for.cond.i.i:                                     ; preds = %if.end8.i.i, %if.end48
  %tcount.addr.0.i.i = phi i32 [ %conv49, %if.end48 ], [ %sub.i.i, %if.end8.i.i ]
  %ct.addr.0.i.i = phi i8* [ %name, %if.end48 ], [ %add.ptr9.i.i, %if.end8.i.i ]
  %cs.addr.0.i.i = phi i8* [ %tmp28, %if.end48 ], [ %add.ptr.i.i, %if.end8.i.i ]
  %tmp29 = bitcast i8* %cs.addr.0.i.i to i64*
  %tmp30 = load i64, i64* %tmp29, align 8
  %tmp31 = bitcast i8* %ct.addr.0.i.i to i64*
  %tmp32 = call { i64, i64 } asm "1:\09mov $2,$0\0A2:\0A.section .fixup,\22ax\22\0A3:\09lea $2,$1\0A\09and $3,$1\0A\09mov ($1),$0\0A\09leal $2,%ecx\0A\09andl $4,%ecx\0A\09shll $$3,%ecx\0A\09shr %cl,$0\0A\09jmp 2b\0A.previous\0A .pushsection \22__ex_table\22,\22a\22\0A .balign 4\0A .long (1b) - .\0A .long (3b) - .\0A .long (ex_handler_default) - .\0A .popsection\0A", "=&r,=&{cx},*m,i,i,~{dirflag},~{fpsr},~{flags}"(i64* %tmp31, i64 -8, i64 7)
  %cmp.i.i = icmp ult i32 %tcount.addr.0.i.i, 8
  %asmresult.i.le.i.le.i.le = extractvalue { i64, i64 } %tmp32, 0
  br i1 %cmp.i.i, label %dentry_cmp.exit, label %if.end.i.i

if.end.i.i:                                       ; preds = %for.cond.i.i
  %cmp3.i.i = icmp eq i64 %tmp30, %asmresult.i.le.i.le.i.le
  br i1 %cmp3.i.i, label %if.end8.i.i, label %for.inc, !prof !0, !misexpect !1

if.end8.i.i:                                      ; preds = %if.end.i.i
  %add.ptr.i.i = getelementptr i8, i8* %cs.addr.0.i.i, i64 8
  %add.ptr9.i.i = getelementptr i8, i8* %ct.addr.0.i.i, i64 8
  %sub.i.i = add i32 %tcount.addr.0.i.i, -8
  %tobool12.i.i = icmp eq i32 %sub.i.i, 0
  br i1 %tobool12.i.i, label %cleanup56, label %for.cond.i.i

dentry_cmp.exit:                                  ; preds = %for.cond.i.i
  %asmresult.i.le.i.le.i.le.le = extractvalue { i64, i64 } %tmp32, 0
  %mul.i.i = shl nuw nsw i32 %tcount.addr.0.i.i, 3
  %sh_prom.i.i = zext i32 %mul.i.i to i64
  %shl.i.i = shl nsw i64 -1, %sh_prom.i.i
  %neg.i.i = xor i64 %shl.i.i, -1
  %xor.i.i = xor i64 %asmresult.i.le.i.le.i.le.le, %tmp30
  %and.i.i = and i64 %xor.i.i, %neg.i.i
  %tobool15.i.i = icmp eq i64 %and.i.i, 0
  br i1 %tobool15.i.i, label %cleanup56, label %for.inc

cleanup56:                                        ; preds = %dentry_cmp.exit, %if.end8.i.i, %if.end36
  %tmp33 = bitcast %struct.hlist_bl_node*** %add.ptr to %struct.dentry*
  store i32 %and.i100134, i32* %seqp, align 4
  br label %cleanup63

for.inc:                                          ; preds = %dentry_cmp.exit, %if.end.i.i, %if.else, %cleanup, %if.end36, %if.then23, %if.end14, %do.body4
  %tmp34 = inttoptr i64 %node.0.in136 to i64*
  %tmp35 = load volatile i64, i64* %tmp34, align 8
  %tobool = icmp eq i64 %tmp35, 0
  br i1 %tobool, label %cleanup63, label %do.body4

cleanup63:                                        ; preds = %for.inc, %cleanup56, %entry
  %retval.2 = phi %struct.dentry* [ %tmp33, %cleanup56 ], [ null, %entry ], [ null, %for.inc ]
  ret %struct.dentry* %retval.2
}

!0 = !{!"branch_weights", i32 2000, i32 1}
!1 = !{!"misexpect", i64 1, i64 2000, i64 1}
