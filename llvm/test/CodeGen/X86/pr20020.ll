; RUN: llc < %s -mtriple=x86_64-apple-macosx -disable-lsr -post-RA-scheduler=1 -break-anti-dependencies=critical  | FileCheck %s

; In PR20020, the critical anti-dependency breaker algorithm mistakenly
; changes the register operands of an 'xorl %eax, %eax' to 'xorl %ecx, %ecx'
; and then immediately reloads %rcx with a value based on the wrong %rax

; CHECK-NOT: xorl %ecx, %ecx
; CHECK: leaq 1(%rax), %rcx


%struct.planet = type { double, double, double }

; Function Attrs: nounwind ssp uwtable
define void @advance(i32 %nbodies, %struct.planet* nocapture %bodies) #0 {
entry:
  %cmp4 = icmp sgt i32 %nbodies, 0
  br i1 %cmp4, label %for.body.preheader, label %for.end38

for.body.preheader:                               ; preds = %entry
  %gep = getelementptr %struct.planet, %struct.planet* %bodies, i64 1, i32 1
  %gep13 = bitcast double* %gep to %struct.planet*
  %0 = add i32 %nbodies, -1
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.inc20
  %iv19 = phi i32 [ %0, %for.body.preheader ], [ %iv.next, %for.inc20 ]
  %iv = phi %struct.planet* [ %gep13, %for.body.preheader ], [ %gep14, %for.inc20 ]
  %iv9 = phi i64 [ %iv.next10, %for.inc20 ], [ 0, %for.body.preheader ]
  %iv.next10 = add nuw nsw i64 %iv9, 1
  %1 = trunc i64 %iv.next10 to i32
  %cmp22 = icmp slt i32 %1, %nbodies
  br i1 %cmp22, label %for.body3.lr.ph, label %for.inc20

for.body3.lr.ph:                                  ; preds = %for.body
  %x = getelementptr inbounds %struct.planet, %struct.planet* %bodies, i64 %iv9, i32 0
  %y = getelementptr inbounds %struct.planet, %struct.planet* %bodies, i64 %iv9, i32 1
  %vx = getelementptr inbounds %struct.planet, %struct.planet* %bodies, i64 %iv9, i32 2
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body3.lr.ph
  %iv20 = phi i32 [ %iv.next21, %for.body3 ], [ %iv19, %for.body3.lr.ph ]
  %iv15 = phi %struct.planet* [ %gep16, %for.body3 ], [ %iv, %for.body3.lr.ph ]
  %iv1517 = bitcast %struct.planet* %iv15 to double*
  %2 = load double, double* %x, align 8
  %gep18 = getelementptr double, double* %iv1517, i64 -1
  %3 = load double, double* %gep18, align 8
  %sub = fsub double %2, %3
  %4 = load double, double* %y, align 8
  %5 = load double, double* %iv1517, align 8
  %sub8 = fsub double %4, %5
  %add10 = fadd double %sub, %sub8
  %call = tail call double @sqrt(double %sub8) #2
  store double %add10, double* %vx, align 8
  %gep16 = getelementptr %struct.planet, %struct.planet* %iv15, i64 1
  %iv.next21 = add i32 %iv20, -1
  %exitcond = icmp eq i32 %iv.next21, 0
  br i1 %exitcond, label %for.inc20, label %for.body3

for.inc20:                                        ; preds = %for.body3, %for.body
  %lftr.wideiv11 = trunc i64 %iv.next10 to i32
  %gep14 = getelementptr %struct.planet, %struct.planet* %iv, i64 1
  %iv.next = add i32 %iv19, -1
  %exitcond12 = icmp eq i32 %lftr.wideiv11, %nbodies
  br i1 %exitcond12, label %for.end38, label %for.body

for.end38:                                        ; preds = %for.inc20, %entry
  ret void
}

; Function Attrs: nounwind
declare double @sqrt(double) #1

attributes #0 = { "frame-pointer"="non-leaf" }
