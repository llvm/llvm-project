; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @bar(i32 %v) local_unnamed_addr #0 {
entry:
  %mul = shl nsw i32 %v, 1
  ret i32 %mul
}

; Function Attrs: norecurse nounwind readonly uwtable
define i32 @foo(i8* nocapture readonly %p) #1 {
entry:
  %targets = alloca [256 x i8*], align 16
  %arrayidx1 = getelementptr inbounds [256 x i8*], [256 x i8*]* %targets, i64 0, i64 93
  store i8* blockaddress(@foo, %if.end), i8** %arrayidx1, align 8
  br label %for.cond2

for.cond2:                                        ; preds = %if.end, %for.cond2, %entry
; CHECK: for.cond2:                                        ; preds = %.split1
  %p.addr.0 = phi i8* [ %p, %entry ], [ %incdec.ptr5, %if.end ], [ %incdec.ptr, %for.cond2 ]
  %incdec.ptr = getelementptr inbounds i8, i8* %p.addr.0, i64 1
  %0 = load i8, i8* %p.addr.0, align 1
  %cond = icmp eq i8 %0, 93
  br i1 %cond, label %if.end.preheader, label %for.cond2

if.end.preheader:                                 ; preds = %for.cond2
  br label %if.end

if.end:                                           ; preds = %if.end.preheader, %if.end
; CHECK: if.end:                                           ; preds = %.split1
  %p.addr.1 = phi i8* [ %incdec.ptr5, %if.end ], [ %incdec.ptr, %if.end.preheader ]
  %incdec.ptr5 = getelementptr inbounds i8, i8* %p.addr.1, i64 1
  %1 = load i8, i8* %p.addr.1, align 1
  %idxprom6 = zext i8 %1 to i64
  %arrayidx7 = getelementptr inbounds [256 x i8*], [256 x i8*]* %targets, i64 0, i64 %idxprom6
  %2 = load i8*, i8** %arrayidx7, align 8
  indirectbr i8* %2, [label %for.cond2, label %if.end]
; CHECK: indirectbr i8* %2, [label %for.cond2, label %if.end]
}

;; If an indirectbr critical edge cannot be split, ignore it.
;; The edge will not be profiled.
; CHECK-LABEL: @cannot_split(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.instrprof.increment
; CHECK: indirect:
; CHECK-NOT:     call void @llvm.instrprof.increment
; CHECK: indirect2:
; CHECK-NEXT:    call void @llvm.instrprof.increment
define i32 @cannot_split(i8* nocapture readonly %p) {
entry:
  %targets = alloca <2 x i8*>, align 16
  store <2 x i8*> <i8* blockaddress(@cannot_split, %indirect), i8* blockaddress(@cannot_split, %end)>, <2 x i8*>* %targets, align 16
  %arrayidx2 = getelementptr inbounds i8, i8* %p, i64 1
  %0 = load i8, i8* %arrayidx2
  %idxprom = sext i8 %0 to i64
  %arrayidx3 = getelementptr inbounds <2 x i8*>, <2 x i8*>* %targets, i64 0, i64 %idxprom
  %1 = load i8*, i8** %arrayidx3, align 8
  br label %indirect

indirect:                                         ; preds = %entry, %indirect
  indirectbr i8* %1, [label %indirect, label %end, label %indirect2]

indirect2:
  ; For this test we do not want critical edges split. Adding a 2nd `indirectbr`
  ; does the trick.
  indirectbr i8* %1, [label %indirect, label %end]

end:                                              ; preds = %indirect
  ret i32 0
}
