; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @bar(i32 %v) local_unnamed_addr #0 {
entry:
  %mul = shl nsw i32 %v, 1
  ret i32 %mul
}

; Function Attrs: norecurse nounwind readonly uwtable
define i32 @foo(ptr nocapture readonly %p) #1 {
entry:
  %targets = alloca [256 x ptr], align 16
  %arrayidx1 = getelementptr inbounds [256 x ptr], ptr %targets, i64 0, i64 93
  store ptr blockaddress(@foo, %if.end), ptr %arrayidx1, align 8
  br label %for.cond2

for.cond2:                                        ; preds = %if.end, %for.cond2, %entry
; CHECK: for.cond2:                                        ; preds = %.split1
  %p.addr.0 = phi ptr [ %p, %entry ], [ %incdec.ptr5, %if.end ], [ %incdec.ptr, %for.cond2 ]
  %incdec.ptr = getelementptr inbounds i8, ptr %p.addr.0, i64 1
  %0 = load i8, ptr %p.addr.0, align 1
  %cond = icmp eq i8 %0, 93
  br i1 %cond, label %if.end.preheader, label %for.cond2

if.end.preheader:                                 ; preds = %for.cond2
  br label %if.end

if.end:                                           ; preds = %if.end.preheader, %if.end
; CHECK: if.end:                                           ; preds = %.split1
  %p.addr.1 = phi ptr [ %incdec.ptr5, %if.end ], [ %incdec.ptr, %if.end.preheader ]
  %incdec.ptr5 = getelementptr inbounds i8, ptr %p.addr.1, i64 1
  %1 = load i8, ptr %p.addr.1, align 1
  %idxprom6 = zext i8 %1 to i64
  %arrayidx7 = getelementptr inbounds [256 x ptr], ptr %targets, i64 0, i64 %idxprom6
  %2 = load ptr, ptr %arrayidx7, align 8
  indirectbr ptr %2, [label %for.cond2, label %if.end]
; CHECK: indirectbr ptr %2, [label %for.cond2, label %if.end]
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
define i32 @cannot_split(ptr nocapture readonly %p) {
entry:
  %targets = alloca <2 x ptr>, align 16
  store <2 x ptr> <ptr blockaddress(@cannot_split, %indirect), ptr blockaddress(@cannot_split, %end)>, ptr %targets, align 16
  %arrayidx2 = getelementptr inbounds i8, ptr %p, i64 1
  %0 = load i8, ptr %arrayidx2
  %idxprom = sext i8 %0 to i64
  %arrayidx3 = getelementptr inbounds <2 x ptr>, ptr %targets, i64 0, i64 %idxprom
  %1 = load ptr, ptr %arrayidx3, align 8
  br label %indirect

indirect:                                         ; preds = %entry, %indirect
  indirectbr ptr %1, [label %indirect, label %end, label %indirect2]

indirect2:
  ; For this test we do not want critical edges split. Adding a 2nd `indirectbr`
  ; does the trick.
  indirectbr ptr %1, [label %indirect, label %end]

end:                                              ; preds = %indirect
  ret i32 0
}
