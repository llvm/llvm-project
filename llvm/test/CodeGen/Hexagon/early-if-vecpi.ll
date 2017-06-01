; RUN: llc -march=hexagon < %s | FileCheck %s

target triple = "hexagon-unknown--elf"

; Check that we can predicate base+offset vector stores.
; CHECK-LABEL: sammy
; CHECK: if{{.*}}vmem(r{{[0-9]+}}+#0) =
define void @sammy(<16 x i32>* nocapture %p, <16 x i32>* nocapture readonly %q, i32 %n) #0 {
entry:
  %0 = load <16 x i32>, <16 x i32>* %q, align 64
  %sub = add nsw i32 %n, -1
  br label %for.body

for.body:                                         ; preds = %if.end, %entry
  %p.addr.011 = phi <16 x i32>* [ %p, %entry ], [ %incdec.ptr, %if.end ]
  %i.010 = phi i32 [ 0, %entry ], [ %add, %if.end ]
  %mul = mul nsw i32 %i.010, %sub
  %add = add nuw nsw i32 %i.010, 1
  %mul1 = mul nsw i32 %add, %n
  %cmp2 = icmp slt i32 %mul, %mul1
  br i1 %cmp2, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  store <16 x i32> %0, <16 x i32>* %p.addr.011, align 64
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %incdec.ptr = getelementptr inbounds <16 x i32>, <16 x i32>* %p.addr.011, i32 1
  %exitcond = icmp eq i32 %add, 100
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %if.end
  ret void
}

; Check that we can predicate post-increment vector stores.
; CHECK-LABEL: danny
; CHECK: if{{.*}}vmem(r{{[0-9]+}}++#1) =
define void @danny(<16 x i32>* nocapture %p, <16 x i32>* nocapture readonly %q, i32 %n) #0 {
entry:
  %0 = load <16 x i32>, <16 x i32>* %q, align 64
  %sub = add nsw i32 %n, -1
  br label %for.body

for.body:                                         ; preds = %if.end, %entry
  %p.addr.012 = phi <16 x i32>* [ %p, %entry ], [ %incdec.ptr3, %if.end ]
  %i.011 = phi i32 [ 0, %entry ], [ %add, %if.end ]
  %mul = mul nsw i32 %i.011, %sub
  %add = add nuw nsw i32 %i.011, 1
  %mul1 = mul nsw i32 %add, %n
  %cmp2 = icmp slt i32 %mul, %mul1
  br i1 %cmp2, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %incdec.ptr = getelementptr inbounds <16 x i32>, <16 x i32>* %p.addr.012, i32 1
  store <16 x i32> %0, <16 x i32>* %p.addr.012, align 64
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %p.addr.1 = phi <16 x i32>* [ %incdec.ptr, %if.then ], [ %p.addr.012, %for.body ]
  %incdec.ptr3 = getelementptr inbounds <16 x i32>, <16 x i32>* %p.addr.1, i32 1
  %exitcond = icmp eq i32 %add, 100
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %if.end
  ret void
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,-hvx-double" }
