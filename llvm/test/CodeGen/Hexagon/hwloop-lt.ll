; RUN: llc -mtriple=hexagon -O3 < %s | FileCheck %s

; CHECK-LABEL: @test_pos1_ir_slt
; CHECK: loop0
; a < b
define void @test_pos1_ir_slt(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 8531, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  br label %for.body

for.body:
  %i.04 = phi i32 [ 8531, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 1
  %cmp = icmp slt i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: @test_pos2_ir_slt
; CHECK: loop0
; a < b
define void @test_pos2_ir_slt(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 9152, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  br label %for.body

for.body:
  %i.04 = phi i32 [ 9152, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 2
  %cmp = icmp slt i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: @test_pos4_ir_slt
; CHECK: loop0
; a < b
define void @test_pos4_ir_slt(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 18851, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  br label %for.body

for.body:
  %i.04 = phi i32 [ 18851, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 4
  %cmp = icmp slt i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: @test_pos8_ir_slt
; CHECK: loop0
; a < b
define void @test_pos8_ir_slt(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 25466, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  br label %for.body

for.body:
  %i.04 = phi i32 [ 25466, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 8
  %cmp = icmp slt i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: @test_pos16_ir_slt
; CHECK: loop0
; a < b
define void @test_pos16_ir_slt(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 9295, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  br label %for.body

for.body:
  %i.04 = phi i32 [ 9295, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 16
  %cmp = icmp slt i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: @test_pos1_ri_slt
; CHECK: loop0
; a < b
define void @test_pos1_ri_slt(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, 31236
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  br label %for.body

for.body:
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 1
  %cmp = icmp slt i32 %inc, 31236
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: @test_pos2_ri_slt
; CHECK: loop0
; a < b
define void @test_pos2_ri_slt(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, 22653
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  br label %for.body

for.body:
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 2
  %cmp = icmp slt i32 %inc, 22653
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: @test_pos4_ri_slt
; CHECK: loop0
; a < b
define void @test_pos4_ri_slt(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, 1431
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  br label %for.body

for.body:
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 4
  %cmp = icmp slt i32 %inc, 1431
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: @test_pos8_ri_slt
; CHECK: loop0
; a < b
define void @test_pos8_ri_slt(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, 22403
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  br label %for.body

for.body:
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 8
  %cmp = icmp slt i32 %inc, 22403
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: @test_pos16_ri_slt
; CHECK: loop0
; a < b
define void @test_pos16_ri_slt(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, 21715
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  br label %for.body

for.body:
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 16
  %cmp = icmp slt i32 %inc, 21715
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: @test_pos1_rr_slt
; CHECK: loop0
; a < b
define void @test_pos1_rr_slt(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  br label %for.body

for.body:
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 1
  %cmp = icmp slt i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: @test_pos2_rr_slt
; CHECK: loop0
; a < b
define void @test_pos2_rr_slt(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  br label %for.body

for.body:
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 2
  %cmp = icmp slt i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: @test_pos4_rr_slt
; CHECK: loop0
; a < b
define void @test_pos4_rr_slt(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  br label %for.body

for.body:
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 4
  %cmp = icmp slt i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: @test_pos8_rr_slt
; CHECK: loop0
; a < b
define void @test_pos8_rr_slt(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  br label %for.body

for.body:
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 8
  %cmp = icmp slt i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: @test_pos16_rr_slt
; CHECK: loop0
; a < b
define void @test_pos16_rr_slt(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  br label %for.body

for.body:
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 16
  %cmp = icmp slt i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}



