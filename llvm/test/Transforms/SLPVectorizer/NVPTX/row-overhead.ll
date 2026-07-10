; RUN: opt < %s -passes=slp-vectorizer -S -mtriple=nvptx64-nvidia-cuda -mcpu=sm_100 | FileCheck %s

; Keep the row/control values scalar across the branchy tail merge.

target triple = "nvptx64-nvidia-cuda"

define void @slp_row_overhead_min(i32 %nnz, ptr addrspace(1) %rows,
                                  ptr addrspace(1) %out, i1 %full,
                                  i1 %aligned) {
; CHECK-LABEL: @slp_row_overhead_min(
; CHECK-NOT: phi <{{[0-9]+}} x i32>
; CHECK-NOT: icmp {{.*}} <{{[0-9]+}} x i32>
; CHECK-NOT: add {{.*}} <{{[0-9]+}} x i32>
; CHECK-NOT: sub {{.*}} <{{[0-9]+}} x i32>
; CHECK-NOT: store <{{[0-9]+}} x i32>
; CHECK: ret void
entry:
  %ctaid = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %tid = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block.base = mul nuw nsw i32 %ctaid, 896
  %block.base64 = zext i32 %block.base to i64
  %row.block = getelementptr inbounds i32, ptr addrspace(1) %rows, i64 %block.base64
  br i1 %full, label %full.block, label %tail.block

full.block:
  br i1 %aligned, label %aligned.loads, label %unaligned.loads

aligned.loads:
  %lane.base.a = mul nuw nsw i32 %tid, 7
  %lane.base.a64 = zext i32 %lane.base.a to i64
  %a0p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %lane.base.a64
  %a0 = load i32, ptr addrspace(1) %a0p, align 4
  %a1i = add nuw nsw i32 %lane.base.a, 1
  %a1i64 = zext i32 %a1i to i64
  %a1p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %a1i64
  %a1 = load i32, ptr addrspace(1) %a1p, align 4
  %a2i = add nuw nsw i32 %lane.base.a, 2
  %a2i64 = zext i32 %a2i to i64
  %a2p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %a2i64
  %a2 = load i32, ptr addrspace(1) %a2p, align 4
  %a3i = add nuw nsw i32 %lane.base.a, 3
  %a3i64 = zext i32 %a3i to i64
  %a3p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %a3i64
  %a3 = load i32, ptr addrspace(1) %a3p, align 4
  %a4i = add nuw nsw i32 %lane.base.a, 4
  %a4i64 = zext i32 %a4i to i64
  %a4p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %a4i64
  %a4 = load i32, ptr addrspace(1) %a4p, align 4
  %a5i = add nuw nsw i32 %lane.base.a, 5
  %a5i64 = zext i32 %a5i to i64
  %a5p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %a5i64
  %a5 = load i32, ptr addrspace(1) %a5p, align 4
  %a6i = add nuw nsw i32 %lane.base.a, 6
  %a6i64 = zext i32 %a6i to i64
  %a6p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %a6i64
  %a6 = load i32, ptr addrspace(1) %a6p, align 4
  br label %merge

unaligned.loads:
  %lane.base.b = mul nuw nsw i32 %tid, 7
  %lane.base.b64 = zext i32 %lane.base.b to i64
  %b0p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %lane.base.b64
  %b0 = load i32, ptr addrspace(1) %b0p, align 4
  %b1i = add nuw nsw i32 %lane.base.b, 1
  %b1i64 = zext i32 %b1i to i64
  %b1p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %b1i64
  %b1 = load i32, ptr addrspace(1) %b1p, align 4
  %b2i = add nuw nsw i32 %lane.base.b, 2
  %b2i64 = zext i32 %b2i to i64
  %b2p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %b2i64
  %b2 = load i32, ptr addrspace(1) %b2p, align 4
  %b3i = add nuw nsw i32 %lane.base.b, 3
  %b3i64 = zext i32 %b3i to i64
  %b3p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %b3i64
  %b3 = load i32, ptr addrspace(1) %b3p, align 4
  %b4i = add nuw nsw i32 %lane.base.b, 4
  %b4i64 = zext i32 %b4i to i64
  %b4p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %b4i64
  %b4 = load i32, ptr addrspace(1) %b4p, align 4
  %b5i = add nuw nsw i32 %lane.base.b, 5
  %b5i64 = zext i32 %b5i to i64
  %b5p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %b5i64
  %b5 = load i32, ptr addrspace(1) %b5p, align 4
  %b6i = add nuw nsw i32 %lane.base.b, 6
  %b6i64 = zext i32 %b6i to i64
  %b6p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %b6i64
  %b6 = load i32, ptr addrspace(1) %b6p, align 4
  br label %merge

tail.block:
  %tail.left = sub nsw i32 %nnz, %block.base
  %tail.last.i = add nsw i32 %tail.left, -1
  %tail.last.i64 = sext i32 %tail.last.i to i64
  %lastp = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %tail.last.i64
  %last = load i32, ptr addrspace(1) %lastp, align 4
  %lane.base.t = mul nuw nsw i32 %tid, 7
  %tail.rem = sub nsw i32 %tail.left, %lane.base.t
  %tail.has0 = icmp sgt i32 %tail.rem, 0
  br i1 %tail.has0, label %tail.load0, label %tail.merge0

tail.load0:
  %t0i64 = zext i32 %lane.base.t to i64
  %t0p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %t0i64
  %t0 = load i32, ptr addrspace(1) %t0p, align 4
  br label %tail.merge0

tail.merge0:
  %c0 = phi i32 [ %t0, %tail.load0 ], [ %last, %tail.block ]
  %tail.has1 = icmp sgt i32 %tail.rem, 1
  br i1 %tail.has1, label %tail.load1, label %tail.merge1

tail.load1:
  %t1i = add nuw nsw i32 %lane.base.t, 1
  %t1i64 = zext i32 %t1i to i64
  %t1p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %t1i64
  %t1 = load i32, ptr addrspace(1) %t1p, align 4
  br label %tail.merge1

tail.merge1:
  %c1 = phi i32 [ %t1, %tail.load1 ], [ %last, %tail.merge0 ]
  %tail.has2 = icmp sgt i32 %tail.rem, 2
  br i1 %tail.has2, label %tail.load2, label %tail.merge2

tail.load2:
  %t2i = add nuw nsw i32 %lane.base.t, 2
  %t2i64 = zext i32 %t2i to i64
  %t2p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %t2i64
  %t2 = load i32, ptr addrspace(1) %t2p, align 4
  br label %tail.merge2

tail.merge2:
  %c2 = phi i32 [ %t2, %tail.load2 ], [ %last, %tail.merge1 ]
  %tail.has3 = icmp sgt i32 %tail.rem, 3
  br i1 %tail.has3, label %tail.load3, label %tail.merge3

tail.load3:
  %t3i = add nuw nsw i32 %lane.base.t, 3
  %t3i64 = zext i32 %t3i to i64
  %t3p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %t3i64
  %t3 = load i32, ptr addrspace(1) %t3p, align 4
  br label %tail.merge3

tail.merge3:
  %c3 = phi i32 [ %t3, %tail.load3 ], [ %last, %tail.merge2 ]
  %tail.has4 = icmp sgt i32 %tail.rem, 4
  br i1 %tail.has4, label %tail.load4, label %tail.merge4

tail.load4:
  %t4i = add nuw nsw i32 %lane.base.t, 4
  %t4i64 = zext i32 %t4i to i64
  %t4p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %t4i64
  %t4 = load i32, ptr addrspace(1) %t4p, align 4
  br label %tail.merge4

tail.merge4:
  %c4 = phi i32 [ %t4, %tail.load4 ], [ %last, %tail.merge3 ]
  %tail.has5 = icmp sgt i32 %tail.rem, 5
  br i1 %tail.has5, label %tail.load5, label %tail.merge5

tail.load5:
  %t5i = add nuw nsw i32 %lane.base.t, 5
  %t5i64 = zext i32 %t5i to i64
  %t5p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %t5i64
  %t5 = load i32, ptr addrspace(1) %t5p, align 4
  br label %tail.merge5

tail.merge5:
  %c5 = phi i32 [ %t5, %tail.load5 ], [ %last, %tail.merge4 ]
  %tail.has6 = icmp sgt i32 %tail.rem, 6
  br i1 %tail.has6, label %tail.load6, label %merge

tail.load6:
  %t6i = add nuw nsw i32 %lane.base.t, 6
  %t6i64 = zext i32 %t6i to i64
  %t6p = getelementptr inbounds i32, ptr addrspace(1) %row.block, i64 %t6i64
  %t6 = load i32, ptr addrspace(1) %t6p, align 4
  br label %merge

merge:
  %r6 = phi i32 [ %a6, %aligned.loads ], [ %b6, %unaligned.loads ], [ %t6, %tail.load6 ], [ %last, %tail.merge5 ]
  %r5 = phi i32 [ %a5, %aligned.loads ], [ %b5, %unaligned.loads ], [ %c5, %tail.load6 ], [ %c5, %tail.merge5 ]
  %r4 = phi i32 [ %a4, %aligned.loads ], [ %b4, %unaligned.loads ], [ %c4, %tail.load6 ], [ %c4, %tail.merge5 ]
  %r3 = phi i32 [ %a3, %aligned.loads ], [ %b3, %unaligned.loads ], [ %c3, %tail.load6 ], [ %c3, %tail.merge5 ]
  %r2 = phi i32 [ %a2, %aligned.loads ], [ %b2, %unaligned.loads ], [ %c2, %tail.load6 ], [ %c2, %tail.merge5 ]
  %r1 = phi i32 [ %a1, %aligned.loads ], [ %b1, %unaligned.loads ], [ %c1, %tail.load6 ], [ %c1, %tail.merge5 ]
  %r0 = phi i32 [ %a0, %aligned.loads ], [ %b0, %unaligned.loads ], [ %c0, %tail.load6 ], [ %c0, %tail.merge5 ]
  %sh0 = tail call i32 asm sideeffect "shfl.sync.idx.b32 $0, $1, $2, 31, -1;", "=r,r,r"(i32 %r0, i32 0)
  %sh6 = tail call i32 asm sideeffect "shfl.sync.idx.b32 $0, $1, $2, 31, -1;", "=r,r,r"(i32 %r6, i32 31)
  %cmp.ends = icmp eq i32 %sh0, %sh6
  br i1 %cmp.ends, label %same.row, label %split.row

same.row:
  store i32 %sh0, ptr addrspace(1) %out, align 4
  ret void

split.row:
  %up6 = tail call i32 asm sideeffect "shfl.sync.up.b32 $0, $1, $2, 0, -1;", "=r,r,r"(i32 %r6, i32 1)
  %d0 = sub nsw i32 %r0, %sh0
  %d1 = sub nsw i32 %r1, %sh0
  %d2 = sub nsw i32 %r2, %sh0
  %d3 = sub nsw i32 %r3, %sh0
  %d4 = sub nsw i32 %r4, %sh0
  %d5 = sub nsw i32 %r5, %sh0
  %cmp06 = icmp ne i32 %r0, %r6
  %cmpup = icmp ne i32 %up6, %r0
  %cmp.int = zext i1 %cmp06 to i32
  %cmpup.int = zext i1 %cmpup to i32
  %mix0 = add i32 %d0, %cmp.int
  %mix1 = add i32 %d5, %cmpup.int
  store i32 %mix0, ptr addrspace(1) %out, align 4
  %out1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %d1, ptr addrspace(1) %out1, align 4
  %out2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 2
  store i32 %d2, ptr addrspace(1) %out2, align 4
  %out3 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 3
  store i32 %d3, ptr addrspace(1) %out3, align 4
  %out4 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 4
  store i32 %d4, ptr addrspace(1) %out4, align 4
  %out5 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 5
  store i32 %mix1, ptr addrspace(1) %out5, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
