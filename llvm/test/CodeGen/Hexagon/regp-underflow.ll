; RUN: llc -march=hexagon -O2 < %s
; REQUIRES: asserts

target triple = "hexagon-unknown--elf"

@g0 = global i32 0, align 4
@g1 = global i32 0, align 4
@g2 = global i32 0, align 4
@g3 = global i32 0, align 4
@g4 = common global [100 x i32] zeroinitializer, align 8
@g5 = common global [100 x i32] zeroinitializer, align 8
@g6 = private unnamed_addr constant [13 x i8] c"ping started\00"
@g7 = private unnamed_addr constant [13 x i8] c"pong started\00"

; Function Attrs: nounwind
define void @f0(ptr nocapture readnone %a0) #0 {
b0:
  tail call void @f1(ptr %a0, i32 0)
  ret void
}

; Function Attrs: nounwind
define internal void @f1(ptr nocapture readnone %a0, i32 %a1) #0 {
b0:
  %v0 = icmp eq i32 %a1, 1
  br i1 %v0, label %b2, label %b1

b1:                                               ; preds = %b0
  %v1 = tail call i32 @f3(ptr @g6)
  store volatile i32 1, ptr @g0, align 4, !tbaa !0
  br label %b3

b2:                                               ; preds = %b0
  %v2 = tail call i32 @f3(ptr @g7)
  store volatile i32 1, ptr @g1, align 4, !tbaa !0
  br label %b3

b3:                                               ; preds = %b3, %b2, %b1
  %v3 = load volatile i32, ptr @g2, align 4, !tbaa !0
  %v4 = icmp eq i32 %v3, 0
  br i1 %v4, label %b3, label %b4

b4:                                               ; preds = %b3
  %v5 = select i1 %v0, ptr @g5, ptr @g4
  br label %b5

b5:                                               ; preds = %b5, %b4
  %v6 = phi ptr [ %v5, %b4 ], [ %v29, %b5 ]
  %v7 = phi i32 [ 0, %b4 ], [ %v27, %b5 ]
  %v8 = tail call i32 asm sideeffect "1:     $0 = memw_locked($2)\0A       $0 = add($0, $3)\0A       memw_locked($2, p0) = $0\0A       if !p0 jump 1b\0A", "=&r,=*m,r,r,*m,~{p0}"(ptr elementtype(i32) @g3, ptr @g3, i32 1, ptr elementtype(i32) @g3), !srcloc !4
  store i32 %v8, ptr %v6, align 4, !tbaa !0
  %v9 = getelementptr i32, ptr %v6, i32 1
  %v10 = tail call i32 asm sideeffect "1:     $0 = memw_locked($2)\0A       $0 = add($0, $3)\0A       memw_locked($2, p0) = $0\0A       if !p0 jump 1b\0A", "=&r,=*m,r,r,*m,~{p0}"(ptr elementtype(i32) @g3, ptr @g3, i32 1, ptr elementtype(i32) @g3), !srcloc !4
  store i32 %v10, ptr %v9, align 4, !tbaa !0
  %v11 = getelementptr i32, ptr %v6, i32 2
  %v12 = tail call i32 asm sideeffect "1:     $0 = memw_locked($2)\0A       $0 = add($0, $3)\0A       memw_locked($2, p0) = $0\0A       if !p0 jump 1b\0A", "=&r,=*m,r,r,*m,~{p0}"(ptr elementtype(i32) @g3, ptr @g3, i32 1, ptr elementtype(i32) @g3), !srcloc !4
  store i32 %v12, ptr %v11, align 4, !tbaa !0
  %v13 = getelementptr i32, ptr %v6, i32 3
  %v14 = tail call i32 asm sideeffect "1:     $0 = memw_locked($2)\0A       $0 = add($0, $3)\0A       memw_locked($2, p0) = $0\0A       if !p0 jump 1b\0A", "=&r,=*m,r,r,*m,~{p0}"(ptr elementtype(i32) @g3, ptr @g3, i32 1, ptr elementtype(i32) @g3), !srcloc !4
  store i32 %v14, ptr %v13, align 4, !tbaa !0
  %v15 = getelementptr i32, ptr %v6, i32 4
  %v16 = tail call i32 asm sideeffect "1:     $0 = memw_locked($2)\0A       $0 = add($0, $3)\0A       memw_locked($2, p0) = $0\0A       if !p0 jump 1b\0A", "=&r,=*m,r,r,*m,~{p0}"(ptr elementtype(i32) @g3, ptr @g3, i32 1, ptr elementtype(i32) @g3), !srcloc !4
  store i32 %v16, ptr %v15, align 4, !tbaa !0
  %v17 = getelementptr i32, ptr %v6, i32 5
  %v18 = tail call i32 asm sideeffect "1:     $0 = memw_locked($2)\0A       $0 = add($0, $3)\0A       memw_locked($2, p0) = $0\0A       if !p0 jump 1b\0A", "=&r,=*m,r,r,*m,~{p0}"(ptr elementtype(i32) @g3, ptr @g3, i32 1, ptr elementtype(i32) @g3), !srcloc !4
  store i32 %v18, ptr %v17, align 4, !tbaa !0
  %v19 = getelementptr i32, ptr %v6, i32 6
  %v20 = tail call i32 asm sideeffect "1:     $0 = memw_locked($2)\0A       $0 = add($0, $3)\0A       memw_locked($2, p0) = $0\0A       if !p0 jump 1b\0A", "=&r,=*m,r,r,*m,~{p0}"(ptr elementtype(i32) @g3, ptr @g3, i32 1, ptr elementtype(i32) @g3), !srcloc !4
  store i32 %v20, ptr %v19, align 4, !tbaa !0
  %v21 = getelementptr i32, ptr %v6, i32 7
  %v22 = tail call i32 asm sideeffect "1:     $0 = memw_locked($2)\0A       $0 = add($0, $3)\0A       memw_locked($2, p0) = $0\0A       if !p0 jump 1b\0A", "=&r,=*m,r,r,*m,~{p0}"(ptr elementtype(i32) @g3, ptr @g3, i32 1, ptr elementtype(i32) @g3), !srcloc !4
  store i32 %v22, ptr %v21, align 4, !tbaa !0
  %v23 = getelementptr i32, ptr %v6, i32 8
  %v24 = tail call i32 asm sideeffect "1:     $0 = memw_locked($2)\0A       $0 = add($0, $3)\0A       memw_locked($2, p0) = $0\0A       if !p0 jump 1b\0A", "=&r,=*m,r,r,*m,~{p0}"(ptr elementtype(i32) @g3, ptr @g3, i32 1, ptr elementtype(i32) @g3), !srcloc !4
  store i32 %v24, ptr %v23, align 4, !tbaa !0
  %v25 = getelementptr i32, ptr %v6, i32 9
  %v26 = tail call i32 asm sideeffect "1:     $0 = memw_locked($2)\0A       $0 = add($0, $3)\0A       memw_locked($2, p0) = $0\0A       if !p0 jump 1b\0A", "=&r,=*m,r,r,*m,~{p0}"(ptr elementtype(i32) @g3, ptr @g3, i32 1, ptr elementtype(i32) @g3), !srcloc !4
  store i32 %v26, ptr %v25, align 4, !tbaa !0
  %v27 = add nsw i32 %v7, 10
  %v28 = icmp eq i32 %v27, 100
  %v29 = getelementptr i32, ptr %v6, i32 10
  br i1 %v28, label %b6, label %b5

b6:                                               ; preds = %b5
  tail call void @f2(i32 0) #1
  ret void
}

; Function Attrs: nounwind
declare void @f2(i32) #1

; Function Attrs: nounwind
declare i32 @f3(ptr nocapture readonly) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-fatures"="+hvx,+hvx-length64b" }
attributes #1 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{i32 12730, i32 12771, i32 12807, i32 12851}
