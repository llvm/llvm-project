; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s
; RUN: llc -O0 --global-isel-abort=1 < %s -mtriple=arm64-eabi | FileCheck %s

@a = common global ptr null, align 8

define void @test(i32 %i, i32 %j) nounwind ssp {
entry:
  ; CHECK: @test
  %j.addr = alloca i32, align 4
  store i32 %j, ptr %j.addr, align 4, !tbaa !0
  ; CHECK: prfum pldl1strm
  call void @llvm.prefetch(ptr %j.addr, i32 0, i32 0, i32 1)
  ; CHECK: prfum pldl3keep
  call void @llvm.prefetch(ptr %j.addr, i32 0, i32 1, i32 1)
  ; CHECK: prfum pldl2keep
  call void @llvm.prefetch(ptr %j.addr, i32 0, i32 2, i32 1)
  ; CHECK: prfum pldl1keep
  call void @llvm.prefetch(ptr %j.addr, i32 0, i32 3, i32 1)

  ; CHECK: prfum plil1strm
  call void @llvm.prefetch(ptr %j.addr, i32 0, i32 0, i32 0)
  ; CHECK: prfum plil3keep
  call void @llvm.prefetch(ptr %j.addr, i32 0, i32 1, i32 0)
  ; CHECK: prfum plil2keep
  call void @llvm.prefetch(ptr %j.addr, i32 0, i32 2, i32 0)
  ; CHECK: prfum plil1keep
  call void @llvm.prefetch(ptr %j.addr, i32 0, i32 3, i32 0)

  ; CHECK: prfum pstl1strm
  call void @llvm.prefetch(ptr %j.addr, i32 1, i32 0, i32 1)
  ; CHECK: prfum pstl3keep
  call void @llvm.prefetch(ptr %j.addr, i32 1, i32 1, i32 1)
  ; CHECK: prfum pstl2keep
  call void @llvm.prefetch(ptr %j.addr, i32 1, i32 2, i32 1)
  ; CHECK: prfum pstl1keep
  call void @llvm.prefetch(ptr %j.addr, i32 1, i32 3, i32 1)

  %tmp1 = load i32, ptr %j.addr, align 4, !tbaa !0
  %add = add nsw i32 %tmp1, %i
  %idxprom = sext i32 %add to i64
  %tmp2 = load ptr, ptr @a, align 8, !tbaa !3
  %arrayidx = getelementptr inbounds i32, ptr %tmp2, i64 %idxprom

  ; CHECK: prfm pldl1strm
  call void @llvm.prefetch(ptr %arrayidx, i32 0, i32 0, i32 1)
  %tmp4 = load ptr, ptr @a, align 8, !tbaa !3
  %arrayidx3 = getelementptr inbounds i32, ptr %tmp4, i64 %idxprom

  ; CHECK: prfm pldl3keep
  call void @llvm.prefetch(ptr %arrayidx3, i32 0, i32 1, i32 1)
  %tmp6 = load ptr, ptr @a, align 8, !tbaa !3
  %arrayidx6 = getelementptr inbounds i32, ptr %tmp6, i64 %idxprom

  ; CHECK: prfm pldl2keep
  call void @llvm.prefetch(ptr %arrayidx6, i32 0, i32 2, i32 1)
  %tmp8 = load ptr, ptr @a, align 8, !tbaa !3
  %arrayidx9 = getelementptr inbounds i32, ptr %tmp8, i64 %idxprom

  ; CHECK: prfm pldl1keep
  call void @llvm.prefetch(ptr %arrayidx9, i32 0, i32 3, i32 1)
  %tmp10 = load ptr, ptr @a, align 8, !tbaa !3
  %arrayidx12 = getelementptr inbounds i32, ptr %tmp10, i64 %idxprom


  ; CHECK: prfm plil1strm
  call void @llvm.prefetch(ptr %arrayidx12, i32 0, i32 0, i32 0)
  %tmp12 = load ptr, ptr @a, align 8, !tbaa !3
  %arrayidx15 = getelementptr inbounds i32, ptr %tmp12, i64 %idxprom

  ; CHECK: prfm plil3keep
  call void @llvm.prefetch(ptr %arrayidx3, i32 0, i32 1, i32 0)
  %tmp14 = load ptr, ptr @a, align 8, !tbaa !3
  %arrayidx18 = getelementptr inbounds i32, ptr %tmp14, i64 %idxprom

  ; CHECK: prfm plil2keep
  call void @llvm.prefetch(ptr %arrayidx6, i32 0, i32 2, i32 0)
  %tmp16 = load ptr, ptr @a, align 8, !tbaa !3
  %arrayidx21 = getelementptr inbounds i32, ptr %tmp16, i64 %idxprom

  ; CHECK: prfm plil1keep
  call void @llvm.prefetch(ptr %arrayidx9, i32 0, i32 3, i32 0)
  %tmp18 = load ptr, ptr @a, align 8, !tbaa !3
  %arrayidx24 = getelementptr inbounds i32, ptr %tmp18, i64 %idxprom


  ; CHECK: prfm pstl1strm
  call void @llvm.prefetch(ptr %arrayidx12, i32 1, i32 0, i32 1)
  %tmp20 = load ptr, ptr @a, align 8, !tbaa !3
  %arrayidx27 = getelementptr inbounds i32, ptr %tmp20, i64 %idxprom

  ; CHECK: prfm pstl3keep
  call void @llvm.prefetch(ptr %arrayidx15, i32 1, i32 1, i32 1)
  %tmp22 = load ptr, ptr @a, align 8, !tbaa !3
  %arrayidx30 = getelementptr inbounds i32, ptr %tmp22, i64 %idxprom

  ; CHECK: prfm pstl2keep
  call void @llvm.prefetch(ptr %arrayidx18, i32 1, i32 2, i32 1)
  %tmp24 = load ptr, ptr @a, align 8, !tbaa !3
  %arrayidx33 = getelementptr inbounds i32, ptr %tmp24, i64 %idxprom

  ; CHECK: prfm pstl1keep
  call void @llvm.prefetch(ptr %arrayidx21, i32 1, i32 3, i32 1)
  ret void
}

declare void @llvm.prefetch(ptr nocapture, i32, i32, i32) nounwind

!0 = !{!"int", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!"any pointer", !1}
