; RUN: llc -mtriple=aarch64 -mattr=+v8.9a --global-isel=0 < %s | FileCheck %s
; RUN: llc -mtriple=aarch64 -mattr=+v8.9a --global-isel=1 --global-isel-abort=1 < %s | FileCheck %s

@a = internal global ptr null, align 8
@b = external global ptr, align 8

define void @test(ptr %i, i32 %j) nounwind ssp {
entry:
  ; CHECK-LABEL: @test
  %j.addr = alloca i32, align 4
  store i32 %j, ptr %j.addr, align 4, !tbaa !0
  %tmp = bitcast ptr %j.addr to ptr

  %i.next = getelementptr i8, ptr %i, i64 2

  ; Verify prefetching works for all the different kinds of pointers we might
  ; want to prefetch.

  ; CHECK: prfm pldl1keep,
  call void @llvm.aarch64.prefetch(ptr null, i32 0, i32 0, i32 0, i32 1)

  ; CHECK: prfum pldl1keep,
  call void @llvm.aarch64.prefetch(ptr %tmp, i32 0, i32 0, i32 0, i32 1)

  ; CHECK: prfm pldl1keep,
  call void @llvm.aarch64.prefetch(ptr %i, i32 0, i32 0, i32 0, i32 1)

  ; CHECK: prfum pldl1keep,
  call void @llvm.aarch64.prefetch(ptr %i.next, i32 0, i32 0, i32 0, i32 1)

  ; CHECK: prfm pldl1keep,
  call void @llvm.aarch64.prefetch(ptr @a, i32 0, i32 0, i32 0, i32 1)

  ; CHECK: prfm pldl1keep,
  call void @llvm.aarch64.prefetch(ptr @b, i32 0, i32 0, i32 0, i32 1)

  ; Verify that we can generate every single valid prefetch value.

  ; CHECK: prfm pstl1keep,
  call void @llvm.aarch64.prefetch(ptr null, i32 1, i32 0, i32 0, i32 1)

  ; CHECK: prfm pldl2keep,
  call void @llvm.aarch64.prefetch(ptr null, i32 0, i32 1, i32 0, i32 1)

  ; CHECK: prfm pldl3keep,
  call void @llvm.aarch64.prefetch(ptr null, i32 0, i32 2, i32 0, i32 1)

  ; CHECK: prfm pldslckeep,
  call void @llvm.aarch64.prefetch(ptr null, i32 0, i32 3, i32 0, i32 1)

  ; CHECK: prfm pldl1strm,
  call void @llvm.aarch64.prefetch(ptr null, i32 0, i32 0, i32 1, i32 1)

  ; CHECK: prfm plil1keep,
  call void @llvm.aarch64.prefetch(ptr null, i32 0, i32 0, i32 0, i32 0)

  ret void
}

declare void @llvm.aarch64.prefetch(ptr readonly, i32 immarg, i32 immarg, i32 immarg, i32 immarg) #0

attributes #0 = { inaccessiblemem_or_argmemonly nounwind willreturn }

!0 = !{!"int", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!"any pointer", !1}
