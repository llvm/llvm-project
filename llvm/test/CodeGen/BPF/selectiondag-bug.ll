; RUN: llc -march=bpf < %s | FileCheck %s
;
; The IR is generated from a bpftrace script (https://github.com/iovisor/bpftrace/issues/1305)
; and then slightly adapted for easy unit testing.
; The llvm bugzilla link: https://bugs.llvm.org/show_bug.cgi?id=47591

%printf_t = type { i64, i64 }

define i64 @"kprobe:blk_update_request"(i8* %0) local_unnamed_addr section "s_kprobe:blk_update_request_1" {
entry:
  %"struct kernfs_node.parent" = alloca i64, align 8
  %printf_args = alloca %printf_t, align 8
  %"struct cgroup.kn" = alloca i64, align 8
  %"struct cgroup_subsys_state.cgroup" = alloca i64, align 8
  %"struct blkcg_gq.blkcg" = alloca i64, align 8
  %"struct bio.bi_blkg" = alloca i64, align 8
  %"struct request.bio" = alloca i64, align 8
  %1 = getelementptr i8, i8* %0, i64 112
  %2 = bitcast i8* %1 to i64*
  %arg0 = load volatile i64, i64* %2, align 8
  %3 = add i64 %arg0, 56
  %4 = bitcast i64* %"struct request.bio" to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* nonnull %4)
  %probe_read = call i64 inttoptr (i64 4 to i64 (i64*, i32, i64)*)(i64* nonnull %"struct request.bio", i32 8, i64 %3)
  %5 = load i64, i64* %"struct request.bio", align 8
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* nonnull %4)
  %6 = add i64 %5, 72
  %7 = bitcast i64* %"struct bio.bi_blkg" to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* nonnull %7)
  %probe_read1 = call i64 inttoptr (i64 5 to i64 (i64*, i32, i64)*)(i64* nonnull %"struct bio.bi_blkg", i32 8, i64 %6)
  %8 = load i64, i64* %"struct bio.bi_blkg", align 8
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* nonnull %7)
  %9 = add i64 %8, 40
  %10 = bitcast i64* %"struct blkcg_gq.blkcg" to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* nonnull %10)
  %probe_read2 = call i64 inttoptr (i64 6 to i64 (i64*, i32, i64)*)(i64* nonnull %"struct blkcg_gq.blkcg", i32 8, i64 %9)
  %11 = load i64, i64* %"struct blkcg_gq.blkcg", align 8
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* nonnull %10)
  %12 = bitcast i64* %"struct cgroup_subsys_state.cgroup" to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* nonnull %12)
  %probe_read3 = call i64 inttoptr (i64 7 to i64 (i64*, i32, i64)*)(i64* nonnull %"struct cgroup_subsys_state.cgroup", i32 8, i64 %11)
  %13 = load i64, i64* %"struct cgroup_subsys_state.cgroup", align 8
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* nonnull %12)
  %14 = add i64 %13, 288
  %15 = bitcast i64* %"struct cgroup.kn" to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* nonnull %15)
  %probe_read4 = call i64 inttoptr (i64 8 to i64 (i64*, i32, i64)*)(i64* nonnull %"struct cgroup.kn", i32 8, i64 %14)
  %16 = load i64, i64* %"struct cgroup.kn", align 8
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* nonnull %15)
  %17 = bitcast %printf_t* %printf_args to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* nonnull %17)
  %18 = add i64 %16, 8
  %19 = bitcast i64* %"struct kernfs_node.parent" to i8*
  %20 = getelementptr inbounds %printf_t, %printf_t* %printf_args, i64 0, i32 0
  store i64 0, i64* %20, align 8
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* nonnull %19)

; CHECK:        call 8
; CHECK-NOT:    r{{[0-9]+}} = 0
; CHECK:        [[REG3:r[0-9]+]] = *(u64 *)(r10 - 24)
; CHECK:        [[REG1:r[0-9]+]] = 0
; CHECK:        *(u64 *)(r10 - 24) = [[REG1]]

  %probe_read5 = call i64 inttoptr (i64 9 to i64 (i64*, i32, i64)*)(i64* nonnull %"struct kernfs_node.parent", i32 8, i64 %18)
  %21 = load i64, i64* %"struct kernfs_node.parent", align 8
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* nonnull %19)
  %22 = getelementptr inbounds %printf_t, %printf_t* %printf_args, i64 0, i32 1
  store i64 %21, i64* %22, align 8
  %get_cpu_id = call i64 inttoptr (i64 18 to i64 ()*)()
  %perf_event_output = call i64 inttoptr (i64 10 to i64 (i8*, i64, i64, %printf_t*, i64)*)(i8* %0, i64 2, i64 %get_cpu_id, %printf_t* nonnull %printf_args, i64 16)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* nonnull %17)
  ret i64 0
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg %0, i8* nocapture %1) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg %0, i8* nocapture %1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind willreturn }
