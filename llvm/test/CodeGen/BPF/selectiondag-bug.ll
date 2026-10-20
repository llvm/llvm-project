; RUN: llc -mtriple=bpf < %s | FileCheck %s
;
; The IR is generated from a bpftrace script (https://github.com/iovisor/bpftrace/issues/1305)
; and then slightly adapted for easy unit testing.
; The llvm bugzilla link: https://bugs.llvm.org/show_bug.cgi?id=47591

%printf_t = type { i64, i64 }

define i64 @"kprobe:blk_update_request"(ptr %0) local_unnamed_addr section "s_kprobe:blk_update_request_1" {
entry:
  %"struct kernfs_node.parent" = alloca i64, align 8
  %printf_args = alloca %printf_t, align 8
  %"struct cgroup.kn" = alloca i64, align 8
  %"struct cgroup_subsys_state.cgroup" = alloca i64, align 8
  %"struct blkcg_gq.blkcg" = alloca i64, align 8
  %"struct bio.bi_blkg" = alloca i64, align 8
  %"struct request.bio" = alloca i64, align 8
  %1 = getelementptr i8, ptr %0, i64 112
  %arg0 = load volatile i64, ptr %1, align 8
  %2 = add i64 %arg0, 56
  %3 = bitcast ptr %"struct request.bio" to ptr
  call void @llvm.lifetime.start.p0(i64 -1, ptr nonnull %3)
  %probe_read = call i64 inttoptr (i64 4 to ptr)(ptr nonnull %"struct request.bio", i32 8, i64 %2)
  %4 = load i64, ptr %"struct request.bio", align 8
  call void @llvm.lifetime.end.p0(i64 -1, ptr nonnull %3)
  %5 = add i64 %4, 72
  %6 = bitcast ptr %"struct bio.bi_blkg" to ptr
  call void @llvm.lifetime.start.p0(i64 -1, ptr nonnull %6)
  %probe_read1 = call i64 inttoptr (i64 5 to ptr)(ptr nonnull %"struct bio.bi_blkg", i32 8, i64 %5)
  %7 = load i64, ptr %"struct bio.bi_blkg", align 8
  call void @llvm.lifetime.end.p0(i64 -1, ptr nonnull %6)
  %8 = add i64 %7, 40
  %9 = bitcast ptr %"struct blkcg_gq.blkcg" to ptr
  call void @llvm.lifetime.start.p0(i64 -1, ptr nonnull %9)
  %probe_read2 = call i64 inttoptr (i64 6 to ptr)(ptr nonnull %"struct blkcg_gq.blkcg", i32 8, i64 %8)
  %10 = load i64, ptr %"struct blkcg_gq.blkcg", align 8
  call void @llvm.lifetime.end.p0(i64 -1, ptr nonnull %9)
  %11 = bitcast ptr %"struct cgroup_subsys_state.cgroup" to ptr
  call void @llvm.lifetime.start.p0(i64 -1, ptr nonnull %11)
  %probe_read3 = call i64 inttoptr (i64 7 to ptr)(ptr nonnull %"struct cgroup_subsys_state.cgroup", i32 8, i64 %10)
  %12 = load i64, ptr %"struct cgroup_subsys_state.cgroup", align 8
  call void @llvm.lifetime.end.p0(i64 -1, ptr nonnull %11)
  %13 = add i64 %12, 288
  %14 = bitcast ptr %"struct cgroup.kn" to ptr
  call void @llvm.lifetime.start.p0(i64 -1, ptr nonnull %14)
  %probe_read4 = call i64 inttoptr (i64 8 to ptr)(ptr nonnull %"struct cgroup.kn", i32 8, i64 %13)
  %15 = load i64, ptr %"struct cgroup.kn", align 8
  call void @llvm.lifetime.end.p0(i64 -1, ptr nonnull %14)
  call void @llvm.lifetime.start.p0(i64 -1, ptr nonnull %printf_args)
  %16 = add i64 %15, 8
  %17 = bitcast ptr %"struct kernfs_node.parent" to ptr
  store i64 0, ptr %printf_args, align 8
  call void @llvm.lifetime.start.p0(i64 -1, ptr nonnull %17)

; CHECK:        call 8
; CHECK-NOT:    r{{[0-9]+}} = 0
; CHECK:        [[REG3:r[0-9]+]] = *(u64 *)(r10 - 24)
; CHECK:        [[REG1:r[0-9]+]] = 0
; CHECK:        *(u64 *)(r10 - 24) = [[REG1]]

  %probe_read5 = call i64 inttoptr (i64 9 to ptr)(ptr nonnull %"struct kernfs_node.parent", i32 8, i64 %16)
  %18 = load i64, ptr %"struct kernfs_node.parent", align 8
  call void @llvm.lifetime.end.p0(i64 -1, ptr nonnull %17)
  %19 = getelementptr inbounds %printf_t, ptr %printf_args, i64 0, i32 1
  store i64 %18, ptr %19, align 8
  %get_cpu_id = call i64 inttoptr (i64 18 to ptr)()
  %perf_event_output = call i64 inttoptr (i64 10 to ptr)(ptr %0, i64 2, i64 %get_cpu_id, ptr nonnull %printf_args, i64 16)
  call void @llvm.lifetime.end.p0(i64 -1, ptr nonnull %printf_args)
  ret i64 0
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg %0, ptr nocapture %1) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg %0, ptr nocapture %1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind willreturn }
