; RUN: llc < %s -march=bpfel | FileCheck -check-prefixes=CHECK,EL %s
; RUN: llc < %s -march=bpfeb | FileCheck -check-prefixes=CHECK,EB %s

%struct.bpf_map_def = type { i32, i32, i32, i32 }
%struct.__sk_buff = type opaque
%struct.routing_key_2 = type { [6 x i8] }

@routing = global %struct.bpf_map_def { i32 1, i32 6, i32 12, i32 1024 }, section "maps", align 4
@routing_miss_0 = global %struct.bpf_map_def { i32 1, i32 1, i32 12, i32 1 }, section "maps", align 4
@test1 = global %struct.bpf_map_def { i32 2, i32 4, i32 8, i32 1024 }, section "maps", align 4
@test1_miss_4 = global %struct.bpf_map_def { i32 2, i32 1, i32 8, i32 1 }, section "maps", align 4
@_license = global [4 x i8] c"GPL\00", section "license", align 1
@llvm.used = appending global [6 x ptr] [ptr @_license, ptr @ebpf_filter, ptr @routing, ptr @routing_miss_0, ptr @test1, ptr @test1_miss_4], section "llvm.metadata"

; Function Attrs: nounwind uwtable
define i32 @ebpf_filter(ptr nocapture readnone %ebpf_packet) #0 section "socket1" {

; EL: r1 = 11033905661445 ll
; EB: r1 = 361984551142686720 ll
; CHECK: *(u64 *)(r10 - 8) = r1

; CHECK: r1 = 0
; CHECK-DAG: *(u16 *)(r10 + 24) = r1
; CHECK-DAG: *(u16 *)(r10 + 22) = r1
; CHECK-DAG: *(u16 *)(r10 + 20) = r1
; CHECK-DAG: *(u16 *)(r10 + 18) = r1
; CHECK-DAG: *(u16 *)(r10 + 16) = r1
; CHECK-DAG: *(u16 *)(r10 + 14) = r1
; CHECK-DAG: *(u16 *)(r10 + 12) = r1
; CHECK-DAG: *(u16 *)(r10 + 10) = r1
; CHECK-DAG: *(u16 *)(r10 + 8) = r1
; CHECK-DAG: *(u16 *)(r10 + 6) = r1
; CHECK-DAG: *(u16 *)(r10 + 4) = r1
; CHECK-DAG: *(u16 *)(r10 + 2) = r1
; CHECK-DAG: *(u16 *)(r10 + 0) = r1
; CHECK-DAG: *(u16 *)(r10 + 26) = r1

; CHECK: r2 = r10
; CHECK: r2 += -8
; CHECK: r1 = routing
; CHECK: call bpf_map_lookup_elem
; CHECK: exit
  %key = alloca %struct.routing_key_2, align 1
  store i8 5, ptr %key, align 1
  %1 = getelementptr inbounds %struct.routing_key_2, ptr %key, i64 0, i32 0, i64 1
  store i8 6, ptr %1, align 1
  %2 = getelementptr inbounds %struct.routing_key_2, ptr %key, i64 0, i32 0, i64 2
  store i8 7, ptr %2, align 1
  %3 = getelementptr inbounds %struct.routing_key_2, ptr %key, i64 0, i32 0, i64 3
  store i8 8, ptr %3, align 1
  %4 = getelementptr inbounds %struct.routing_key_2, ptr %key, i64 0, i32 0, i64 4
  store i8 9, ptr %4, align 1
  %5 = getelementptr inbounds %struct.routing_key_2, ptr %key, i64 0, i32 0, i64 5
  store i8 10, ptr %5, align 1
  %6 = getelementptr inbounds %struct.routing_key_2, ptr %key, i64 1, i32 0, i64 0
  call void @llvm.memset.p0.i64(ptr %6, i8 0, i64 30, i1 false)
  %7 = call i32 (ptr, ptr, ...) @bpf_map_lookup_elem(ptr nonnull @routing, ptr nonnull %key) #3
  ret i32 undef
}

; Function Attrs: nounwind argmemonly
declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) #1

declare i32 @bpf_map_lookup_elem(...) #2
