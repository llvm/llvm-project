; RUN: llc < %s -march=bpf | FileCheck %s

%struct.bpf_map_def = type { i32, i32, i32, i32 }
%struct.__sk_buff = type opaque
%struct.routing_key_2 = type { [6 x i8] }

@routing = global %struct.bpf_map_def { i32 1, i32 6, i32 12, i32 1024 }, section "maps", align 4
@routing_miss_0 = global %struct.bpf_map_def { i32 1, i32 1, i32 12, i32 1 }, section "maps", align 4
@test1 = global %struct.bpf_map_def { i32 2, i32 4, i32 8, i32 1024 }, section "maps", align 4
@test1_miss_4 = global %struct.bpf_map_def { i32 2, i32 1, i32 8, i32 1 }, section "maps", align 4
@_license = global [4 x i8] c"GPL\00", section "license", align 1
@llvm.used = appending global [6 x i8*] [i8* getelementptr inbounds ([4 x i8], [4 x i8]* @_license, i32 0, i32 0), i8* bitcast (i32 (%struct.__sk_buff*)* @ebpf_filter to i8*), i8* bitcast (%struct.bpf_map_def* @routing to i8*), i8* bitcast (%struct.bpf_map_def* @routing_miss_0 to i8*), i8* bitcast (%struct.bpf_map_def* @test1 to i8*), i8* bitcast (%struct.bpf_map_def* @test1_miss_4 to i8*)], section "llvm.metadata"

; Function Attrs: nounwind uwtable
define i32 @ebpf_filter(%struct.__sk_buff* nocapture readnone %ebpf_packet) #0 section "socket1" {
  %key = alloca %struct.routing_key_2, align 1
  %1 = getelementptr inbounds %struct.routing_key_2, %struct.routing_key_2* %key, i64 0, i32 0, i64 0
; CHECK: mov	r1, 5
; CHECK: stb	-8(r10), r1
  store i8 5, i8* %1, align 1
  %2 = getelementptr inbounds %struct.routing_key_2, %struct.routing_key_2* %key, i64 0, i32 0, i64 1
; CHECK: mov	r1, 6
; CHECK: stb	-7(r10), r1
  store i8 6, i8* %2, align 1
  %3 = getelementptr inbounds %struct.routing_key_2, %struct.routing_key_2* %key, i64 0, i32 0, i64 2
; CHECK: mov	r1, 7
; CHECK: stb	-6(r10), r1
  store i8 7, i8* %3, align 1
  %4 = getelementptr inbounds %struct.routing_key_2, %struct.routing_key_2* %key, i64 0, i32 0, i64 3
; CHECK: mov	r1, 8
; CHECK: stb	-5(r10), r1
  store i8 8, i8* %4, align 1
  %5 = getelementptr inbounds %struct.routing_key_2, %struct.routing_key_2* %key, i64 0, i32 0, i64 4
; CHECK: mov	r1, 9
; CHECK: stb	-4(r10), r1
  store i8 9, i8* %5, align 1
  %6 = getelementptr inbounds %struct.routing_key_2, %struct.routing_key_2* %key, i64 0, i32 0, i64 5
; CHECK: mov	r1, 10
; CHECK: stb	-3(r10), r1
  store i8 10, i8* %6, align 1
  %7 = getelementptr inbounds %struct.routing_key_2, %struct.routing_key_2* %key, i64 1, i32 0, i64 0
; CHECK: mov	r1, r10
; CHECK: addi	r1, -2
; CHECK: mov	r2, 0
; CHECK: sth	6(r1), r2
; CHECK: sth	4(r1), r2
; CHECK: sth	2(r1), r2
; CHECK: sth	24(r10), r2
; CHECK: sth	22(r10), r2
; CHECK: sth	20(r10), r2
; CHECK: sth	18(r10), r2
; CHECK: sth	16(r10), r2
; CHECK: sth	14(r10), r2
; CHECK: sth	12(r10), r2
; CHECK: sth	10(r10), r2
; CHECK: sth	8(r10), r2
; CHECK: sth	6(r10), r2
; CHECK: sth	-2(r10), r2
; CHECK: sth	26(r10), r2
  call void @llvm.memset.p0i8.i64(i8* %7, i8 0, i64 30, i32 1, i1 false)
  %8 = call i32 (%struct.bpf_map_def*, %struct.routing_key_2*, ...) bitcast (i32 (...)* @bpf_map_lookup_elem to i32 (%struct.bpf_map_def*, %struct.routing_key_2*, ...)*)(%struct.bpf_map_def* nonnull @routing, %struct.routing_key_2* nonnull %key) #3
  ret i32 undef
}

; Function Attrs: nounwind argmemonly
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) #1

declare i32 @bpf_map_lookup_elem(...) #2
