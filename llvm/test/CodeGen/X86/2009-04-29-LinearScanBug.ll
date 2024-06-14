; RUN: llc < %s -mtriple=i386-apple-darwin10
; rdar://6837009

	%0 = type { ptr, ptr, ptr, i32 }
	%1 = type { %2 }
	%2 = type { %struct.pf_addr, %struct.pf_addr }
	%3 = type { %struct.in6_addr }
	%4 = type { [4 x i32] }
	%5 = type { ptr, [4 x i8] }
	%6 = type { ptr, ptr }
	%7 = type { ptr, ptr, ptr, i32 }
	%8 = type { ptr }
	%9 = type { ptr }
	%10 = type { %11 }
	%11 = type { ptr, ptr, ptr }
	%12 = type { [2 x %struct.pf_rulequeue], %13, %13 }
	%13 = type { ptr, ptr, i32, i32, i32 }
	%14 = type { ptr, ptr, ptr, i32 }
	%15 = type { ptr, ptr, ptr, i32 }
	%16 = type { ptr, ptr }
	%17 = type { %18 }
	%18 = type { %struct.pkthdr, %19 }
	%19 = type { %struct.m_ext, [176 x i8] }
	%20 = type { ptr, ptr }
	%21 = type { i32, %22 }
	%22 = type { ptr, [4 x i8] }
	%23 = type { ptr }
	%24 = type { %struct.pf_ike_state }
	%25 = type { ptr, ptr, ptr, i32 }
	%26 = type { ptr, ptr, ptr, i32 }
	%struct.anon = type { ptr, ptr }
	%struct.au_mask_t = type { i32, i32 }
	%struct.bpf_if = type opaque
	%struct.dlil_threading_info = type opaque
	%struct.ether_header = type { [6 x i8], [6 x i8], i16 }
	%struct.ext_refsq = type { ptr, ptr }
	%struct.hook_desc = type { %struct.hook_desc_head, ptr, ptr }
	%struct.hook_desc_head = type { ptr, ptr }
	%struct.if_data_internal = type { i8, i8, i8, i8, i8, i8, i8, i8, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32, %struct.au_mask_t, i32, i32, i32 }
	%struct.ifaddr = type { ptr, ptr, ptr, ptr, %struct.ifaddrhead, ptr, i32, i32, i32, ptr, ptr, i32 }
	%struct.ifaddrhead = type { ptr, ptr }
	%struct.ifmultiaddr = type { %20, ptr, ptr, ptr, i32, ptr, i32, ptr }
	%struct.ifmultihead = type { ptr }
	%struct.ifnet = type { ptr, ptr, %16, %struct.ifaddrhead, i32, ptr, i32, ptr, i16, i16, i16, i16, i32, ptr, i32, %struct.if_data_internal, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, %struct.ifnet_filter_head, i32, ptr, i32, %struct.ifmultihead, i32, ptr, ptr, ptr, ptr, ptr, ptr, %struct.ifqueue, [1 x i32], i32, %struct.ifprefixhead, ptr, %21, i32, ptr, ptr, ptr, %struct.route }
	%struct.ifnet_demux_desc = type { i32, ptr, i32 }
	%struct.ifnet_filter = type opaque
	%struct.ifnet_filter_head = type { ptr, ptr }
	%struct.ifprefix = type { ptr, ptr, %struct.ifprefixhead, i8, i8 }
	%struct.ifprefixhead = type { ptr, ptr }
	%struct.ifqueue = type { ptr, ptr, i32, i32, i32 }
	%struct.in6_addr = type { %4 }
	%struct.in_addr = type { i32 }
	%struct.kev_d_vectors = type { i32, ptr }
	%struct.kev_msg = type { i32, i32, i32, i32, [5 x %struct.kev_d_vectors] }
	%struct.lck_mtx_t = type { [3 x i32] }
	%struct.lck_rw_t = type <{ [3 x i32] }>
	%struct.m_ext = type { ptr, ptr, i32, ptr, %struct.ext_refsq, ptr }
	%struct.m_hdr = type { ptr, ptr, i32, ptr, i16, i16 }
	%struct.m_tag = type { %struct.packet_tags, i16, i16, i32 }
	%struct.mbuf = type { %struct.m_hdr, %17 }
	%struct.packet_tags = type { ptr }
	%struct.pf_addr = type { %3 }
	%struct.pf_addr_wrap = type <{ %1, %5, i8, i8, [6 x i8] }>
	%struct.pf_anchor = type { %14, %14, ptr, %struct.pf_anchor_node, [64 x i8], [1024 x i8], %struct.pf_ruleset, i32, i32 }
	%struct.pf_anchor_node = type { ptr }
	%struct.pf_app_state = type { ptr, ptr, ptr, %24 }
	%struct.pf_ike_state = type { i64 }
	%struct.pf_mtag = type { ptr, i32, i32, i16, i8, i8 }
	%struct.pf_palist = type { ptr, ptr }
	%struct.pf_pdesc = type { %struct.pf_threshold, i64, %23, %struct.pf_addr, %struct.pf_addr, ptr, ptr, ptr, ptr, ptr, i32, ptr, ptr, i32, i16, i8, i8, i8, i8 }
	%struct.pf_pool = type { %struct.pf_palist, [2 x i32], ptr, [4 x i8], %struct.in6_addr, %struct.pf_addr, i32, [2 x i16], i8, i8, [1 x i32] }
	%struct.pf_pooladdr = type <{ %struct.pf_addr_wrap, %struct.pf_palist, [2 x i32], [16 x i8], ptr, [1 x i32] }>
	%struct.pf_rule = type <{ %struct.pf_rule_addr, %struct.pf_rule_addr, [8 x %struct.pf_rule_ptr], [64 x i8], [16 x i8], [64 x i8], [64 x i8], [64 x i8], [64 x i8], [32 x i8], %struct.pf_rulequeue, [2 x i32], %struct.pf_pool, i64, [2 x i64], [2 x i64], ptr, [4 x i8], ptr, [4 x i8], ptr, [4 x i8], i32, i32, [26 x i32], i32, i32, i32, i32, i32, i32, %struct.au_mask_t, i32, i32, i32, i32, i32, i32, i32, i16, i16, i16, i16, i16, [2 x i8], %struct.pf_rule_gid, %struct.pf_rule_gid, i32, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, [2 x i8] }>
	%struct.pf_rule_addr = type <{ %struct.pf_addr_wrap, %struct.pf_rule_xport, i8, [7 x i8] }>
	%struct.pf_rule_gid = type { [2 x i32], i8, [3 x i8] }
	%struct.pf_rule_ptr = type { ptr, [4 x i8] }
	%struct.pf_rule_xport = type { i32, [4 x i8] }
	%struct.pf_rulequeue = type { ptr, ptr }
	%struct.pf_ruleset = type { [5 x %12], ptr, i32, i32, i32 }
	%struct.pf_src_node = type <{ %26, %struct.pf_addr, %struct.pf_addr, %struct.pf_rule_ptr, ptr, [2 x i64], [2 x i64], i32, i32, %struct.pf_threshold, i64, i64, i8, i8, [2 x i8] }>
	%struct.pf_state = type <{ i64, i32, i32, %struct.anon, %struct.anon, %0, %struct.pf_state_peer, %struct.pf_state_peer, %struct.pf_rule_ptr, %struct.pf_rule_ptr, %struct.pf_rule_ptr, %struct.pf_addr, %struct.hook_desc_head, ptr, ptr, ptr, ptr, ptr, [2 x i64], [2 x i64], i64, i64, i64, i16, i8, i8, i8, i8, [6 x i8] }>
	%struct.pf_state_host = type { %struct.pf_addr, %struct.in_addr }
	%struct.pf_state_key = type { %struct.pf_state_host, %struct.pf_state_host, %struct.pf_state_host, i8, i8, i8, i8, ptr, %25, %25, %struct.anon, i16 }
	%struct.pf_state_peer = type { i32, i32, i32, i16, i8, i8, i16, i8, ptr, [3 x i8] }
	%struct.pf_state_scrub = type { %struct.au_mask_t, i32, i32, i32, i16, i8, i8, i32 }
	%struct.pf_threshold = type { i32, i32, i32, i32 }
	%struct.pfi_dynaddr = type { %6, %struct.pf_addr, %struct.pf_addr, %struct.pf_addr, %struct.pf_addr, ptr, ptr, ptr, i32, i32, i32, i8, i8 }
	%struct.pfi_kif = type { [16 x i8], %15, [2 x [2 x [2 x i64]]], [2 x [2 x [2 x i64]]], i64, i32, ptr, ptr, i32, i32, %6 }
	%struct.pfr_ktable = type { %struct.pfr_tstats, %7, %8, ptr, ptr, ptr, ptr, ptr, i64, i32 }
	%struct.pfr_table = type { [1024 x i8], [32 x i8], i32, i8 }
	%struct.pfr_tstats = type { %struct.pfr_table, [2 x [3 x i64]], [2 x [3 x i64]], i64, i64, i64, i32, [2 x i32] }
	%struct.pkthdr = type { i32, ptr, ptr, i32, i32, i32, i16, i16, %struct.packet_tags }
	%struct.proto_hash_entry = type opaque
	%struct.radix_mask = type { i16, i8, i8, ptr, %9, i32 }
	%struct.radix_node = type { ptr, ptr, i16, i8, i8, %10 }
	%struct.radix_node_head = type { ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, [3 x %struct.radix_node], i32 }
	%struct.route = type { ptr, i32, %struct.sockaddr }
	%struct.rt_metrics = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [4 x i32] }
	%struct.rtentry = type { [2 x %struct.radix_node], ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, %struct.rt_metrics, ptr, ptr, i32, %struct.lck_mtx_t }
	%struct.sockaddr = type { i8, i8, [14 x i8] }
	%struct.tcphdr = type { i16, i16, i32, i32, i8, i8, i16, i16, i16 }
	%struct.thread = type opaque
@llvm.used = appending global [1 x ptr] [ptr @pf_state_compare_ext_gwy], section "llvm.metadata"		; <ptr> [#uses=0]

define fastcc i32 @pf_state_compare_ext_gwy(ptr nocapture %a, ptr nocapture %b) nounwind optsize ssp {
entry:
	%0 = zext i8 0 to i32		; <i32> [#uses=2]
	%1 = load i8, ptr null, align 1		; <i8> [#uses=2]
	%2 = zext i8 %1 to i32		; <i32> [#uses=1]
	%3 = sub i32 %0, %2		; <i32> [#uses=1]
	%4 = icmp eq i8 0, %1		; <i1> [#uses=1]
	br i1 %4, label %bb1, label %bb79

bb1:		; preds = %entry
	%5 = load i8, ptr null, align 4		; <i8> [#uses=2]
	%6 = zext i8 %5 to i32		; <i32> [#uses=2]
	%7 = getelementptr %struct.pf_state_key, ptr %b, i32 0, i32 3		; <ptr> [#uses=1]
	%8 = load i8, ptr %7, align 4		; <i8> [#uses=2]
	%9 = zext i8 %8 to i32		; <i32> [#uses=1]
	%10 = sub i32 %6, %9		; <i32> [#uses=1]
	%11 = icmp eq i8 %5, %8		; <i1> [#uses=1]
	br i1 %11, label %bb3, label %bb79

bb3:		; preds = %bb1
	switch i32 %0, label %bb23 [
		i32 1, label %bb4
		i32 6, label %bb6
		i32 17, label %bb10
		i32 47, label %bb17
		i32 50, label %bb21
		i32 58, label %bb4
	]

bb4:		; preds = %bb3, %bb3
	%12 = load i16, ptr null, align 4		; <i16> [#uses=1]
	%13 = zext i16 %12 to i32		; <i32> [#uses=1]
	%14 = sub i32 0, %13		; <i32> [#uses=1]
	br i1 false, label %bb23, label %bb79

bb6:		; preds = %bb3
	%15 = load i16, ptr null, align 4		; <i16> [#uses=1]
	%16 = zext i16 %15 to i32		; <i32> [#uses=1]
	%17 = sub i32 0, %16		; <i32> [#uses=1]
	ret i32 %17

bb10:		; preds = %bb3
	%18 = load i8, ptr null, align 1		; <i8> [#uses=2]
	%19 = zext i8 %18 to i32		; <i32> [#uses=1]
	%20 = sub i32 0, %19		; <i32> [#uses=1]
	%21 = icmp eq i8 0, %18		; <i1> [#uses=1]
	br i1 %21, label %bb12, label %bb79

bb12:		; preds = %bb10
	%22 = load i16, ptr null, align 4		; <i16> [#uses=1]
	%23 = zext i16 %22 to i32		; <i32> [#uses=1]
	%24 = sub i32 0, %23		; <i32> [#uses=1]
	ret i32 %24

bb17:		; preds = %bb3
	%25 = load i8, ptr null, align 1		; <i8> [#uses=2]
	%26 = icmp eq i8 %25, 1		; <i1> [#uses=1]
	br i1 %26, label %bb18, label %bb23

bb18:		; preds = %bb17
	%27 = icmp eq i8 %25, 0		; <i1> [#uses=1]
	br i1 %27, label %bb19, label %bb23

bb19:		; preds = %bb18
	%28 = load i16, ptr null, align 4		; <i16> [#uses=1]
	%29 = zext i16 %28 to i32		; <i32> [#uses=1]
	%30 = sub i32 0, %29		; <i32> [#uses=1]
	br i1 false, label %bb23, label %bb79

bb21:		; preds = %bb3
	%31 = getelementptr %struct.pf_state_key, ptr %a, i32 0, i32 1, i32 1, i32 0		; <ptr> [#uses=1]
	%32 = load i32, ptr %31, align 4		; <i32> [#uses=2]
	%33 = getelementptr %struct.pf_state_key, ptr %b, i32 0, i32 1, i32 1, i32 0		; <ptr> [#uses=1]
	%34 = load i32, ptr %33, align 4		; <i32> [#uses=2]
	%35 = sub i32 %32, %34		; <i32> [#uses=1]
	%36 = icmp eq i32 %32, %34		; <i1> [#uses=1]
	br i1 %36, label %bb23, label %bb79

bb23:		; preds = %bb21, %bb19, %bb18, %bb17, %bb4, %bb3
	%cond = icmp eq i32 %6, 2		; <i1> [#uses=1]
	br i1 %cond, label %bb24, label %bb70

bb24:		; preds = %bb23
	ret i32 1

bb70:		; preds = %bb23
	%37 = load ptr, ptr null, align 4		; <ptr> [#uses=3]
	br i1 false, label %bb78, label %bb73

bb73:		; preds = %bb70
	%38 = load ptr, ptr null, align 4		; <ptr> [#uses=2]
	%39 = icmp eq ptr %38, null		; <i1> [#uses=1]
	br i1 %39, label %bb78, label %bb74

bb74:		; preds = %bb73
	%40 = ptrtoint ptr %37 to i32		; <i32> [#uses=1]
	%41 = sub i32 0, %40		; <i32> [#uses=1]
	%42 = icmp eq ptr %38, %37		; <i1> [#uses=1]
	br i1 %42, label %bb76, label %bb79

bb76:		; preds = %bb74
	%43 = tail call i32 %37(ptr null, ptr null) nounwind		; <i32> [#uses=1]
	ret i32 %43

bb78:		; preds = %bb73, %bb70
	ret i32 0

bb79:		; preds = %bb74, %bb21, %bb19, %bb10, %bb4, %bb1, %entry
	%.0 = phi i32 [ %3, %entry ], [ %10, %bb1 ], [ %14, %bb4 ], [ %20, %bb10 ], [ %30, %bb19 ], [ %35, %bb21 ], [ %41, %bb74 ]		; <i32> [#uses=1]
	ret i32 %.0
}
