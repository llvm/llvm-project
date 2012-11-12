; RUN: llc < %s -mtriple=i386-apple-darwin10
; rdar://6837009

	%0 = type { %struct.pf_state*, %struct.pf_state*, %struct.pf_state*, i32 }
	%1 = type { %2 }
	%2 = type { %struct.pf_addr, %struct.pf_addr }
	%3 = type { %struct.in6_addr }
	%4 = type { [4 x i32] }
	%5 = type { %struct.pfi_dynaddr*, [4 x i8] }
	%6 = type { %struct.pfi_dynaddr*, %struct.pfi_dynaddr** }
	%7 = type { %struct.pfr_ktable*, %struct.pfr_ktable*, %struct.pfr_ktable*, i32 }
	%8 = type { %struct.pfr_ktable* }
	%9 = type { i8* }
	%10 = type { %11 }
	%11 = type { i8*, i8*, %struct.radix_node* }
	%12 = type { [2 x %struct.pf_rulequeue], %13, %13 }
	%13 = type { %struct.pf_rulequeue*, %struct.pf_rule**, i32, i32, i32 }
	%14 = type { %struct.pf_anchor*, %struct.pf_anchor*, %struct.pf_anchor*, i32 }
	%15 = type { %struct.pfi_kif*, %struct.pfi_kif*, %struct.pfi_kif*, i32 }
	%16 = type { %struct.ifnet*, %struct.ifnet** }
	%17 = type { %18 }
	%18 = type { %struct.pkthdr, %19 }
	%19 = type { %struct.m_ext, [176 x i8] }
	%20 = type { %struct.ifmultiaddr*, %struct.ifmultiaddr** }
	%21 = type { i32, %22 }
	%22 = type { i8*, [4 x i8] }
	%23 = type { %struct.tcphdr* }
	%24 = type { %struct.pf_ike_state }
	%25 = type { %struct.pf_state_key*, %struct.pf_state_key*, %struct.pf_state_key*, i32 }
	%26 = type { %struct.pf_src_node*, %struct.pf_src_node*, %struct.pf_src_node*, i32 }
	%struct.anon = type { %struct.pf_state*, %struct.pf_state** }
	%struct.au_mask_t = type { i32, i32 }
	%struct.bpf_if = type opaque
	%struct.dlil_threading_info = type opaque
	%struct.ether_header = type { [6 x i8], [6 x i8], i16 }
	%struct.ext_refsq = type { %struct.ext_refsq*, %struct.ext_refsq* }
	%struct.hook_desc = type { %struct.hook_desc_head, void (i8*)*, i8* }
	%struct.hook_desc_head = type { %struct.hook_desc*, %struct.hook_desc** }
	%struct.if_data_internal = type { i8, i8, i8, i8, i8, i8, i8, i8, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32, %struct.au_mask_t, i32, i32, i32 }
	%struct.ifaddr = type { %struct.sockaddr*, %struct.sockaddr*, %struct.sockaddr*, %struct.ifnet*, %struct.ifaddrhead, void (i32, %struct.rtentry*, %struct.sockaddr*)*, i32, i32, i32, void (%struct.ifaddr*)*, void (%struct.ifaddr*, i32)*, i32 }
	%struct.ifaddrhead = type { %struct.ifaddr*, %struct.ifaddr** }
	%struct.ifmultiaddr = type { %20, %struct.sockaddr*, %struct.ifmultiaddr*, %struct.ifnet*, i32, i8*, i32, void (i8*)* }
	%struct.ifmultihead = type { %struct.ifmultiaddr* }
	%struct.ifnet = type { i8*, i8*, %16, %struct.ifaddrhead, i32, i32 (%struct.ifnet*, %struct.sockaddr*)*, i32, %struct.bpf_if*, i16, i16, i16, i16, i32, i8*, i32, %struct.if_data_internal, i32, i32 (%struct.ifnet*, %struct.mbuf*)*, i32 (%struct.ifnet*, i32, i8*)*, i32 (%struct.ifnet*, i32, i32 (%struct.ifnet*, %struct.mbuf*)*)*, void (%struct.ifnet*)*, i32 (%struct.ifnet*, %struct.mbuf*, i8*, i32*)*, void (%struct.ifnet*, %struct.kev_msg*)*, i32 (%struct.ifnet*, %struct.mbuf**, %struct.sockaddr*, i8*, i8*)*, i32, %struct.ifnet_filter_head, i32, i8*, i32, %struct.ifmultihead, i32, i32 (%struct.ifnet*, i32, %struct.ifnet_demux_desc*, i32)*, i32 (%struct.ifnet*, i32)*, %struct.proto_hash_entry*, i8*, %struct.dlil_threading_info*, i8*, %struct.ifqueue, [1 x i32], i32, %struct.ifprefixhead, %struct.lck_rw_t*, %21, i32, %struct.thread*, %struct.pfi_kif*, %struct.lck_mtx_t*, %struct.route }
	%struct.ifnet_demux_desc = type { i32, i8*, i32 }
	%struct.ifnet_filter = type opaque
	%struct.ifnet_filter_head = type { %struct.ifnet_filter*, %struct.ifnet_filter** }
	%struct.ifprefix = type { %struct.sockaddr*, %struct.ifnet*, %struct.ifprefixhead, i8, i8 }
	%struct.ifprefixhead = type { %struct.ifprefix*, %struct.ifprefix** }
	%struct.ifqueue = type { i8*, i8*, i32, i32, i32 }
	%struct.in6_addr = type { %4 }
	%struct.in_addr = type { i32 }
	%struct.kev_d_vectors = type { i32, i8* }
	%struct.kev_msg = type { i32, i32, i32, i32, [5 x %struct.kev_d_vectors] }
	%struct.lck_mtx_t = type { [3 x i32] }
	%struct.lck_rw_t = type <{ [3 x i32] }>
	%struct.m_ext = type { i8*, void (i8*, i32, i8*)*, i32, i8*, %struct.ext_refsq, %struct.au_mask_t* }
	%struct.m_hdr = type { %struct.mbuf*, %struct.mbuf*, i32, i8*, i16, i16 }
	%struct.m_tag = type { %struct.packet_tags, i16, i16, i32 }
	%struct.mbuf = type { %struct.m_hdr, %17 }
	%struct.packet_tags = type { %struct.m_tag* }
	%struct.pf_addr = type { %3 }
	%struct.pf_addr_wrap = type <{ %1, %5, i8, i8, [6 x i8] }>
	%struct.pf_anchor = type { %14, %14, %struct.pf_anchor*, %struct.pf_anchor_node, [64 x i8], [1024 x i8], %struct.pf_ruleset, i32, i32 }
	%struct.pf_anchor_node = type { %struct.pf_anchor* }
	%struct.pf_app_state = type { void (%struct.pf_state*, i32, i32, %struct.pf_pdesc*, %struct.pfi_kif*)*, i32 (%struct.pf_app_state*, %struct.pf_app_state*)*, i32 (%struct.pf_app_state*, %struct.pf_app_state*)*, %24 }
	%struct.pf_ike_state = type { i64 }
	%struct.pf_mtag = type { i8*, i32, i32, i16, i8, i8 }
	%struct.pf_palist = type { %struct.pf_pooladdr*, %struct.pf_pooladdr** }
	%struct.pf_pdesc = type { %struct.pf_threshold, i64, %23, %struct.pf_addr, %struct.pf_addr, %struct.pf_rule*, %struct.pf_addr*, %struct.pf_addr*, %struct.ether_header*, %struct.mbuf*, i32, %struct.pf_mtag*, i16*, i32, i16, i8, i8, i8, i8 }
	%struct.pf_pool = type { %struct.pf_palist, [2 x i32], %struct.pf_pooladdr*, [4 x i8], %struct.in6_addr, %struct.pf_addr, i32, [2 x i16], i8, i8, [1 x i32] }
	%struct.pf_pooladdr = type <{ %struct.pf_addr_wrap, %struct.pf_palist, [2 x i32], [16 x i8], %struct.pfi_kif*, [1 x i32] }>
	%struct.pf_rule = type <{ %struct.pf_rule_addr, %struct.pf_rule_addr, [8 x %struct.pf_rule_ptr], [64 x i8], [16 x i8], [64 x i8], [64 x i8], [64 x i8], [64 x i8], [32 x i8], %struct.pf_rulequeue, [2 x i32], %struct.pf_pool, i64, [2 x i64], [2 x i64], %struct.pfi_kif*, [4 x i8], %struct.pf_anchor*, [4 x i8], %struct.pfr_ktable*, [4 x i8], i32, i32, [26 x i32], i32, i32, i32, i32, i32, i32, %struct.au_mask_t, i32, i32, i32, i32, i32, i32, i32, i16, i16, i16, i16, i16, [2 x i8], %struct.pf_rule_gid, %struct.pf_rule_gid, i32, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, [2 x i8] }>
	%struct.pf_rule_addr = type <{ %struct.pf_addr_wrap, %struct.pf_rule_xport, i8, [7 x i8] }>
	%struct.pf_rule_gid = type { [2 x i32], i8, [3 x i8] }
	%struct.pf_rule_ptr = type { %struct.pf_rule*, [4 x i8] }
	%struct.pf_rule_xport = type { i32, [4 x i8] }
	%struct.pf_rulequeue = type { %struct.pf_rule*, %struct.pf_rule** }
	%struct.pf_ruleset = type { [5 x %12], %struct.pf_anchor*, i32, i32, i32 }
	%struct.pf_src_node = type <{ %26, %struct.pf_addr, %struct.pf_addr, %struct.pf_rule_ptr, %struct.pfi_kif*, [2 x i64], [2 x i64], i32, i32, %struct.pf_threshold, i64, i64, i8, i8, [2 x i8] }>
	%struct.pf_state = type <{ i64, i32, i32, %struct.anon, %struct.anon, %0, %struct.pf_state_peer, %struct.pf_state_peer, %struct.pf_rule_ptr, %struct.pf_rule_ptr, %struct.pf_rule_ptr, %struct.pf_addr, %struct.hook_desc_head, %struct.pf_state_key*, %struct.pfi_kif*, %struct.pfi_kif*, %struct.pf_src_node*, %struct.pf_src_node*, [2 x i64], [2 x i64], i64, i64, i64, i16, i8, i8, i8, i8, [6 x i8] }>
	%struct.pf_state_host = type { %struct.pf_addr, %struct.in_addr }
	%struct.pf_state_key = type { %struct.pf_state_host, %struct.pf_state_host, %struct.pf_state_host, i8, i8, i8, i8, %struct.pf_app_state*, %25, %25, %struct.anon, i16 }
	%struct.pf_state_peer = type { i32, i32, i32, i16, i8, i8, i16, i8, %struct.pf_state_scrub*, [3 x i8] }
	%struct.pf_state_scrub = type { %struct.au_mask_t, i32, i32, i32, i16, i8, i8, i32 }
	%struct.pf_threshold = type { i32, i32, i32, i32 }
	%struct.pfi_dynaddr = type { %6, %struct.pf_addr, %struct.pf_addr, %struct.pf_addr, %struct.pf_addr, %struct.pfr_ktable*, %struct.pfi_kif*, i8*, i32, i32, i32, i8, i8 }
	%struct.pfi_kif = type { [16 x i8], %15, [2 x [2 x [2 x i64]]], [2 x [2 x [2 x i64]]], i64, i32, i8*, %struct.ifnet*, i32, i32, %6 }
	%struct.pfr_ktable = type { %struct.pfr_tstats, %7, %8, %struct.radix_node_head*, %struct.radix_node_head*, %struct.pfr_ktable*, %struct.pfr_ktable*, %struct.pf_ruleset*, i64, i32 }
	%struct.pfr_table = type { [1024 x i8], [32 x i8], i32, i8 }
	%struct.pfr_tstats = type { %struct.pfr_table, [2 x [3 x i64]], [2 x [3 x i64]], i64, i64, i64, i32, [2 x i32] }
	%struct.pkthdr = type { i32, %struct.ifnet*, i8*, i32, i32, i32, i16, i16, %struct.packet_tags }
	%struct.proto_hash_entry = type opaque
	%struct.radix_mask = type { i16, i8, i8, %struct.radix_mask*, %9, i32 }
	%struct.radix_node = type { %struct.radix_mask*, %struct.radix_node*, i16, i8, i8, %10 }
	%struct.radix_node_head = type { %struct.radix_node*, i32, i32, %struct.radix_node* (i8*, i8*, %struct.radix_node_head*, %struct.radix_node*)*, %struct.radix_node* (i8*, i8*, %struct.radix_node_head*, %struct.radix_node*)*, %struct.radix_node* (i8*, i8*, %struct.radix_node_head*)*, %struct.radix_node* (i8*, i8*, %struct.radix_node_head*)*, %struct.radix_node* (i8*, %struct.radix_node_head*)*, %struct.radix_node* (i8*, %struct.radix_node_head*, i32 (%struct.radix_node*, i8*)*, i8*)*, %struct.radix_node* (i8*, i8*, %struct.radix_node_head*)*, %struct.radix_node* (i8*, i8*, %struct.radix_node_head*, i32 (%struct.radix_node*, i8*)*, i8*)*, %struct.radix_node* (i8*, %struct.radix_node_head*)*, i32 (%struct.radix_node_head*, i32 (%struct.radix_node*, i8*)*, i8*)*, i32 (%struct.radix_node_head*, i8*, i8*, i32 (%struct.radix_node*, i8*)*, i8*)*, void (%struct.radix_node*, %struct.radix_node_head*)*, [3 x %struct.radix_node], i32 }
	%struct.route = type { %struct.rtentry*, i32, %struct.sockaddr }
	%struct.rt_metrics = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [4 x i32] }
	%struct.rtentry = type { [2 x %struct.radix_node], %struct.sockaddr*, i32, i32, %struct.ifnet*, %struct.ifaddr*, %struct.sockaddr*, i8*, void (i8*)*, %struct.rt_metrics, %struct.rtentry*, %struct.rtentry*, i32, %struct.lck_mtx_t }
	%struct.sockaddr = type { i8, i8, [14 x i8] }
	%struct.tcphdr = type { i16, i16, i32, i32, i8, i8, i16, i16, i16 }
	%struct.thread = type opaque
@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 (%struct.pf_state_key*, %struct.pf_state_key*)* @pf_state_compare_ext_gwy to i8*)], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define fastcc i32 @pf_state_compare_ext_gwy(%struct.pf_state_key* nocapture %a, %struct.pf_state_key* nocapture %b) nounwind optsize ssp {
entry:
	%0 = zext i8 0 to i32		; <i32> [#uses=2]
	%1 = load i8* null, align 1		; <i8> [#uses=2]
	%2 = zext i8 %1 to i32		; <i32> [#uses=1]
	%3 = sub i32 %0, %2		; <i32> [#uses=1]
	%4 = icmp eq i8 0, %1		; <i1> [#uses=1]
	br i1 %4, label %bb1, label %bb79

bb1:		; preds = %entry
	%5 = load i8* null, align 4		; <i8> [#uses=2]
	%6 = zext i8 %5 to i32		; <i32> [#uses=2]
	%7 = getelementptr %struct.pf_state_key* %b, i32 0, i32 3		; <i8*> [#uses=1]
	%8 = load i8* %7, align 4		; <i8> [#uses=2]
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
	%12 = load i16* null, align 4		; <i16> [#uses=1]
	%13 = zext i16 %12 to i32		; <i32> [#uses=1]
	%14 = sub i32 0, %13		; <i32> [#uses=1]
	br i1 false, label %bb23, label %bb79

bb6:		; preds = %bb3
	%15 = load i16* null, align 4		; <i16> [#uses=1]
	%16 = zext i16 %15 to i32		; <i32> [#uses=1]
	%17 = sub i32 0, %16		; <i32> [#uses=1]
	ret i32 %17

bb10:		; preds = %bb3
	%18 = load i8* null, align 1		; <i8> [#uses=2]
	%19 = zext i8 %18 to i32		; <i32> [#uses=1]
	%20 = sub i32 0, %19		; <i32> [#uses=1]
	%21 = icmp eq i8 0, %18		; <i1> [#uses=1]
	br i1 %21, label %bb12, label %bb79

bb12:		; preds = %bb10
	%22 = load i16* null, align 4		; <i16> [#uses=1]
	%23 = zext i16 %22 to i32		; <i32> [#uses=1]
	%24 = sub i32 0, %23		; <i32> [#uses=1]
	ret i32 %24

bb17:		; preds = %bb3
	%25 = load i8* null, align 1		; <i8> [#uses=2]
	%26 = icmp eq i8 %25, 1		; <i1> [#uses=1]
	br i1 %26, label %bb18, label %bb23

bb18:		; preds = %bb17
	%27 = icmp eq i8 %25, 0		; <i1> [#uses=1]
	br i1 %27, label %bb19, label %bb23

bb19:		; preds = %bb18
	%28 = load i16* null, align 4		; <i16> [#uses=1]
	%29 = zext i16 %28 to i32		; <i32> [#uses=1]
	%30 = sub i32 0, %29		; <i32> [#uses=1]
	br i1 false, label %bb23, label %bb79

bb21:		; preds = %bb3
	%31 = getelementptr %struct.pf_state_key* %a, i32 0, i32 1, i32 1, i32 0		; <i32*> [#uses=1]
	%32 = load i32* %31, align 4		; <i32> [#uses=2]
	%33 = getelementptr %struct.pf_state_key* %b, i32 0, i32 1, i32 1, i32 0		; <i32*> [#uses=1]
	%34 = load i32* %33, align 4		; <i32> [#uses=2]
	%35 = sub i32 %32, %34		; <i32> [#uses=1]
	%36 = icmp eq i32 %32, %34		; <i1> [#uses=1]
	br i1 %36, label %bb23, label %bb79

bb23:		; preds = %bb21, %bb19, %bb18, %bb17, %bb4, %bb3
	%cond = icmp eq i32 %6, 2		; <i1> [#uses=1]
	br i1 %cond, label %bb24, label %bb70

bb24:		; preds = %bb23
	ret i32 1

bb70:		; preds = %bb23
	%37 = load i32 (%struct.pf_app_state*, %struct.pf_app_state*)** null, align 4		; <i32 (%struct.pf_app_state*, %struct.pf_app_state*)*> [#uses=3]
	br i1 false, label %bb78, label %bb73

bb73:		; preds = %bb70
	%38 = load i32 (%struct.pf_app_state*, %struct.pf_app_state*)** null, align 4		; <i32 (%struct.pf_app_state*, %struct.pf_app_state*)*> [#uses=2]
	%39 = icmp eq i32 (%struct.pf_app_state*, %struct.pf_app_state*)* %38, null		; <i1> [#uses=1]
	br i1 %39, label %bb78, label %bb74

bb74:		; preds = %bb73
	%40 = ptrtoint i32 (%struct.pf_app_state*, %struct.pf_app_state*)* %37 to i32		; <i32> [#uses=1]
	%41 = sub i32 0, %40		; <i32> [#uses=1]
	%42 = icmp eq i32 (%struct.pf_app_state*, %struct.pf_app_state*)* %38, %37		; <i1> [#uses=1]
	br i1 %42, label %bb76, label %bb79

bb76:		; preds = %bb74
	%43 = tail call i32 %37(%struct.pf_app_state* null, %struct.pf_app_state* null) nounwind		; <i32> [#uses=1]
	ret i32 %43

bb78:		; preds = %bb73, %bb70
	ret i32 0

bb79:		; preds = %bb74, %bb21, %bb19, %bb10, %bb4, %bb1, %entry
	%.0 = phi i32 [ %3, %entry ], [ %10, %bb1 ], [ %14, %bb4 ], [ %20, %bb10 ], [ %30, %bb19 ], [ %35, %bb21 ], [ %41, %bb74 ]		; <i32> [#uses=1]
	ret i32 %.0
}
