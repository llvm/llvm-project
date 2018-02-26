; RUN: llc < %s -mtriple=powerpc64le-unknown-unknown -verify-machineinstrs \
; RUN:   -mcpu=pwr8 | FileCheck %s
module asm "\09.section \22___kcrctab+numa_node\22, \22a\22\09"
module asm "\09.weak\09__crc_numa_node\09"
module asm "\09.long\09__crc_numa_node\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+_numa_mem_\22, \22a\22\09"
module asm "\09.weak\09__crc__numa_mem_\09"
module asm "\09.long\09__crc__numa_mem_\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+node_states\22, \22a\22\09"
module asm "\09.weak\09__crc_node_states\09"
module asm "\09.long\09__crc_node_states\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+totalram_pages\22, \22a\22\09"
module asm "\09.weak\09__crc_totalram_pages\09"
module asm "\09.long\09__crc_totalram_pages\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+movable_zone\22, \22a\22\09"
module asm "\09.weak\09__crc_movable_zone\09"
module asm "\09.long\09__crc_movable_zone\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+nr_node_ids\22, \22a\22\09"
module asm "\09.weak\09__crc_nr_node_ids\09"
module asm "\09.long\09__crc_nr_node_ids\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+nr_online_nodes\22, \22a\22\09"
module asm "\09.weak\09__crc_nr_online_nodes\09"
module asm "\09.long\09__crc_nr_online_nodes\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab_gpl+split_page\22, \22a\22\09"
module asm "\09.weak\09__crc_split_page\09"
module asm "\09.long\09__crc_split_page\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+__alloc_pages_nodemask\22, \22a\22\09"
module asm "\09.weak\09__crc___alloc_pages_nodemask\09"
module asm "\09.long\09__crc___alloc_pages_nodemask\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+__get_free_pages\22, \22a\22\09"
module asm "\09.weak\09__crc___get_free_pages\09"
module asm "\09.long\09__crc___get_free_pages\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+get_zeroed_page\22, \22a\22\09"
module asm "\09.weak\09__crc_get_zeroed_page\09"
module asm "\09.long\09__crc_get_zeroed_page\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+__free_pages\22, \22a\22\09"
module asm "\09.weak\09__crc___free_pages\09"
module asm "\09.long\09__crc___free_pages\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+free_pages\22, \22a\22\09"
module asm "\09.weak\09__crc_free_pages\09"
module asm "\09.long\09__crc_free_pages\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+__page_frag_cache_drain\22, \22a\22\09"
module asm "\09.weak\09__crc___page_frag_cache_drain\09"
module asm "\09.long\09__crc___page_frag_cache_drain\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+page_frag_alloc\22, \22a\22\09"
module asm "\09.weak\09__crc_page_frag_alloc\09"
module asm "\09.long\09__crc_page_frag_alloc\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+page_frag_free\22, \22a\22\09"
module asm "\09.weak\09__crc_page_frag_free\09"
module asm "\09.long\09__crc_page_frag_free\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+alloc_pages_exact\22, \22a\22\09"
module asm "\09.weak\09__crc_alloc_pages_exact\09"
module asm "\09.long\09__crc_alloc_pages_exact\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+free_pages_exact\22, \22a\22\09"
module asm "\09.weak\09__crc_free_pages_exact\09"
module asm "\09.long\09__crc_free_pages_exact\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab_gpl+nr_free_buffer_pages\22, \22a\22\09"
module asm "\09.weak\09__crc_nr_free_buffer_pages\09"
module asm "\09.long\09__crc_nr_free_buffer_pages\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab_gpl+si_mem_available\22, \22a\22\09"
module asm "\09.weak\09__crc_si_mem_available\09"
module asm "\09.long\09__crc_si_mem_available\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+si_meminfo\22, \22a\22\09"
module asm "\09.weak\09__crc_si_meminfo\09"
module asm "\09.long\09__crc_si_meminfo\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+adjust_managed_page_count\22, \22a\22\09"
module asm "\09.weak\09__crc_adjust_managed_page_count\09"
module asm "\09.long\09__crc_adjust_managed_page_count\09"
module asm "\09.previous\09\09\09\09\09"
module asm "\09.section \22___kcrctab+free_reserved_area\22, \22a\22\09"
module asm "\09.weak\09__crc_free_reserved_area\09"
module asm "\09.long\09__crc_free_reserved_area\09"
module asm "\09.previous\09\09\09\09\09"

@nr_cpu_ids = external local_unnamed_addr global i32, align 4

; Function Attrs: nounwind
define void @__alloc_pages_nodemask() #0 {
entry:
  %0 = call i64 asm sideeffect "ld${1:U}${1:X} $0,$1", "=r,*m"(i64* undef)
  br i1 undef, label %do.body.lr.ph.i.i.i, label %zone_page_state_snapshot.exit.i.i
; CHECK: ld 3, 0(3)

do.body.lr.ph.i.i.i:                              ; preds = %entry
  br label %do.body.i.i.i

do.body.i.i.i:                                    ; preds = %do.body.i.i.i, %do.body.lr.ph.i.i.i
  %x.022.i.i.i = phi i64 [ %0, %do.body.lr.ph.i.i.i ], [ %add7.i.i.i, %do.body.i.i.i ]
  %1 = load i8, i8* undef, align 1
  %conv.i.i458.i = sext i8 %1 to i64
  %add7.i.i.i = add i64 %x.022.i.i.i, %conv.i.i458.i
  %2 = load i32, i32* @nr_cpu_ids, align 4
  %cmp.i1.i.i = icmp ult i32 0, %2
  br i1 %cmp.i1.i.i, label %do.body.i.i.i, label %zone_page_state_snapshot.exit.i.i

zone_page_state_snapshot.exit.i.i:                ; preds = %do.body.i.i.i, %entry
  %x.0.lcssa.i.i.i = phi i64 [ %0, %entry ], [ %add7.i.i.i, %do.body.i.i.i ]
  %3 = icmp sgt i64 %x.0.lcssa.i.i.i, 0
  unreachable
}
