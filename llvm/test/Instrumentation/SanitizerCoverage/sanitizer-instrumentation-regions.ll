; Test that functions that are not instrumented as a whole are instrumented
; inside llvm.sanitizer.instrumentation.begin()/end() regions.
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=4 \
; RUN:   -sanitizer-coverage-trace-pc -sanitizer-coverage-trace-compares \
; RUN:   -sanitizer-coverage-trace-loads -sanitizer-coverage-trace-stores \
; RUN:   -sanitizer-coverage-trace-divs \
; RUN:   -sanitizer-coverage-trace-pc-entry-exit -sanitizer-coverage-stack-depth \
; RUN:   -sanitizer-coverage-control-flow -S | FileCheck %s
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 \
; RUN:   -sanitizer-coverage-inline-8bit-counters -sanitizer-coverage-pc-table \
; RUN:   -S | FileCheck %s --check-prefix=PCTABLE
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=4 \
; RUN:   -sanitizer-coverage-trace-pc-guard -sanitizer-coverage-trace-compares \
; RUN:   -sanitizer-coverage-gated-trace-callbacks -S \
; RUN:   | FileCheck %s --check-prefix=GATED

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.sanitizer.instrumentation.begin()
declare void @llvm.sanitizer.instrumentation.end()
declare void @foo()

; PCTABLE: @__sancov_gen_{{.*}} = private constant {{.*}} [ptr blockaddress(@region_disable, %sancov.region), ptr null], section "__sancov_pcs"

; Only the code between begin() and end() is instrumented. The block that
; contains begin() is split right after it, so that the coverage callback runs
; inside the region, and the function gets no entry/exit callbacks, no stack
; depth tracking and no control flow table.
define i32 @region(ptr %p, i32 %x, ptr %fp) nosanitize_coverage {
entry:
  %out.cmp = icmp eq i32 %x, 42
  %out.load = load i32, ptr %p
  call void @llvm.sanitizer.instrumentation.begin()
  %in.cmp = icmp eq i32 %x, 7
  %in.load = load i32, ptr %p
  store i32 %x, ptr %p
  %in.div = udiv i32 %in.load, %x
  call void %fp()
  call void @llvm.sanitizer.instrumentation.end()
  store i32 %out.load, ptr %p
  call void %fp()
  %r = select i1 %out.cmp, i32 %in.div, i32 0
  ret i32 %r
}

; CHECK-LABEL: define i32 @region(
; CHECK-NEXT:  entry:
; CHECK-NOT:     call void @__sanitizer_cov
; CHECK:         call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    br label %sancov.region
; CHECK:       sancov.region:
; CHECK-NEXT:    call void @__sanitizer_cov_trace_pc()
; CHECK-NEXT:    call void @__sanitizer_cov_trace_const_cmp4(i32 7, i32 %x)
; CHECK-NEXT:    %in.cmp = icmp eq i32 %x, 7
; CHECK-NEXT:    call void @__sanitizer_cov_load4(ptr %p)
; CHECK-NEXT:    %in.load = load i32, ptr %p
; CHECK-NEXT:    call void @__sanitizer_cov_store4(ptr %p)
; CHECK-NEXT:    store i32 %x, ptr %p
; CHECK-NEXT:    call void @__sanitizer_cov_trace_div4(i32 %x)
; CHECK-NEXT:    %in.div = udiv i32 %in.load, %x
; CHECK-NEXT:    %0 = ptrtoint ptr %fp to i64
; CHECK-NEXT:    call void @__sanitizer_cov_trace_pc_indir(i64 %0)
; CHECK-NEXT:    call void %fp()
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.end()
; CHECK-NOT:     call void @__sanitizer_cov
; CHECK:         ret i32 %r

; Every block of a region is covered, including the blocks inserted to split
; critical edges: pruning does not apply, as it assumes that all blocks are
; instrumented. A block whose only in-region instruction is end() is not.
define void @region_blocks(i1 %c) nosanitize_coverage {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  br i1 %c, label %then, label %join

then:
  call void @foo()
  br label %join

join:
  call void @llvm.sanitizer.instrumentation.end()
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: define void @region_blocks(
; CHECK:       entry:
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    br label %sancov.region
; CHECK:       sancov.region:
; CHECK-NEXT:    call void @__sanitizer_cov_trace_pc()
; CHECK-NEXT:    br i1 %c, label %then, label %entry.join_crit_edge
; CHECK:       entry.join_crit_edge:
; CHECK-NEXT:    call void @__sanitizer_cov_trace_pc()
; CHECK:       then:
; CHECK-NEXT:    call void @__sanitizer_cov_trace_pc()
; CHECK-NEXT:    call void @foo()
; CHECK:       join:
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.end()
; CHECK:       exit:
; CHECK-NEXT:    ret void

; disable_sanitizer_instrumentation functions are handled the same way.
define void @region_disable(i32 %x) disable_sanitizer_instrumentation {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  %c = icmp eq i32 %x, 7
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; CHECK-LABEL: define void @region_disable(
; CHECK:         call void @llvm.sanitizer.instrumentation.begin()
; CHECK:       sancov.region:
; CHECK-NEXT:    call void @__sanitizer_cov_trace_pc()
; CHECK-NEXT:    call void @__sanitizer_cov_trace_const_cmp4(i32 7, i32 %x)
; CHECK:         call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT:    ret void

; Static allocas after the region coverage point stay in the entry block when
; it is split; otherwise they would become dynamic allocas.
define void @entry_allocas(i32 %x) nosanitize_coverage {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  call void @foo()
  %a = alloca i32
  %b = alloca i32, i32 %x
  %c = alloca i64
  store i32 %x, ptr %a
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; CHECK-LABEL: define void @entry_allocas(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    %a = alloca i32
; CHECK-NEXT:    %c = alloca i64
; CHECK-NEXT:    br label %sancov.region
; CHECK:       sancov.region:
; CHECK-NEXT:    call void @__sanitizer_cov_trace_pc()
; CHECK-NEXT:    call void @foo()
; CHECK-NEXT:    %b = alloca i32, i32 %x
; CHECK:         store i32 %x, ptr %a

; Only the critical edges inside the regions are split (see @region_blocks);
; the ones outside are left alone.
define void @outside_crit_edge(i1 %c) nosanitize_coverage {
entry:
  br i1 %c, label %then, label %join

then:
  call void @foo()
  br label %join

join:
  call void @llvm.sanitizer.instrumentation.begin()
  call void @foo()
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; CHECK-LABEL: define void @outside_crit_edge(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %c, label %then, label %join
; CHECK-NOT:   crit_edge
; CHECK:       join:
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    br label %sancov.region
; CHECK:       sancov.region:
; CHECK-NEXT:    call void @__sanitizer_cov_trace_pc()
; CHECK-NEXT:    call void @foo()

; With gated callbacks, the gate is read inside the region, not in the entry
; block.
; GATED-LABEL: define void @region_disable(
; GATED-NEXT:  entry:
; GATED-NEXT:    call void @llvm.sanitizer.instrumentation.begin()
; GATED-NEXT:    br label %sancov.region
; GATED:       sancov.region:
; GATED-NEXT:    load i64, ptr @__sancov_should_track
; GATED-NOT:   @llvm.sanitizer.instrumentation.end()
; GATED:         call void @__sanitizer_cov_trace_pc_guard(
; GATED:         call void @__sanitizer_cov_trace_const_cmp4(i32 7, i32 %x)
; GATED:         call void @llvm.sanitizer.instrumentation.end()

; A block reachable without passing through begin() is not in a region.
define void @diamond(i1 %c, i32 %x) nosanitize_coverage {
entry:
  br i1 %c, label %enter, label %join

enter:
  call void @llvm.sanitizer.instrumentation.begin()
  br label %join

join:
  %cmp = icmp eq i32 %x, 7
  call void @foo()
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; CHECK-LABEL: define void @diamond(
; CHECK:       enter:
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    br label %sancov.region
; CHECK:       sancov.region:
; CHECK-NEXT:    call void @__sanitizer_cov_trace_pc()
; CHECK:       join:
; CHECK-NOT:     call void @__sanitizer_cov
; CHECK:         ret void

; Regions nest: the code after an inner end() is still covered.
define void @nested(i32 %x) nosanitize_coverage {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.end()
  %cmp = icmp eq i32 %x, 7
  call void @llvm.sanitizer.instrumentation.end()
  %cmp2 = icmp eq i32 %x, 8
  ret void
}

; CHECK-LABEL: define void @nested(
; CHECK:         call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    br label %sancov.region
; CHECK:       sancov.region:
; CHECK-NEXT:    call void @__sanitizer_cov_trace_pc()
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT:    call void @__sanitizer_cov_trace_const_cmp4(i32 7, i32 %x)
; CHECK-NEXT:    %cmp = icmp eq i32 %x, 7
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT:    %cmp2 = icmp eq i32 %x, 8
; CHECK-NEXT:    ret void

; Without regions, nosanitize_coverage functions are left alone.
define void @no_region(i32 %x) nosanitize_coverage {
entry:
  %cmp = icmp eq i32 %x, 7
  call void @foo()
  ret void
}

; CHECK-LABEL: define void @no_region(
; CHECK-NOT:     call void @__sanitizer_cov
; CHECK:         ret void

; Instrumented functions are instrumented as usual, regions or not.
define void @instrumented(i32 %x) nounwind {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  call void @foo()
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; CHECK-LABEL: define void @instrumented(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @__sanitizer_cov_trace_pc_entry()
; CHECK:         call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    call void @foo()
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT:    call void @__sanitizer_cov_trace_pc_exit()

; The PC table refers to the split-off region block, never to the entry block of
; a region-only function.
; PCTABLE-LABEL: define void @region_disable(
; PCTABLE:         call void @llvm.sanitizer.instrumentation.begin()
; PCTABLE:       sancov.region:
; PCTABLE-NEXT:    load i8, ptr @__sancov_gen_
; PCTABLE:         call void @llvm.sanitizer.instrumentation.end()
