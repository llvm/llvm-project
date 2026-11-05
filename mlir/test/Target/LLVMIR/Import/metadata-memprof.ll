; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; A single MemInfoBlock: one allocation call stack and its allocation-type tag.

; CHECK-LABEL: @simple_memprof
define ptr @simple_memprof() {
  ; CHECK: llvm.call @alloc({{.*}}) {llvm.callsite = array<i64: 123456789>, llvm.memprof = [#llvm.memprof_mib<call_stack = [123456789], alloc_type = "cold">]}
  %p = call ptr @alloc(i64 16), !memprof !0, !callsite !2
  ret ptr %p
}

declare ptr @alloc(i64)

!0 = !{!1}
!1 = !{!2, !"cold"}
!2 = !{i64 123456789}

; // -----

; Several MemInfoBlocks, a multi-frame call stack with a top-bit hash, and two
; trailing (context id, total size) pairs, one with an id above INT64_MAX.

; CHECK-LABEL: @rich_memprof
define ptr @rich_memprof(i64 %n) {
  ; CHECK: llvm.callsite = array<i64: 333>
  ; CHECK-SAME: llvm.memprof = [#llvm.memprof_mib<call_stack = [111, -6528110295079665978], alloc_type = "cold", context_sizes = <id = 999, size = 4096>, <id = 18446744073709551614, size = 8>>, #llvm.memprof_mib<call_stack = [333], alloc_type = "notcold">]
  %p = call ptr @alloc(i64 %n), !memprof !0, !callsite !6
  ret ptr %p
}

declare ptr @alloc(i64)

!0 = !{!1, !5}
!1 = !{!2, !"cold", !3, !4}
!2 = !{i64 111, i64 -6528110295079665978}
!3 = !{i64 999, i64 4096}
!4 = !{i64 -2, i64 8}
!5 = !{!6, !"notcold"}
!6 = !{i64 333}

; // -----

; An invoke is a `CallBase` too, so it carries memprof just as a call does.

; CHECK-LABEL: @memprof_on_invoke
define ptr @memprof_on_invoke() personality ptr @__gxx_personality_v0 {
  ; CHECK: llvm.invoke @alloc
  ; CHECK-SAME: llvm.callsite = array<i64: 7>, llvm.memprof = [#llvm.memprof_mib<call_stack = [7], alloc_type = "hot">]
  %p = invoke ptr @alloc(i64 16) to label %ok unwind label %bad, !memprof !0, !callsite !2
ok:
  ret ptr %p
bad:
  %l = landingpad { ptr, i32 } cleanup
  ret ptr null
}

declare ptr @alloc(i64)
declare i32 @__gxx_personality_v0(...)

!0 = !{!1}
!1 = !{!2, !"hot"}
!2 = !{i64 7}

; // -----

; A call carrying only `!callsite`: intermediate frames have no allocation of
; their own, so there is no `!memprof` record for them.

; CHECK-LABEL: @callsite_only
define ptr @callsite_only() {
  ; CHECK: llvm.call @alloc({{.*}}) {llvm.callsite = array<i64: 111, 222>}
  %p = call ptr @alloc(i64 16), !callsite !0
  ret ptr %p
}

declare ptr @alloc(i64)

!0 = !{i64 111, i64 222}
