; Check which functions indirect call promotion can match to profiled targets,
; with and without -icp-lto. In LTO mode, functions are named as if they had
; external linkage, i.e. without a file name prefix.

; RUN: rm -rf %t && split-file %s %t

; A function LTO has internalized, without a GUID. This shouldn't happen outside
; of tests, as LTO pre-link pipelines assign GUIDs. It's only found by its plain
; name, in LTO mode.
; RUN: opt -passes=pgo-icall-prom -icp-lto -pass-remarks=pgo-icall-prom -S \
; RUN:   %t/internalized.ll 2>&1 | FileCheck %s --check-prefix=PROMOTED
; RUN: opt -passes=pgo-icall-prom -pass-remarks=pgo-icall-prom -S \
; RUN:   %t/internalized.ll 2>&1 | FileCheck %s --check-prefix=NOT-PROMOTED

; With a GUID, it's found by that, with or without LTO mode.
; RUN: opt -passes=pgo-icall-prom -icp-lto -pass-remarks=pgo-icall-prom -S \
; RUN:   %t/internalized-guid.ll 2>&1 | FileCheck %s --check-prefix=PROMOTED
; RUN: opt -passes=pgo-icall-prom -pass-remarks=pgo-icall-prom -S \
; RUN:   %t/internalized-guid.ll 2>&1 | FileCheck %s --check-prefix=PROMOTED

; Sample profiles key local functions by their plain name, so they're only
; found in LTO mode. This is true even with a GUID, as the GUID of a local
; function includes the file name.
; RUN: opt -passes=pgo-icall-prom -icp-lto -icp-samplepgo \
; RUN:   -pass-remarks=pgo-icall-prom -S %t/sample-local.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=PROMOTED
; RUN: opt -passes=pgo-icall-prom -icp-samplepgo \
; RUN:   -pass-remarks=pgo-icall-prom -S %t/sample-local.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOT-PROMOTED
; RUN: opt -passes=pgo-icall-prom -icp-lto -icp-samplepgo \
; RUN:   -pass-remarks=pgo-icall-prom -S %t/sample-local-guid.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=PROMOTED
; RUN: opt -passes=pgo-icall-prom -icp-samplepgo \
; RUN:   -pass-remarks=pgo-icall-prom -S %t/sample-local-guid.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOT-PROMOTED

; PROMOTED: remark: {{.*}} Promote indirect call to {{.*}}callee{{.*}} with count 100 out of 100
; PROMOTED: icmp eq ptr %fp, @{{.*}}callee

; NOT-PROMOTED-NOT: remark:
; NOT-PROMOTED: call void %fp(), !prof

;--- internalized.ll
source_filename = "a.cc"

define internal void @callee() {
  ret void
}

define void @caller(ptr %fp) {
  call void %fp(), !prof !0
  ret void
}

; MD5("callee")
!0 = !{!"VP", i32 0, i64 100, i64 -6621307527791283157, i64 100}

;--- internalized-guid.ll
source_filename = "a.cc"

define internal void @callee() !guid !1 {
  ret void
}

define void @caller(ptr %fp) {
  call void %fp(), !prof !0
  ret void
}

; MD5("callee")
!0 = !{!"VP", i32 0, i64 100, i64 -6621307527791283157, i64 100}
; The GUID it was assigned before LTO internalized it: MD5("callee")
!1 = !{i64 -6621307527791283157}

;--- sample-local.ll
source_filename = "a.cc"

define internal void @_ZL6calleev() {
  ret void
}

define void @caller(ptr %fp) {
  call void %fp(), !prof !0
  ret void
}

; MD5("_ZL6calleev")
!0 = !{!"VP", i32 0, i64 100, i64 4442334423817298497, i64 100}

;--- sample-local-guid.ll
source_filename = "a.cc"

define internal void @_ZL6calleev() !guid !1 {
  ret void
}

define void @caller(ptr %fp) {
  call void %fp(), !prof !0
  ret void
}

; MD5("_ZL6calleev")
!0 = !{!"VP", i32 0, i64 100, i64 4442334423817298497, i64 100}
; MD5("a.cc;_ZL6calleev")
!1 = !{i64 -2075408193200958154}
