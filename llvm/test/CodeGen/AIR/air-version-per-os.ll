; The !air.version, !air.language_version and !air.compile_options metadata are
; derived from the target macOS major parsed out of the triple, so a single
; backend build emits AIR matching whichever macOS it targets. Values verified
; against Apple's `xcrun metal -mmacosx-version-min=N`.
;
; RUN: llc -mtriple=air64_v25-apple-macosx13.0.0 %s -o - | FileCheck %s --check-prefix=OS13
; RUN: llc -mtriple=air64_v26-apple-macosx14.0.0 %s -o - | FileCheck %s --check-prefix=OS14
; RUN: llc -mtriple=air64_v27-apple-macosx15.0.0 %s -o - | FileCheck %s --check-prefix=OS15
; RUN: llc -mtriple=air64_v28-apple-macosx26.0.0 %s -o - | FileCheck %s --check-prefix=OS26
; RUN: llc -mtriple=air %s -o - | FileCheck %s --check-prefix=FALLBACK

define void @kernel(ptr addrspace(1) %p) {
entry:
  store i32 0, ptr addrspace(1) %p
  ret void
}

; OS13-DAG: !air.version = !{[[V13:![0-9]+]]}
; OS13-DAG: !air.language_version = !{[[L13:![0-9]+]]}
; OS13-DAG: [[V13]] = !{i32 2, i32 5, i32 0}
; OS13-DAG: [[L13]] = !{!"AIR", i32 3, i32 0, i32 0}

; OS14-DAG: !air.version = !{[[V14:![0-9]+]]}
; OS14-DAG: !air.language_version = !{[[L14:![0-9]+]]}
; OS14-DAG: [[V14]] = !{i32 2, i32 6, i32 0}
; OS14-DAG: [[L14]] = !{!"AIR", i32 3, i32 1, i32 0}

; OS15-DAG: !air.version = !{[[V15:![0-9]+]]}
; OS15-DAG: !air.language_version = !{[[L15:![0-9]+]]}
; OS15-DAG: [[V15]] = !{i32 2, i32 7, i32 0}
; OS15-DAG: [[L15]] = !{!"AIR", i32 3, i32 2, i32 0}

; OS26-DAG: !air.version = !{[[V26:![0-9]+]]}
; OS26-DAG: !air.language_version = !{[[L26:![0-9]+]]}
; OS26-DAG: [[V26]] = !{i32 2, i32 8, i32 0}
; OS26-DAG: [[L26]] = !{!"AIR", i32 4, i32 0, i32 0}

; A bare "air" triple (no macosx component) falls back to the current shipping
; target (macOS 16 / 26-era).
; FALLBACK-DAG: !air.version = !{[[VF:![0-9]+]]}
; FALLBACK-DAG: !air.language_version = !{[[LF:![0-9]+]]}
; FALLBACK-DAG: [[VF]] = !{i32 2, i32 8, i32 0}
; FALLBACK-DAG: [[LF]] = !{!"AIR", i32 4, i32 0, i32 0}

; All targets emit the three compile options the stricter older drivers require.
; OS13-DAG: !air.compile_options = !{
; OS26-DAG: !air.compile_options = !{
