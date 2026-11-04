; REQUIRES:x86_64-linux

; Test flattening indirect calls into "VP" MD_prof metadata, in prelink.

; RUN: split-file %s %t
; RUN: llvm-ctxprof-util fromYAML --input %t/profile.yaml --output %t/profile.ctxprofdata
; RUN: opt -passes=ctx-prof-flatten-prethinlink %t/example.ll -use-ctx-profile=%t/profile.ctxprofdata \
; RUN:   -S -o - | FileCheck %s --check-prefix=PRELINK

; PRELINK:      call void @llvm.instrprof.callsite(ptr @foo, i64 1234, i32 2, i32 0, ptr %p)
; PRELINK-NEXT: call void %p(), !prof ![[VPPROF:[0-9]+]]
; PRELINK-NEXT: call void @llvm.instrprof.callsite(ptr @foo, i64 1234, i32 2, i32 1, ptr @bar)
; PRELINK-NEXT: call void @bar(){{$}}
; PRELINK:      ![[VPPROF]] = !{!"VP", i32 0, i64 25, i64 5678, i64 20, i64 5555, i64 5}

; RUN: cp %t/example.ll %t/1234.ll
; RUN: opt -passes=ctx-prof-flatten %t/1234.ll -use-ctx-profile=%t/profile.ctxprofdata \
; RUN:   -S -o - | FileCheck %s --check-prefix=POSTLINK
; RUN: opt -passes=ctx-prof-flatten %t/example.ll -use-ctx-profile=%t/profile.ctxprofdata \
; RUN:   -S -o - | FileCheck %s --check-prefix=POSTLINK

; POSTLINK-NOT: call void %p(), !prof
;--- example.ll

declare !guid !0 void @bar()

define void @foo(ptr %p) !guid !1 {
  call void @llvm.instrprof.increment(ptr @foo, i64 1234, i32 1, i32 0)
  call void @llvm.instrprof.callsite(ptr @foo, i64 1234, i32 2, i32 0, ptr %p)
  call void %p()
  call void @llvm.instrprof.callsite(ptr @foo, i64 1234, i32 2, i32 1, ptr @bar)
  call void @bar()
  ret void
}

!0 = !{i64 8888}
!1 = !{i64 1234}

;--- profile.yaml
Contexts:
  - Guid: 1234
    TotalRootEntryCount: 5
    Counters: [5]
    Callsites:
      - - Guid: 5555
          Counters: [1]
        - Guid: 5678
          Counters: [4]
      - - Guid: 8888
          Counters: [5]
