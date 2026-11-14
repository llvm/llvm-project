;; Test -import-declaration-for-indirect-calls: when the definition of the
;; target of a profiled indirect call is not imported, a declaration is imported
;; instead and the target is exported from its defining module, so that the
;; backend can promote the indirect call to it under -icp-allow-decls. Only
;; edges at least as hot as -import-declaration-for-indirect-calls-min-hotness
;; (hot by default) are considered.

; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/main.ll -o %t/main.o
; RUN: opt -thinlto-bc %t/foo.ll -o %t/foo.o

;; The indirect call edge from foo to bar is hot.
; RUN: llvm-dis %t/foo.o -o - | FileCheck %s --check-prefix=SUMMARY
; SUMMARY: gv: (name: "foo", {{.*}} calls: ((callee: ^{{[0-9]+}}, hotness: hot, indirect: 1))

;; bar is noinline, so its definition is not imported into foo's module. It is
;; only referenced via its address in main's module, so it would be internalized
;; unless the declaration import exports it.
; RUN: llvm-lto2 run %t/main.o %t/foo.o \
; RUN:  -import-declaration-for-indirect-calls -icp-allow-decls \
; RUN:  -r=%t/foo.o,foo,plx \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,foo, \
; RUN:  -r=%t/main.o,bar,pl \
; RUN:  -r=%t/main.o,fptr,plx \
; RUN:  -thinlto-threads=1 \
; RUN:  -pass-remarks=pgo-icall-prom \
; RUN:  -save-temps \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefix=REMARKS
; RUN: llvm-dis %t.out.2.3.import.bc -o - | FileCheck %s --check-prefix=IMPORT
; RUN: llvm-nm %t.out.1 | FileCheck %s --check-prefix=EXPORTED
; RUN: llvm-nm %t.out.2 | FileCheck %s --check-prefix=REFERENCED

; REMARKS: Promote indirect call to bar with count 10000 out of 10000
; IMPORT: declare !guid !{{[0-9]+}} void @bar()
; EXPORTED: T bar
; REFERENCED: U bar

;; Raising the hotness bar above the edge's hotness disables the declaration
;; import: bar is not found in foo's module and is internalized.
; RUN: llvm-lto2 run %t/main.o %t/foo.o \
; RUN:  -import-declaration-for-indirect-calls \
; RUN:  -import-declaration-for-indirect-calls-min-hotness=critical \
; RUN:  -icp-allow-decls \
; RUN:  -r=%t/foo.o,foo,plx \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,foo, \
; RUN:  -r=%t/main.o,bar,pl \
; RUN:  -r=%t/main.o,fptr,plx \
; RUN:  -thinlto-threads=1 \
; RUN:  -pass-remarks-missed=pgo-icall-prom \
; RUN:  -o %t.gated.out 2>&1 | FileCheck %s --check-prefix=REMARKS-GATED
; RUN: llvm-nm %t.gated.out.1 | FileCheck %s --check-prefix=INTERNALIZED

; REMARKS-GATED: Cannot promote indirect call: target with md5sum 16434608426314478903 not found (count=10000)
; INTERNALIZED: t bar

;--- foo.ll
source_filename = "foo.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %fp) !prof !15 {
entry:
  call void %fp(), !prof !0
  ret void
}

!0 = !{!"VP", i32 0, i64 10000, i64 -2012135647395072713, i64 10000}

;; Profile summary with a hot count threshold of 100, so the call edge is hot.
!llvm.module.flags = !{!1}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 10000}
!6 = !{!"MaxInternalCount", i64 10000}
!7 = !{!"MaxFunctionCount", i64 10000}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 100, i32 1}
!13 = !{i32 990000, i64 100, i32 1}
!14 = !{i32 999999, i64 1, i32 2}
!15 = !{!"function_entry_count", i64 10000}

;--- main.ll
source_filename = "main.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@fptr = global ptr @bar

define i32 @main() {
entry:
  %fp = load ptr, ptr @fptr
  call void @foo(ptr %fp)
  ret i32 0
}

declare void @foo(ptr)

;; The noinline attribute prevents importing bar's definition.
define void @bar() #0 {
entry:
  ret void
}

attributes #0 = { noinline }
