;; Tests that globals referenced via !implicit.ref metadata on a function
;; are added as explicit reference edges in the module summary.
;; This ensures ThinLTO liveness analysis marks them live when the function
;; is live, preventing GlobalDCE from eliminating them.

; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s

target datalayout = "E-m:a-p:32:32-Fi32-i64:64-n32-f64:32:64"
target triple = "powerpc-ibm-aix7.3.0.0"

@counter = global i32 0, align 4
@__loadtime_comment_str = internal unnamed_addr constant [14 x i8] c"Copyright TU1\00",
                          section "__loadtime_comment", align 1
@llvm.compiler.used = appending global [1 x ptr] [ptr @__loadtime_comment_str], 
                     section "llvm.metadata"

define void @f_tu1() !implicit.ref !0 {
entry:
  %0 = load i32, ptr @counter, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr @counter, align 4
  ret void
}

!llvm.module.flags = !{!1, !2}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 1, !"EnableSplitLTOUnit", i32 0}
!0 = !{ptr @__loadtime_comment_str}

;; Verify key globals appear in the summary (order is platform-dependent
;; since internal globals have GUIDs that include the module path).
; CHECK-DAG: gv: (name: "__loadtime_comment_str"{{.*}}linkage: internal{{.*}}notEligibleToImport: 1
; CHECK-DAG: gv: (name: "counter"

;; f_tu1 must have exactly two ref edges -- counter (existing IR use) and
;; __loadtime_comment_str (added by findImplicitRefEdges fix). It must also
;; be notEligibleToImport because it references an internal global.
; CHECK-DAG: gv: (name: "f_tu1"{{.*}}notEligibleToImport: 1{{.*}}refs: (^{{[0-9]+}}, ^{{[0-9]+}})