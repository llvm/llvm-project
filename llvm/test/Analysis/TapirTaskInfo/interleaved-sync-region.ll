; Test Tapir's may-happen-in-parallel analysis when detaches and
; discriminating syncs interleave the set of parallel tasks.  Thanks
; to George Stelle for the inspiration for this test.

; RUN: opt -analyze -tasks -print-may-happen-in-parallel < %s 2>&1 | FileCheck %s
; RUN: opt -passes="print<tasks>" -print-may-happen-in-parallel -disable-output < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local void @interleaved_syncreg() #0 {
entry:
  %retval = alloca i32, align 4
  %syncrega = call token @llvm.syncregion.start()
  %syncregb = call token @llvm.syncregion.start()
  br label %while.body

while.body:                                       ; preds = %entry, %det.cont4
  sync within %syncrega, label %sync.continue

sync.continue:                                    ; preds = %while.body
  detach within %syncrega, label %det.achd, label %det.cont

det.achd:                                         ; preds = %sync.continue
  %call = call i32 (...) @foo()
  reattach within %syncrega, label %det.cont

det.cont:                                         ; preds = %det.achd, %sync.continue
  sync within %syncregb, label %sync.continue1

sync.continue1:                                   ; preds = %det.cont
  detach within %syncregb, label %det.achd2, label %det.cont4

det.achd2:                                        ; preds = %sync.continue1
  %call3 = call i32 (...) @bar()
  reattach within %syncregb, label %det.cont4

det.cont4:                                        ; preds = %det.achd2, %sync.continue1
  br label %while.body
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

declare dso_local i32 @foo(...)

declare dso_local i32 @bar(...)

attributes #0 = { nounwind uwtable  }
attributes #1 = { argmemonly nounwind }

; CHECK: task at depth 0 containing: <task entry><func sp entry>%entry<sp exit><phi sp entry>%while.body<sp exit><sync sp entry>%sync.continue<sp exit><phi sp entry>%det.cont<sp exit><sync sp entry>%sync.continue1<sp exit><phi sp entry>%det.cont4<sp exit>
; CHECK: task at depth 1 containing: <task entry><task sp entry>%det.achd2<sp exit><task exit>
; CHECK: task at depth 1 containing: <task entry><task sp entry>%det.achd<sp exit><task exit>

; CHECK: Spindle @ while.body may happen in parallel with:
; CHECK: task @ det.achd2
; CHECK: task @ det.achd
; CHECK: Spindle @ sync.continue may happen in parallel with:
; CHECK: task @ det.achd2
; CHECK-NOT:  task @ det.achd
; CHECK: Spindle @ det.cont may happen in parallel with:
; CHECK: task @ det.achd
; CHECK: task @ det.achd2
; CHECK: Spindle @ sync.continue1 may happen in parallel with:
; CHECK: task @ det.achd
; CHECK-NOT:  task @ det.achd2
; CHECK: Spindle @ det.cont4 may happen in parallel with:
; CHECK: task @ det.achd2
; CHECK: task @ det.achd
