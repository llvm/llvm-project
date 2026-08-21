; NOTE: Do not autogenerate. This test is about four frame offsets, and full
; generated assertions bury them in the rest of the function.
; RUN: llc -mtriple=aarch64-unknown-windows-msvc -O2 < %s | FileCheck %s

; The async context slot has to sit directly below the frame record. If it is
; placed above the record instead, MachineFrameInfo's view of the callee-save
; area disagrees with the prologue by 8 bytes, and PEI's scavenger fills the
; resulting hole with a live local that then shares an address with the saved
; caller x29.
;
; Check the record, the context slot below it, and that the scavenged spill goes
; below the callee-save area rather than into it. Before the fix the last one
; was `str x7, [x29]`, aliasing the saved x29 stored at sp+88.

declare ptr @llvm.swift.async.context.addr() nounwind
declare swiftcc void @swift_task_dealloc()

define swifttailcc void @test(ptr %ctx, ptr %vw0, ptr %vw1, ptr %vw2, ptr %vw3, ptr %obj0, ptr %obj1, ptr %obj2, ptr %obj3, ptr %obj4) {
; CHECK-LABEL: test:
; CHECK:      stp x29, x30, [sp, #88]
; CHECK:      str xzr, [sp, #80]
; CHECK-NEXT: .seh_nop
; CHECK:      add x29, sp, #88
; CHECK:      str x7, [sp, #8]
entryresume.0:
  %ctxaddr = tail call ptr @llvm.swift.async.context.addr()
  %reloaded = load ptr, ptr null, align 8
  call swiftcc void @swift_task_dealloc()
  %destroy0 = load ptr, ptr %ctx, align 8
  tail call void %destroy0(ptr %reloaded, ptr %obj4)
  %destroy1 = load ptr, ptr %obj1, align 8
  tail call void %destroy1(ptr %vw2, ptr null)
  %destroy2 = load ptr, ptr %obj3, align 8
  tail call void %destroy2(ptr %vw1, ptr %obj2)
  %destroy3 = load ptr, ptr %vw3, align 8
  tail call void %destroy3(ptr %vw0, ptr %obj0)
  ret void
}
