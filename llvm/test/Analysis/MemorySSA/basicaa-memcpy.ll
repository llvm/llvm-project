; RUN: opt -disable-output -passes='print<memoryssa>' %s 2>&1 | FileCheck %s

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind

define void @source_clobber(ptr %a, ptr %b) {
; CHECK-LABEL: @source_clobber(
; CHECK-NEXT:  ; 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr %a, ptr %b, i64 128, i1 false)
; CHECK-NEXT:  ; MemoryUse(1)
; CHECK-NEXT:    [[X:%.*]] = load i8, ptr %b
; CHECK-NEXT:    ret void
;
  call void @llvm.memcpy.p0.p0.i64(ptr %a, ptr %b, i64 128, i1 false)
  %x = load i8, ptr %b
  ret void
}
