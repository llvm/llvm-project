; RUN: opt -disable-output -basic-aa -enable-new-pm=0 -print-memoryssa %s 2>&1 | FileCheck %s

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind

define void @source_clobber(i8* %a, i8* %b) {
; CHECK-LABEL: @source_clobber(
; CHECK-NEXT:  ; 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT:    call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* %b, i64 128, i1 false)
; CHECK-NEXT:  ; MemoryUse(1) MayAlias
; CHECK-NEXT:    [[X:%.*]] = load i8, i8* %b
; CHECK-NEXT:    ret void
;
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* %b, i64 128, i1 false)
  %x = load i8, i8* %b
  ret void
}
