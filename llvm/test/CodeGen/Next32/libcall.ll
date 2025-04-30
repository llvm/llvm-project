; RUN: llc -mtriple=next32 -mcpu=next32gen2 -debug-only=isel 2>&1 < %s | FileCheck %s

; REQUIRES: asserts

define dso_local i64 @libcall1(i64 noundef %0) {
; Check whether we've chained the generated libcall correctly
; CHECK-LABEL: ===== Instruction selection ends:
; CHECK: t0: ch,glue = EntryToken
; CHECK: t[[A:[0-9]+]]: ch = FEEDER_ARGS {{.*}}, t0
; CHECK: t[[B:[0-9]+]]: ch,glue = CALL TargetExternalSymbol:i32'__lshrdi3', {{.*}}, t[[A]]
; CHECK: t[[C:[0-9]+]]: ch,glue = SYM_INSTR MCSymbol:ch, t[[B]], t[[B]]:1
; CHECK: t[[D:[0-9]+]]: ch = TokenFactor t[[A]], t[[C]]
; CHECK: t{{.*}}: ch = RET {{.*}}, t[[D]]
  %2 = lshr i64 %0, 3
  ret i64 %2
}

define dso_local i64 @libcall2(ptr %0, i64 %1) {
; Check whether the memory operation is moved before the libcall
; CHECK-LABEL: ===== Instruction selection ends:
; CHECK: t0: ch,glue = EntryToken
; CHECK: t[[A:[0-9]+]]: ch = FEEDER_ARGS {{.*}}, t0
; CHECK: t{{.*}}: i32,ch = GMEMWRITE {{.*}}, t[[A]]
  %3 = lshr i64 %1, 3
  store i64 %1, ptr %0, align 8
  ret i64 %3
}
