; RUN: llc < %s -mtriple=ve | FileCheck %s
; RUN: llc < %s -mtriple=ve -relocation-model=pic \
; RUN:     | FileCheck %s --check-prefix=PIC

;;; Tests for lea instruction and its optimizations

%struct.buffer = type { i64, [1 x i8] }

@data = internal global i8 0, align 1
@buf = internal global %struct.buffer zeroinitializer, align 8

; Function Attrs: norecurse nounwind readnone
define nonnull i8* @lea_basic() {
; CHECK-LABEL: lea_basic:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, data@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, data@hi(, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
;
; PIC-LABEL: lea_basic:
; PIC:       # %bb.0:
; PIC-NEXT:    st %s15, 24(, %s11)
; PIC-NEXT:    st %s16, 32(, %s11)
; PIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; PIC-NEXT:    and %s15, %s15, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; PIC-NEXT:    lea %s0, data@gotoff_lo
; PIC-NEXT:    and %s0, %s0, (32)0
; PIC-NEXT:    lea.sl %s0, data@gotoff_hi(%s0, %s15)
; PIC-NEXT:    ld %s16, 32(, %s11)
; PIC-NEXT:    ld %s15, 24(, %s11)
; PIC-NEXT:    b.l.t (, %s10)
  ret i8* @data
}

; Function Attrs: norecurse nounwind readnone
define i8* @lea_offset() {
; CHECK-LABEL: lea_offset:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, buf@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, buf@hi(8, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
;
; PIC-LABEL: lea_offset:
; PIC:       # %bb.0:
; PIC-NEXT:    st %s15, 24(, %s11)
; PIC-NEXT:    st %s16, 32(, %s11)
; PIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; PIC-NEXT:    and %s15, %s15, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; PIC-NEXT:    lea %s0, buf@gotoff_lo
; PIC-NEXT:    and %s0, %s0, (32)0
; PIC-NEXT:    lea.sl %s0, buf@gotoff_hi(, %s0)
; PIC-NEXT:    lea %s0, 8(%s0, %s15)
; PIC-NEXT:    ld %s16, 32(, %s11)
; PIC-NEXT:    ld %s15, 24(, %s11)
; PIC-NEXT:    b.l.t (, %s10)
  ret i8* getelementptr inbounds (%struct.buffer, %struct.buffer* @buf, i64 0, i32 1, i64 0)
}
