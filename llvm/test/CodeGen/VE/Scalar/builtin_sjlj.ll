; RUN: llc < %s -mtriple=ve | FileCheck %s
; RUN: llc < %s -mtriple=ve -relocation-model=pic | \
; RUN:     FileCheck %s -check-prefix=PIC

%struct.__jmp_buf_tag = type { [25 x i64], i64, [16 x i64] }

@buf = common global [1 x %struct.__jmp_buf_tag] zeroinitializer, align 8

; Function Attrs: noinline nounwind optnone
define signext i32 @t_setjmp() {
; CHECK-LABEL: t_setjmp:
; CHECK:       .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    st %s18, 48(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s19, 56(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s20, 64(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s21, 72(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s22, 80(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s23, 88(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s24, 96(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s25, 104(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s26, 112(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s27, 120(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s28, 128(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s29, 136(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s30, 144(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s31, 152(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s32, 160(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s33, 168(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    lea %s0, buf@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, buf@hi(, %s0)
; CHECK-NEXT:    st %s9, (, %s0)
; CHECK-NEXT:    st %s11, 16(, %s0)
; CHECK-NEXT:    lea %s1, .LBB{{[0-9]+}}_3@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, .LBB{{[0-9]+}}_3@hi(, %s1)
; CHECK-NEXT:    st %s1, 8(, %s0)
; CHECK-NEXT:    # EH_SJlJ_SETUP .LBB{{[0-9]+}}_3
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    lea %s0, 0
; CHECK-NEXT:    br.l.t .LBB{{[0-9]+}}_2
; CHECK-NEXT:  .LBB{{[0-9]+}}_3: # Block address taken
; CHECK-NEXT:    lea %s0, 1
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    ld %s33, 168(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s32, 160(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s31, 152(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s30, 144(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s29, 136(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s28, 128(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s27, 120(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s26, 112(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s25, 104(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s24, 96(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s23, 88(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s22, 80(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s21, 72(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s20, 64(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s19, 56(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s18, 48(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    or %s11, 0, %s9
;
; PIC-LABEL: t_setjmp:
; PIC:       # %bb.0:
; PIC-NEXT:    st %s9, (, %s11)
; PIC-NEXT:    st %s10, 8(, %s11)
; PIC-NEXT:    st %s15, 24(, %s11)
; PIC-NEXT:    st %s16, 32(, %s11)
; PIC-NEXT:    or %s9, 0, %s11
; PIC-NEXT:    lea %s11, -176(, %s11)
; PIC-NEXT:    brge.l %s11, %s8, .LBB0_5
; PIC-NEXT:  # %bb.4:
; PIC-NEXT:    ld %s61, 24(, %s14)
; PIC-NEXT:    or %s62, 0, %s0
; PIC-NEXT:    lea %s63, 315
; PIC-NEXT:    shm.l %s63, (%s61)
; PIC-NEXT:    shm.l %s8, 8(%s61)
; PIC-NEXT:    shm.l %s11, 16(%s61)
; PIC-NEXT:    monc
; PIC-NEXT:    or %s0, 0, %s62
; PIC-NEXT:  .LBB0_5:
; PIC-NEXT:    st %s18, 48(, %s9) # 8-byte Folded Spill
; PIC-NEXT:    st %s19, 56(, %s9) # 8-byte Folded Spill
; PIC-NEXT:    st %s20, 64(, %s9) # 8-byte Folded Spill
; PIC-NEXT:    st %s21, 72(, %s9) # 8-byte Folded Spill
; PIC-NEXT:    st %s22, 80(, %s9) # 8-byte Folded Spill
; PIC-NEXT:    st %s23, 88(, %s9) # 8-byte Folded Spill
; PIC-NEXT:    st %s24, 96(, %s9) # 8-byte Folded Spill
; PIC-NEXT:    st %s25, 104(, %s9) # 8-byte Folded Spill
; PIC-NEXT:    st %s26, 112(, %s9) # 8-byte Folded Spill
; PIC-NEXT:    st %s27, 120(, %s9) # 8-byte Folded Spill
; PIC-NEXT:    st %s28, 128(, %s9) # 8-byte Folded Spill
; PIC-NEXT:    st %s29, 136(, %s9) # 8-byte Folded Spill
; PIC-NEXT:    st %s30, 144(, %s9) # 8-byte Folded Spill
; PIC-NEXT:    st %s31, 152(, %s9) # 8-byte Folded Spill
; PIC-NEXT:    st %s32, 160(, %s9) # 8-byte Folded Spill
; PIC-NEXT:    st %s33, 168(, %s9) # 8-byte Folded Spill
; PIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; PIC-NEXT:    and %s15, %s15, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; PIC-NEXT:    lea %s0, buf@got_lo
; PIC-NEXT:    and %s0, %s0, (32)0
; PIC-NEXT:    lea.sl %s0, buf@got_hi(, %s0)
; PIC-NEXT:    ld %s0, (%s0, %s15)
; PIC-NEXT:    st %s9, (, %s0)
; PIC-NEXT:    st %s11, 16(, %s0)
; PIC-NEXT:    lea %s1, .LBB0_3@gotoff_lo
; PIC-NEXT:    and %s1, %s1, (32)0
; PIC-NEXT:    lea.sl %s1, .LBB0_3@gotoff_hi(%s1, %s15)
; PIC-NEXT:    st %s1, 8(, %s0)
; PIC-NEXT:    # EH_SJlJ_SETUP .LBB0_3
; PIC-NEXT:  # %bb.1:
; PIC-NEXT:    lea %s0, 0
; PIC-NEXT:    br.l.t .LBB0_2
; PIC-NEXT:  .LBB0_3: # Block address taken
; PIC-NEXT:    lea %s0, 1
; PIC-NEXT:  .LBB0_2:
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    ld %s33, 168(, %s9) # 8-byte Folded Reload
; PIC-NEXT:    ld %s32, 160(, %s9) # 8-byte Folded Reload
; PIC-NEXT:    ld %s31, 152(, %s9) # 8-byte Folded Reload
; PIC-NEXT:    ld %s30, 144(, %s9) # 8-byte Folded Reload
; PIC-NEXT:    ld %s29, 136(, %s9) # 8-byte Folded Reload
; PIC-NEXT:    ld %s28, 128(, %s9) # 8-byte Folded Reload
; PIC-NEXT:    ld %s27, 120(, %s9) # 8-byte Folded Reload
; PIC-NEXT:    ld %s26, 112(, %s9) # 8-byte Folded Reload
; PIC-NEXT:    ld %s25, 104(, %s9) # 8-byte Folded Reload
; PIC-NEXT:    ld %s24, 96(, %s9) # 8-byte Folded Reload
; PIC-NEXT:    ld %s23, 88(, %s9) # 8-byte Folded Reload
; PIC-NEXT:    ld %s22, 80(, %s9) # 8-byte Folded Reload
; PIC-NEXT:    ld %s21, 72(, %s9) # 8-byte Folded Reload
; PIC-NEXT:    ld %s20, 64(, %s9) # 8-byte Folded Reload
; PIC-NEXT:    ld %s19, 56(, %s9) # 8-byte Folded Reload
; PIC-NEXT:    ld %s18, 48(, %s9) # 8-byte Folded Reload
; PIC-NEXT:    or %s11, 0, %s9
; PIC-NEXT:    ld %s16, 32(, %s11)
; PIC-NEXT:    ld %s15, 24(, %s11)
; PIC-NEXT:    ld %s10, 8(, %s11)
; PIC-NEXT:    ld %s9, (, %s11)
; PIC-NEXT:    b.l.t (, %s10)
  %1 = call ptr @llvm.frameaddress(i32 0)
  store ptr %1, ptr @buf, align 8
  %2 = call ptr @llvm.stacksave()
  store ptr %2, ptr getelementptr inbounds (ptr, ptr @buf, i64 2), align 8
  %3 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %3
}

; Function Attrs: nounwind readnone
declare ptr @llvm.frameaddress(i32)

; Function Attrs: nounwind
declare ptr @llvm.stacksave()

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr)

; Function Attrs: noinline nounwind optnone
define void @t_longjmp() {
; CHECK-LABEL: t_longjmp:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, buf@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, buf@hi(, %s0)
; CHECK-NEXT:    ld %s9, (, %s0)
; CHECK-NEXT:    ld %s1, 8(, %s0)
; CHECK-NEXT:    or %s10, 0, %s0
; CHECK-NEXT:    ld %s11, 16(, %s0)
; CHECK-NEXT:    b.l.t (, %s1)
;
; PIC-LABEL: t_longjmp:
; PIC:       # %bb.0:
; PIC-NEXT:    st %s9, (, %s11)
; PIC-NEXT:    st %s10, 8(, %s11)
; PIC-NEXT:    st %s15, 24(, %s11)
; PIC-NEXT:    st %s16, 32(, %s11)
; PIC-NEXT:    or %s9, 0, %s11
; PIC-NEXT:    lea %s11, -176(, %s11)
; PIC-NEXT:    brge.l.t %s11, %s8, .LBB1_2
; PIC-NEXT:  # %bb.1:
; PIC-NEXT:    ld %s61, 24(, %s14)
; PIC-NEXT:    or %s62, 0, %s0
; PIC-NEXT:    lea %s63, 315
; PIC-NEXT:    shm.l %s63, (%s61)
; PIC-NEXT:    shm.l %s8, 8(%s61)
; PIC-NEXT:    shm.l %s11, 16(%s61)
; PIC-NEXT:    monc
; PIC-NEXT:    or %s0, 0, %s62
; PIC-NEXT:  .LBB1_2:
; PIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; PIC-NEXT:    and %s15, %s15, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; PIC-NEXT:    lea %s0, buf@got_lo
; PIC-NEXT:    and %s0, %s0, (32)0
; PIC-NEXT:    lea.sl %s0, buf@got_hi(, %s0)
; PIC-NEXT:    ld %s0, (%s0, %s15)
; PIC-NEXT:    ld %s9, (, %s0)
; PIC-NEXT:    ld %s1, 8(, %s0)
; PIC-NEXT:    or %s10, 0, %s0
; PIC-NEXT:    ld %s11, 16(, %s0)
; PIC-NEXT:    b.l.t (, %s1)
  call void @llvm.eh.sjlj.longjmp(ptr @buf)
  unreachable
                                                  ; No predecessors!
  ret void
}

; Function Attrs: noreturn nounwind
declare void @llvm.eh.sjlj.longjmp(ptr)

