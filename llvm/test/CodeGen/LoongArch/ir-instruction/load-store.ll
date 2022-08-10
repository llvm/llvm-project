; RUN: llc --mtriple=loongarch32 --mattr=+d --relocation-model=static < %s | FileCheck %s --check-prefixes=ALL,LA32NOPIC,LA32
; RUN: llc --mtriple=loongarch32 --mattr=+d --relocation-model=pic < %s | FileCheck %s --check-prefixes=ALL,LA32PIC,LA32
; RUN: llc --mtriple=loongarch64 --mattr=+d --relocation-model=static < %s | FileCheck %s --check-prefixes=ALL,LA64NOPIC,LA64
; RUN: llc --mtriple=loongarch64 --mattr=+d --relocation-model=pic < %s | FileCheck %s --check-prefixes=ALL,LA64PIC,LA64

;; Check load from and store to global variables.
@G = dso_local global i32 zeroinitializer, align 4
@arr = dso_local global [10 x i32] zeroinitializer, align 4

define i32 @load_store_global() nounwind {
; ALL-LABEL:      load_store_global:
; ALL:            # %bb.0:

; LA32NOPIC-NEXT:   pcalau12i $a0, G
; LA32NOPIC-NEXT:   addi.w $a1, $a0, G
; LA32PIC-NEXT:     pcalau12i $a0, .LG$local
; LA32PIC-NEXT:     addi.w $a1, $a0, .LG$local
; LA32-NEXT:        ld.w $a0, $a1, 0
; LA32-NEXT:        addi.w $a0, $a0, 1
; LA32-NEXT:        st.w $a0, $a1, 0

; LA64NOPIC-NEXT:   pcalau12i $a0, G
; LA64NOPIC-NEXT:   addi.d $a1, $a0, G
; LA64PIC-NEXT:     pcalau12i $a0, .LG$local
; LA64PIC-NEXT:     addi.d $a1, $a0, .LG$local
; LA64-NEXT:        ld.w $a0, $a1, 0
; LA64-NEXT:        addi.d $a0, $a0, 1
; LA64-NEXT:        st.w $a0, $a1, 0

; ALL-NEXT:         jirl $zero, $ra, 0

  %v = load i32, ptr @G
  %sum = add i32 %v, 1
  store i32 %sum, ptr @G
  ret i32 %sum
}

define i32 @load_store_global_array(i32 %a) nounwind {
; ALL-LABEL: load_store_global_array:
; ALL:       # %bb.0:

; LA32NOPIC-NEXT:   pcalau12i $a1, arr
; LA32NOPIC-NEXT:   addi.w $a2, $a1, arr
; LA32PIC-NEXT:     pcalau12i $a1, .Larr$local
; LA32PIC-NEXT:     addi.w $a2, $a1, .Larr$local
; LA32-NEXT:        ld.w $a1, $a2, 0
; LA32-NEXT:        st.w $a0, $a2, 0
; LA32NOPIC-NEXT:   ld.w $a3, $a2, 0
; LA32NOPIC-NEXT:   st.w $a0, $a2, 0
; LA32PIC-NEXT:     ld.w $a3, $a2, 36
; LA32PIC-NEXT:     st.w $a0, $a2, 36

; LA64NOPIC-NEXT:   pcalau12i $a1, arr
; LA64NOPIC-NEXT:   addi.d $a2, $a1, arr
; LA64PIC-NEXT:     pcalau12i $a1, .Larr$local
; LA64PIC-NEXT:     addi.d $a2, $a1, .Larr$local
; LA64-NEXT:        ld.w $a1, $a2, 0
; LA64-NEXT:        st.w $a0, $a2, 0
; LA64NOPIC-NEXT:   ld.w $a3, $a2, 0
; LA64NOPIC-NEXT:   st.w $a0, $a2, 0
; LA64PIC-NEXT:     ld.w $a3, $a2, 36
; LA64PIC-NEXT:     st.w $a0, $a2, 36

; ALL-NEXT:         move $a0, $a1
; ALL-NEXT:         jirl $zero, $ra, 0

  %1 = load volatile i32, ptr @arr, align 4
  store i32 %a, ptr @arr, align 4
  %2 = getelementptr [10 x i32], ptr @arr, i32 0, i32 9
  %3 = load volatile i32, ptr %2, align 4
  store i32 %a, ptr %2, align 4
  ret i32 %1
}

;; Check indexed and unindexed, sext, zext and anyext loads.

define i64 @ld_b(ptr %a) nounwind {
; LA32-LABEL: ld_b:
; LA32:       # %bb.0:
; LA32-NEXT:    ld.b $a1, $a0, 0
; LA32-NEXT:    ld.b $a0, $a0, 1
; LA32-NEXT:    srai.w $a1, $a0, 31
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: ld_b:
; LA64:       # %bb.0:
; LA64-NEXT:    ld.b $a1, $a0, 0
; LA64-NEXT:    ld.b $a0, $a0, 1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i8, ptr %a, i64 1
  %2 = load i8, ptr %1
  %3 = sext i8 %2 to i64
  %4 = load volatile i8, ptr %a
  ret i64 %3
}

define i64 @ld_h(ptr %a) nounwind {
; LA32-LABEL: ld_h:
; LA32:       # %bb.0:
; LA32-NEXT:    ld.h $a1, $a0, 0
; LA32-NEXT:    ld.h $a0, $a0, 4
; LA32-NEXT:    srai.w $a1, $a0, 31
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: ld_h:
; LA64:       # %bb.0:
; LA64-NEXT:    ld.h $a1, $a0, 0
; LA64-NEXT:    ld.h $a0, $a0, 4
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i16, ptr %a, i64 2
  %2 = load i16, ptr %1
  %3 = sext i16 %2 to i64
  %4 = load volatile i16, ptr %a
  ret i64 %3
}

define i64 @ld_w(ptr %a) nounwind {
; LA32-LABEL: ld_w:
; LA32:       # %bb.0:
; LA32-NEXT:    ld.w $a1, $a0, 0
; LA32-NEXT:    ld.w $a0, $a0, 12
; LA32-NEXT:    srai.w $a1, $a0, 31
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: ld_w:
; LA64:       # %bb.0:
; LA64-NEXT:    ld.w $a1, $a0, 0
; LA64-NEXT:    ld.w $a0, $a0, 12
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i32, ptr %a, i64 3
  %2 = load i32, ptr %1
  %3 = sext i32 %2 to i64
  %4 = load volatile i32, ptr %a
  ret i64 %3
}

define i64 @ld_d(ptr %a) nounwind {
; LA32-LABEL: ld_d:
; LA32:       # %bb.0:
; LA32-NEXT:    ld.w $a1, $a0, 4
; LA32-NEXT:    ld.w $a1, $a0, 0
; LA32-NEXT:    ld.w $a1, $a0, 28
; LA32-NEXT:    ld.w $a0, $a0, 24
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: ld_d:
; LA64:       # %bb.0:
; LA64-NEXT:    ld.d $a1, $a0, 0
; LA64-NEXT:    ld.d $a0, $a0, 24
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i64, ptr %a, i64 3
  %2 = load i64, ptr %1
  %3 = load volatile i64, ptr %a
  ret i64 %2
}

define i64 @ld_bu(ptr %a) nounwind {
; LA32-LABEL: ld_bu:
; LA32:       # %bb.0:
; LA32-NEXT:    ld.bu $a1, $a0, 0
; LA32-NEXT:    ld.bu $a2, $a0, 4
; LA32-NEXT:    add.w $a0, $a2, $a1
; LA32-NEXT:    sltu $a1, $a0, $a2
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: ld_bu:
; LA64:       # %bb.0:
; LA64-NEXT:    ld.bu $a1, $a0, 0
; LA64-NEXT:    ld.bu $a0, $a0, 4
; LA64-NEXT:    add.d $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i8, ptr %a, i64 4
  %2 = load i8, ptr %1
  %3 = zext i8 %2 to i64
  %4 = load volatile i8, ptr %a
  %5 = zext i8 %4 to i64
  %6 = add i64 %3, %5
  ret i64 %6
}

define i64 @ld_hu(ptr %a) nounwind {
; LA32-LABEL: ld_hu:
; LA32:       # %bb.0:
; LA32-NEXT:    ld.hu $a1, $a0, 0
; LA32-NEXT:    ld.hu $a2, $a0, 10
; LA32-NEXT:    add.w $a0, $a2, $a1
; LA32-NEXT:    sltu $a1, $a0, $a2
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: ld_hu:
; LA64:       # %bb.0:
; LA64-NEXT:    ld.hu $a1, $a0, 0
; LA64-NEXT:    ld.hu $a0, $a0, 10
; LA64-NEXT:    add.d $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i16, ptr %a, i64 5
  %2 = load i16, ptr %1
  %3 = zext i16 %2 to i64
  %4 = load volatile i16, ptr %a
  %5 = zext i16 %4 to i64
  %6 = add i64 %3, %5
  ret i64 %6
}

define i64 @ld_wu(ptr %a) nounwind {
; LA32-LABEL: ld_wu:
; LA32:       # %bb.0:
; LA32-NEXT:    ld.w $a1, $a0, 0
; LA32-NEXT:    ld.w $a2, $a0, 20
; LA32-NEXT:    add.w $a0, $a2, $a1
; LA32-NEXT:    sltu $a1, $a0, $a2
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: ld_wu:
; LA64:       # %bb.0:
; LA64-NEXT:    ld.wu $a1, $a0, 0
; LA64-NEXT:    ld.wu $a0, $a0, 20
; LA64-NEXT:    add.d $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i32, ptr %a, i64 5
  %2 = load i32, ptr %1
  %3 = zext i32 %2 to i64
  %4 = load volatile i32, ptr %a
  %5 = zext i32 %4 to i64
  %6 = add i64 %3, %5
  ret i64 %6
}

define i64 @ldx_b(ptr %a, i64 %idx) nounwind {
; LA32-LABEL: ldx_b:
; LA32:       # %bb.0:
; LA32-NEXT:    add.w $a1, $a0, $a1
; LA32-NEXT:    ld.b $a2, $a1, 0
; LA32-NEXT:    ld.b $a0, $a0, 0
; LA32-NEXT:    srai.w $a1, $a2, 31
; LA32-NEXT:    move $a0, $a2
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: ldx_b:
; LA64:       # %bb.0:
; LA64-NEXT:    ldx.b $a1, $a0, $a1
; LA64-NEXT:    ld.b $a0, $a0, 0
; LA64-NEXT:    move $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i8, ptr %a, i64 %idx
  %2 = load i8, ptr %1
  %3 = sext i8 %2 to i64
  %4 = load volatile i8, ptr %a
  ret i64 %3
}

define i64 @ldx_h(ptr %a, i64 %idx) nounwind {
; LA32-LABEL: ldx_h:
; LA32:       # %bb.0:
; LA32-NEXT:    slli.w $a1, $a1, 1
; LA32-NEXT:    add.w $a1, $a0, $a1
; LA32-NEXT:    ld.h $a2, $a1, 0
; LA32-NEXT:    ld.h $a0, $a0, 0
; LA32-NEXT:    srai.w $a1, $a2, 31
; LA32-NEXT:    move $a0, $a2
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: ldx_h:
; LA64:       # %bb.0:
; LA64-NEXT:    slli.d $a1, $a1, 1
; LA64-NEXT:    ldx.h $a1, $a0, $a1
; LA64-NEXT:    ld.h $a0, $a0, 0
; LA64-NEXT:    move $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i16, ptr %a, i64 %idx
  %2 = load i16, ptr %1
  %3 = sext i16 %2 to i64
  %4 = load volatile i16, ptr %a
  ret i64 %3
}

define i64 @ldx_w(ptr %a, i64 %idx) nounwind {
; LA32-LABEL: ldx_w:
; LA32:       # %bb.0:
; LA32-NEXT:    slli.w $a1, $a1, 2
; LA32-NEXT:    add.w $a1, $a0, $a1
; LA32-NEXT:    ld.w $a2, $a1, 0
; LA32-NEXT:    ld.w $a0, $a0, 0
; LA32-NEXT:    srai.w $a1, $a2, 31
; LA32-NEXT:    move $a0, $a2
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: ldx_w:
; LA64:       # %bb.0:
; LA64-NEXT:    slli.d $a1, $a1, 2
; LA64-NEXT:    ldx.w $a1, $a0, $a1
; LA64-NEXT:    ld.w $a0, $a0, 0
; LA64-NEXT:    move $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i32, ptr %a, i64 %idx
  %2 = load i32, ptr %1
  %3 = sext i32 %2 to i64
  %4 = load volatile i32, ptr %a
  ret i64 %3
}

define i64 @ldx_d(ptr %a, i64 %idx) nounwind {
; LA32-LABEL: ldx_d:
; LA32:       # %bb.0:
; LA32-NEXT:    slli.w $a1, $a1, 3
; LA32-NEXT:    add.w $a1, $a0, $a1
; LA32-NEXT:    ld.w $a2, $a1, 0
; LA32-NEXT:    ld.w $a3, $a0, 0
; LA32-NEXT:    ld.w $a1, $a1, 4
; LA32-NEXT:    ld.w $a0, $a0, 4
; LA32-NEXT:    move $a0, $a2
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: ldx_d:
; LA64:       # %bb.0:
; LA64-NEXT:    slli.d $a1, $a1, 3
; LA64-NEXT:    ldx.d $a1, $a0, $a1
; LA64-NEXT:    ld.d $a0, $a0, 0
; LA64-NEXT:    move $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i64, ptr %a, i64 %idx
  %2 = load i64, ptr %1
  %3 = load volatile i64, ptr %a
  ret i64 %2
}

define i64 @ldx_bu(ptr %a, i64 %idx) nounwind {
; LA32-LABEL: ldx_bu:
; LA32:       # %bb.0:
; LA32-NEXT:    add.w $a1, $a0, $a1
; LA32-NEXT:    ld.bu $a1, $a1, 0
; LA32-NEXT:    ld.bu $a0, $a0, 0
; LA32-NEXT:    add.w $a0, $a1, $a0
; LA32-NEXT:    sltu $a1, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: ldx_bu:
; LA64:       # %bb.0:
; LA64-NEXT:    ldx.bu $a1, $a0, $a1
; LA64-NEXT:    ld.bu $a0, $a0, 0
; LA64-NEXT:    add.d $a0, $a1, $a0
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i8, ptr %a, i64 %idx
  %2 = load i8, ptr %1
  %3 = zext i8 %2 to i64
  %4 = load volatile i8, ptr %a
  %5 = zext i8 %4 to i64
  %6 = add i64 %3, %5
  ret i64 %6
}

define i64 @ldx_hu(ptr %a, i64 %idx) nounwind {
; LA32-LABEL: ldx_hu:
; LA32:       # %bb.0:
; LA32-NEXT:    slli.w $a1, $a1, 1
; LA32-NEXT:    add.w $a1, $a0, $a1
; LA32-NEXT:    ld.hu $a1, $a1, 0
; LA32-NEXT:    ld.hu $a0, $a0, 0
; LA32-NEXT:    add.w $a0, $a1, $a0
; LA32-NEXT:    sltu $a1, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: ldx_hu:
; LA64:       # %bb.0:
; LA64-NEXT:    slli.d $a1, $a1, 1
; LA64-NEXT:    ldx.hu $a1, $a0, $a1
; LA64-NEXT:    ld.hu $a0, $a0, 0
; LA64-NEXT:    add.d $a0, $a1, $a0
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i16, ptr %a, i64 %idx
  %2 = load i16, ptr %1
  %3 = zext i16 %2 to i64
  %4 = load volatile i16, ptr %a
  %5 = zext i16 %4 to i64
  %6 = add i64 %3, %5
  ret i64 %6
}

define i64 @ldx_wu(ptr %a, i64 %idx) nounwind {
; LA32-LABEL: ldx_wu:
; LA32:       # %bb.0:
; LA32-NEXT:    slli.w $a1, $a1, 2
; LA32-NEXT:    add.w $a1, $a0, $a1
; LA32-NEXT:    ld.w $a1, $a1, 0
; LA32-NEXT:    ld.w $a0, $a0, 0
; LA32-NEXT:    add.w $a0, $a1, $a0
; LA32-NEXT:    sltu $a1, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: ldx_wu:
; LA64:       # %bb.0:
; LA64-NEXT:    slli.d $a1, $a1, 2
; LA64-NEXT:    ldx.wu $a1, $a0, $a1
; LA64-NEXT:    ld.wu $a0, $a0, 0
; LA64-NEXT:    add.d $a0, $a1, $a0
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i32, ptr %a, i64 %idx
  %2 = load i32, ptr %1
  %3 = zext i32 %2 to i64
  %4 = load volatile i32, ptr %a
  %5 = zext i32 %4 to i64
  %6 = add i64 %3, %5
  ret i64 %6
}

;; Check indexed and unindexed stores.

define void @st_b(ptr %a, i8 %b) nounwind {
; ALL-LABEL: st_b:
; ALL:       # %bb.0:
; ALL-NEXT:    st.b $a1, $a0, 6
; ALL-NEXT:    st.b $a1, $a0, 0
; ALL-NEXT:    jirl $zero, $ra, 0
  store i8 %b, ptr %a
  %1 = getelementptr i8, ptr %a, i64 6
  store i8 %b, ptr %1
  ret void
}

define void @st_h(ptr %a, i16 %b) nounwind {
; ALL-LABEL: st_h:
; ALL:       # %bb.0:
; ALL-NEXT:    st.h $a1, $a0, 14
; ALL-NEXT:    st.h $a1, $a0, 0
; ALL-NEXT:    jirl $zero, $ra, 0
  store i16 %b, ptr %a
  %1 = getelementptr i16, ptr %a, i64 7
  store i16 %b, ptr %1
  ret void
}

define void @st_w(ptr %a, i32 %b) nounwind {
; ALL-LABEL: st_w:
; ALL:       # %bb.0:
; ALL-NEXT:    st.w $a1, $a0, 28
; ALL-NEXT:    st.w $a1, $a0, 0
; ALL-NEXT:    jirl $zero, $ra, 0
  store i32 %b, ptr %a
  %1 = getelementptr i32, ptr %a, i64 7
  store i32 %b, ptr %1
  ret void
}

define void @st_d(ptr %a, i64 %b) nounwind {
; LA32-LABEL: st_d:
; LA32:       # %bb.0:
; LA32-NEXT:    st.w $a2, $a0, 68
; LA32-NEXT:    st.w $a2, $a0, 4
; LA32-NEXT:    st.w $a1, $a0, 64
; LA32-NEXT:    st.w $a1, $a0, 0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: st_d:
; LA64:       # %bb.0:
; LA64-NEXT:    st.d $a1, $a0, 64
; LA64-NEXT:    st.d $a1, $a0, 0
; LA64-NEXT:    jirl $zero, $ra, 0
  store i64 %b, ptr %a
  %1 = getelementptr i64, ptr %a, i64 8
  store i64 %b, ptr %1
  ret void
}

define void @stx_b(ptr %dst, i64 %idx, i8 %val) nounwind {
; LA32-LABEL: stx_b:
; LA32:       # %bb.0:
; LA32-NEXT:    add.w $a0, $a0, $a1
; LA32-NEXT:    st.b $a3, $a0, 0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: stx_b:
; LA64:       # %bb.0:
; LA64-NEXT:    stx.b $a2, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i8, ptr %dst, i64 %idx
  store i8 %val, ptr %1
  ret void
}

define void @stx_h(ptr %dst, i64 %idx, i16 %val) nounwind {
; LA32-LABEL: stx_h:
; LA32:       # %bb.0:
; LA32-NEXT:    slli.w $a1, $a1, 1
; LA32-NEXT:    add.w $a0, $a0, $a1
; LA32-NEXT:    st.h $a3, $a0, 0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: stx_h:
; LA64:       # %bb.0:
; LA64-NEXT:    slli.d $a1, $a1, 1
; LA64-NEXT:    stx.h $a2, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i16, ptr %dst, i64 %idx
  store i16 %val, ptr %1
  ret void
}

define void @stx_w(ptr %dst, i64 %idx, i32 %val) nounwind {
; LA32-LABEL: stx_w:
; LA32:       # %bb.0:
; LA32-NEXT:    slli.w $a1, $a1, 2
; LA32-NEXT:    add.w $a0, $a0, $a1
; LA32-NEXT:    st.w $a3, $a0, 0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: stx_w:
; LA64:       # %bb.0:
; LA64-NEXT:    slli.d $a1, $a1, 2
; LA64-NEXT:    stx.w $a2, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i32, ptr %dst, i64 %idx
  store i32 %val, ptr %1
  ret void
}

define void @stx_d(ptr %dst, i64 %idx, i64 %val) nounwind {
; LA32-LABEL: stx_d:
; LA32:       # %bb.0:
; LA32-NEXT:    slli.w $a1, $a1, 3
; LA32-NEXT:    add.w $a0, $a0, $a1
; LA32-NEXT:    st.w $a4, $a0, 4
; LA32-NEXT:    st.w $a3, $a0, 0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: stx_d:
; LA64:       # %bb.0:
; LA64-NEXT:    slli.d $a1, $a1, 3
; LA64-NEXT:    stx.d $a2, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i64, ptr %dst, i64 %idx
  store i64 %val, ptr %1
  ret void
}

;; Check load from and store to an i1 location.
define i64 @load_sext_zext_anyext_i1(ptr %a) nounwind {
  ;; sextload i1
; LA32-LABEL: load_sext_zext_anyext_i1:
; LA32:       # %bb.0:
; LA32-NEXT:    ld.b $a1, $a0, 0
; LA32-NEXT:    ld.bu $a1, $a0, 1
; LA32-NEXT:    ld.bu $a2, $a0, 2
; LA32-NEXT:    sub.w $a0, $a2, $a1
; LA32-NEXT:    sltu $a1, $a2, $a1
; LA32-NEXT:    sub.w $a1, $zero, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: load_sext_zext_anyext_i1:
; LA64:       # %bb.0:
; LA64-NEXT:    ld.b $a1, $a0, 0
; LA64-NEXT:    ld.bu $a1, $a0, 1
; LA64-NEXT:    ld.bu $a0, $a0, 2
; LA64-NEXT:    sub.d $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i1, ptr %a, i64 1
  %2 = load i1, ptr %1
  %3 = sext i1 %2 to i64
  ;; zextload i1
  %4 = getelementptr i1, ptr %a, i64 2
  %5 = load i1, ptr %4
  %6 = zext i1 %5 to i64
  %7 = add i64 %3, %6
  ;; extload i1 (anyext). Produced as the load is unused.
  %8 = load volatile i1, ptr %a
  ret i64 %7
}

define i16 @load_sext_zext_anyext_i1_i16(ptr %a) nounwind {
  ;; sextload i1
; LA32-LABEL: load_sext_zext_anyext_i1_i16:
; LA32:       # %bb.0:
; LA32-NEXT:    ld.b $a1, $a0, 0
; LA32-NEXT:    ld.bu $a1, $a0, 1
; LA32-NEXT:    ld.bu $a0, $a0, 2
; LA32-NEXT:    sub.w $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: load_sext_zext_anyext_i1_i16:
; LA64:       # %bb.0:
; LA64-NEXT:    ld.b $a1, $a0, 0
; LA64-NEXT:    ld.bu $a1, $a0, 1
; LA64-NEXT:    ld.bu $a0, $a0, 2
; LA64-NEXT:    sub.d $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr i1, ptr %a, i64 1
  %2 = load i1, ptr %1
  %3 = sext i1 %2 to i16
  ;; zextload i1
  %4 = getelementptr i1, ptr %a, i64 2
  %5 = load i1, ptr %4
  %6 = zext i1 %5 to i16
  %7 = add i16 %3, %6
  ;; extload i1 (anyext). Produced as the load is unused.
  %8 = load volatile i1, ptr %a
  ret i16 %7
}

define i64 @ld_sd_constant(i64 %a) nounwind {
; LA32-LABEL: ld_sd_constant:
; LA32:       # %bb.0:
; LA32-NEXT:    lu12i.w $a3, -136485
; LA32-NEXT:    ori $a4, $a3, 3823
; LA32-NEXT:    ld.w $a2, $a4, 0
; LA32-NEXT:    st.w $a0, $a4, 0
; LA32-NEXT:    ori $a0, $a3, 3827
; LA32-NEXT:    ld.w $a3, $a0, 0
; LA32-NEXT:    st.w $a1, $a0, 0
; LA32-NEXT:    move $a0, $a2
; LA32-NEXT:    move $a1, $a3
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: ld_sd_constant:
; LA64:       # %bb.0:
; LA64-NEXT:    lu12i.w $a1, -136485
; LA64-NEXT:    ori $a1, $a1, 3823
; LA64-NEXT:    lu32i.d $a1, -147729
; LA64-NEXT:    lu52i.d $a2, $a1, -534
; LA64-NEXT:    ld.d $a1, $a2, 0
; LA64-NEXT:    st.d $a0, $a2, 0
; LA64-NEXT:    move $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = inttoptr i64 16045690984833335023 to ptr
  %2 = load volatile i64, ptr %1
  store i64 %a, ptr %1
  ret i64 %2
}

;; Check load from and store to a float location.
define float @load_store_float(ptr %a, float %b) nounwind {
; ALL-LABEL: load_store_float:
; ALL:       # %bb.0:
; ALL-NEXT:    fld.s $fa1, $a0, 4
; ALL-NEXT:    fst.s $fa0, $a0, 4
; ALL-NEXT:    fmov.s $fa0, $fa1
; ALL-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr float, ptr %a, i64 1
  %2 = load float, ptr %1
  store float %b, ptr %1
  ret float %2
}

;; Check load from and store to a double location.
define double @load_store_double(ptr %a, double %b) nounwind {
; ALL-LABEL: load_store_double:
; ALL:       # %bb.0:
; ALL-NEXT:    fld.d $fa1, $a0, 8
; ALL-NEXT:    fst.d $fa0, $a0, 8
; ALL-NEXT:    fmov.d $fa0, $fa1
; ALL-NEXT:    jirl $zero, $ra, 0
  %1 = getelementptr double, ptr %a, i64 1
  %2 = load double, ptr %1
  store double %b, ptr %1
  ret double %2
}
