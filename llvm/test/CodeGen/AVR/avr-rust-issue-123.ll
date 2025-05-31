; RUN: llc -O1 < %s -mtriple=avr | FileCheck %s

; This test ensures that the Select8/Select16 expansion
; pass inserts an unconditional branch to the previous adjacent
; basic block when inserting new basic blocks when the
; prior block has a fallthrough.
;
; Before this bug was fixed, Select8/Select16 expansion
; would leave a dangling fallthrough to an undefined block.
;
; The BranchFolding pass would later rearrange the basic
; blocks based on predecessor/successor list assumptions
; which were made incorrect due to the invalid Select
; expansion.

; More information in
; https://github.com/avr-rust/rust/issues/123.

%UInt8 = type <{ i8 }>
%UInt32 = type <{ i32 }>
%Sb = type <{ i1 }>

@delayFactor = hidden global %UInt8 zeroinitializer, align 1
@delay = hidden global %UInt32 zeroinitializer, align 4
@flag = hidden global %Sb zeroinitializer, align 1

declare void @eeprom_write(i16, i8)

; CHECK-LABEL: update_register
define hidden void @update_register(i8 %arg, i8 %arg1) {
entry:
  ; CHECK: push [[PRELUDER:r[0-9]+]]
  ; CHECK: cpi  r24, 7
  switch i8 %arg, label %bb7 [
    i8 6, label %bb
    i8 7, label %bb6
  ]

; CHECK-NOT: ret
bb:                                               ; preds = %entry
  %tmp = icmp ugt i8 %arg1, 90
  %tmp2 = icmp ult i8 %arg1, 5
  %. = select i1 %tmp2, i8 5, i8 %arg1
  %tmp3 = select i1 %tmp, i8 90, i8 %.
  ; CHECK: sts delayFactor, r{{[0-9]+}}
  store i8 %tmp3, ptr @delayFactor, align 1
  %tmp4 = zext i8 %tmp3 to i32
  %tmp5 = mul nuw nsw i32 %tmp4, 100
  ; CHECK:      sts  delay+3, r{{[0-9]+}}
  ; CHECK-NEXT: sts  delay+2, r{{[0-9]+}}
  ; CHECK-NEXT: sts  delay+1, r{{[0-9]+}}
  ; CHECK-NEXT: sts  delay, r{{[0-9]+}}
  store i32 %tmp5, ptr @delay, align 4
  tail call void @eeprom_write(i16 34, i8 %tmp3)
  br label %bb7

bb6:                                              ; preds = %entry
  %not. = icmp ne i8 %arg1, 0
  %.2 = zext i1 %not. to i8
  store i1 %not., ptr @flag, align 1

  ; CHECK: call eeprom_write
  tail call void @eeprom_write(i16 35, i8 %.2)
  br label %bb7

  ; CHECK: LBB0_{{[0-9]+}}
  ; CHECK: pop [[PRELUDER]]
  ; CHECK-NEXT: ret
bb7:                                              ; preds = %bb6, %bb, %entry
  ret void
}
; CHECK-NOT: LBB0_{{[0-9]+}}:
; CHECK-LABEL: .Lfunc_end0
; CHECK: .size  update_register, .Lfunc_end0-update_register
