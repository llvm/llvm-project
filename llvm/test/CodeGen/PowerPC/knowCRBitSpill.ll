; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:     -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 \
; RUN:     -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -ppc-late-peephole=false \
; RUN:     -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | FileCheck %s \
; RUN:     --implicit-check-not creqv --implicit-check-not crxor


; For known CRBit spills, CRSET/CRUNSET, it is more efficient to just load and
; spill the known value.  These tests verify that for CRSET and CRUNSET spills
; we do not extract the bit for spilling.

%struct.anon = type { i32 }

@b = common dso_local global ptr null, align 8
@a = common dso_local global i64 0, align 8

; Function Attrs: nounwind
define dso_local signext i32 @spillCRSET(i32 signext %p1, i32 signext %p2) {
; CHECK-LABEL: spillCRSET:
; CHECK:        # %bb.2:
; CHECK-DAG:    crnot [[CREG:.*]]*cr5+lt, eq
; CHECK-DAG:    mfocrf [[REG2:.*]], [[CREG]]
; CHECK-DAG:    rlwinm [[REG2]], [[REG2]]
; CHECK:        .LBB0_3:
; CHECK-NOT:    #UNENCODED_NOP
; CHECK:        lis [[REG1:.*]], -32768
; CHECK:        .LBB0_4:
; CHECK-NOT:    mfocrf [[REG2:.*]], [[CREG]]
; CHECK-NOT:    rlwinm [[REG2]], [[REG2]]
; CHECK:        stw [[REG1]]
; CHECK:        # %bb.5:

entry:
  %tobool = icmp eq i32 %p2, 0
  %tobool2 = icmp eq i32 %p1, 0
  br label %redo_first_pass

redo_first_pass:                                  ; preds = %for.end, %entry
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %redo_first_pass
  %call = tail call signext i32 @fn2() #2
  %tobool1 = icmp ne i32 %call, 0
  br label %if.end

if.end:                                           ; preds = %redo_first_pass, %if.then
  %c.1.off0 = phi i1 [ %tobool1, %if.then ], [ true, %redo_first_pass ]
  br i1 %tobool2, label %if.end4, label %if.then3

if.then3:                                         ; preds = %if.end
  %0 = load ptr, ptr @b, align 8
  store i32 1, ptr %0, align 4
  br label %if.end4

if.end4:                                          ; preds = %if.end, %if.then3
  tail call void asm sideeffect "#DO_NOTHING", "~{cr0},~{cr1},~{cr2},~{cr3},~{cr4},~{cr5},~{cr6},~{cr7}"()
  br i1 %c.1.off0, label %if.then6, label %if.end13

if.then6:                                         ; preds = %if.end4
  %1 = load i64, ptr @a, align 8
  %cmp21 = icmp eq i64 %1, 0
  br i1 %cmp21, label %if.end13, label %for.body

for.body:                                         ; preds = %if.then6, %for.body
  %s.122 = phi i64 [ %inc, %for.body ], [ 0, %if.then6 ]
  %call7 = tail call signext i32 @fn3()
  %inc = add nuw i64 %s.122, 1
  %exitcond = icmp eq i64 %inc, %1
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  br i1 %cmp21, label %if.end13, label %redo_first_pass

if.end13:                                         ; preds = %if.then6, %for.end, %if.end4
  ret i32 0
}

%struct.p5rx = type { i32 }

; Function Attrs: nounwind
define dso_local signext i32 @spillCRUNSET(ptr readonly %p1, i32 signext %p2, i32 signext %p3) {
; CHECK-LABEL: spillCRUNSET:
; CHECK:       # %bb.0: # %entry
; CHECK-DAG:    li [[REG1:.*]], 0
; CHECK-NOT:    #UNENCODED_NOP
; CHECK-NOT:    mfocrf [[REG2:.*]], [[CREG]]
; CHECK-NOT:    rlwinm [[REG2]], [[REG2]]
; CHECK:        stw [[REG1]]
; CHECK:        .LBB1_1:
entry:
  %and = and i32 %p3, 128
  %tobool = icmp eq i32 %and, 0
  %tobool2 = icmp eq ptr %p1, null
  %tobool12 = icmp eq i32 %p2, 0
  br label %redo_first_pass

redo_first_pass:                                  ; preds = %if.end11, %entry
  %a.0.off0 = phi i1 [ false, %entry ], [ %a.1.off0, %if.end11 ]
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %redo_first_pass
  %call = tail call signext i32 @fn2()
  %tobool1 = icmp ne i32 %call, 0
  br label %if.end

if.end:                                           ; preds = %redo_first_pass, %if.then
  %a.1.off0 = phi i1 [ %tobool1, %if.then ], [ %a.0.off0, %redo_first_pass ]
  tail call void asm sideeffect "#DO_NOTHING", "~{cr0},~{cr1},~{cr2},~{cr3},~{cr4},~{cr5},~{cr6},~{cr7}"()
  br i1 %tobool2, label %if.end11, label %land.lhs.true

land.lhs.true:                                    ; preds = %if.end
  %call3 = tail call signext i32 @fn3()
  %tobool4 = icmp eq i32 %call3, 0
  br i1 %tobool4, label %if.end11, label %land.lhs.true5

land.lhs.true5:                                   ; preds = %land.lhs.true
  %0 = load i32, ptr %p1, align 4
  %tobool6 = icmp eq i32 %0, 0
  %a.1.off0.not = xor i1 %a.1.off0, true
  %brmerge = or i1 %tobool6, %a.1.off0.not
  br i1 %brmerge, label %if.end11, label %if.then9

if.then9:                                         ; preds = %land.lhs.true5
  %call10 = tail call signext i32 @fn4()
  br label %if.end11

if.end11:                                         ; preds = %land.lhs.true5, %land.lhs.true, %if.end, %if.then9
  br i1 %tobool12, label %if.end14, label %redo_first_pass

if.end14:                                         ; preds = %if.end11
  ret i32 0
}

declare signext i32 @fn2(...)
declare signext i32 @fn3(...)
declare signext i32 @fn4(...)
