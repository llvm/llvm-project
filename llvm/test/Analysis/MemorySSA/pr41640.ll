; RUN: opt -disable-output -passes='loop-mssa(licm),print<memoryssa>' < %s 2>&1 | FileCheck %s
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

@g_248 = external dso_local local_unnamed_addr global i32, align 4
@g_976 = external dso_local global i64, align 8
@g_1087 = external dso_local global i32**, align 8

; CHECK-LABEL: @f1()
; CHECK: 5 = MemoryPhi(
; CHECK-NOT: 7 = MemoryPhi(
define dso_local fastcc void @f1() unnamed_addr #0 {
label0:
  br i1 undef, label %thread-pre-split.i.preheader, label %label5

thread-pre-split.i.preheader:                     ; preds = %label0
  br label %thread-pre-split.i

thread-pre-split.i.us:                            ; preds = %.critedge1.i.us
  br i1 undef, label %.preheader.i.us.preheader, label %label2

.preheader.i.us.preheader:                        ; preds = %thread-pre-split.i.us
  br label %.preheader.i.us

.preheader.i.us:                                  ; preds = %._crit_edge.i.us, %.preheader.i.us.preheader
  br i1 undef, label %.lr.ph.i.us, label %._crit_edge.i.us

.lr.ph.i.us:                                      ; preds = %.preheader.i.us
  br label %label1

label1:                                      ; preds = %label1, %.lr.ph.i.us
  br i1 undef, label %label1, label %._crit_edge.i.us

._crit_edge.i.us:                                 ; preds = %label1, %.preheader.i.us
  br i1 undef, label %.preheader.i.us, label %._crit_edge5.i.us

._crit_edge5.i.us:                                ; preds = %._crit_edge.i.us
  br label %label2

label2:                                  ; preds = %._crit_edge5.i.us, %thread-pre-split.i.us
  tail call void @foo16()
  br i1 undef, label %.lr.ph8.i.us.preheader, label %label4

.lr.ph8.i.us.preheader:                           ; preds = %label2
  br label %.lr.ph8.i.us

.lr.ph8.i.us:                                     ; preds = %.lr.ph8.i.us, %.lr.ph8.i.us.preheader
  %tmp3 = load volatile i64, i64* @g_976, align 8
  br i1 undef, label %.lr.ph8.i.us, label %._crit_edge9.i.us

._crit_edge9.i.us:                                ; preds = %.lr.ph8.i.us
  br label %label4

label4:                                      ; preds = %._crit_edge9.i.us, %label2
  br i1 true, label %f9.exit, label %.critedge1.i.us

.critedge1.i.us:                                  ; preds = %label4
  br i1 undef, label %thread-pre-split.i.us, label %f9.exit

label5:                                      ; preds = %label0
  unreachable

thread-pre-split.i:                               ; preds = %.critedge1.i, %thread-pre-split.i.preheader
  br i1 undef, label %.preheader.i.preheader, label %.critedge1.i

.preheader.i.preheader:                           ; preds = %thread-pre-split.i
  br label %.preheader.i

.preheader.i:                                     ; preds = %._crit_edge.i, %.preheader.i.preheader
  br i1 undef, label %.lr.ph.i, label %._crit_edge.i

.lr.ph.i:                                         ; preds = %.preheader.i
  br label %label6

label6:                                      ; preds = %label6, %.lr.ph.i
  br i1 undef, label %label6, label %._crit_edge.i.loopexit

._crit_edge.i.loopexit:                           ; preds = %label6
  br label %._crit_edge.i

._crit_edge.i:                                    ; preds = %._crit_edge.i.loopexit, %.preheader.i
  br i1 undef, label %.preheader.i, label %._crit_edge5.i

._crit_edge5.i:                                   ; preds = %._crit_edge.i
  br label %.critedge1.i

.critedge1.i:                                     ; preds = %._crit_edge5.i, %thread-pre-split.i
  %tmp7 = load i32, i32* @g_248, align 4
  %tmp8 = xor i32 %tmp7, 55987
  store i32 %tmp8, i32* @g_248, align 4
  br i1 undef, label %thread-pre-split.i, label %f9.exit.loopexit

f9.exit.loopexit:                                 ; preds = %.critedge1.i
  br label %f9.exit

f9.exit:                                          ; preds = %f9.exit.loopexit, %.critedge1.i.us, %label4
  %tmp9 = load volatile i32**, i32*** @g_1087, align 8
  ret void
}

declare dso_local void @foo16() local_unnamed_addr #1

attributes #0 = { "target-features"="+transactional-execution,+vector" }
attributes #1 = { "use-soft-float"="false" }
