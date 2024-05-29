; RUN: opt %loadNPMPolly -S -passes=polly-codegen < %s | FileCheck %s
; XFAIL: *
;
; The boundary context contains a constant that does not fit in 64 bits. Hence,
; make sure we bail out. On certain systems, e.g. AOSP, no runtime support for
; 128bit operations is available and consequently the code generation of large
; values might cause linker errors.
;
; CHECK: br i1 false, label %polly.start, label %bb11.pre_entry_bb

target triple = "x86_64-unknown-linux-gnu"

@global = external global i32, align 4
@global1 = external global i32, align 4

; Function Attrs: nounwind uwtable
define void @hoge(ptr %arg) #0 {
bb:
  br label %bb5

bb5:                                              ; preds = %bb
  %tmp = load i32, ptr @global, align 4
  %tmp6 = sext i32 %tmp to i64
  br label %bb11

bb7:                                              ; preds = %bb19
  %tmp8 = load i32, ptr @global1, align 4
  %tmp9 = sext i32 %tmp8 to i64
  %tmp10 = icmp slt i64 %tmp13, %tmp9
  br i1 %tmp10, label %bb11, label %bb20

bb11:                                             ; preds = %bb7, %bb5
  %tmp12 = phi i64 [ %tmp6, %bb5 ], [ %tmp13, %bb7 ]
  %tmp13 = add i64 %tmp12, 1
  %tmp14 = getelementptr inbounds i8, ptr %arg, i64 %tmp13
  %tmp15 = load i8, ptr %tmp14, align 1
  br i1 false, label %bb16, label %bb17

bb16:                                             ; preds = %bb11
  br label %bb18

bb17:                                             ; preds = %bb11
  br label %bb18

bb18:                                             ; preds = %bb17, %bb16
  br label %bb19

bb19:                                             ; preds = %bb19, %bb18
  br i1 undef, label %bb19, label %bb7

bb20:                                             ; preds = %bb7
  br label %bb21

bb21:                                             ; preds = %bb20
  ret void
}
