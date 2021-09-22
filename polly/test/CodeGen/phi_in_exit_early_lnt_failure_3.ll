; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
; This caused an lnt crash at some point, just verify it will run through and
; produce the PHI node in the exit we are looking for.
;
; CHECK-LABEL: polly.merge_new_and_old:
; CHECK-NEXT:     %n2ptr.2.ph.merge = phi i8* [ %n2ptr.2.ph.final_reload, %polly.exiting ], [ %n2ptr.2.ph, %if.end.45.region_exiting ]
;
; CHECK-LABEL: if.end.45:
; CHECK-NEXT:     %n2ptr.2 = phi i8* [ %add.ptr25, %entry ], [ %add.ptr25, %while.cond.preheader ], [ %n2ptr.2.ph.merge, %polly.merge_new_and_old ]

%struct.bc_struct.0.2.4.6.8.15.24.27.29.32.38.46.48.92.93.94.95.97.99.100.102.105.107.111.118.119.121 = type { i32, i32, i32, i32, [1024 x i8] }

; Function Attrs: nounwind uwtable
declare %struct.bc_struct.0.2.4.6.8.15.24.27.29.32.38.46.48.92.93.94.95.97.99.100.102.105.107.111.118.119.121* @new_num() #0

; Function Attrs: nounwind uwtable
define void @_do_add(%struct.bc_struct.0.2.4.6.8.15.24.27.29.32.38.46.48.92.93.94.95.97.99.100.102.105.107.111.118.119.121* %n2) #0 {
entry:
  %call = tail call %struct.bc_struct.0.2.4.6.8.15.24.27.29.32.38.46.48.92.93.94.95.97.99.100.102.105.107.111.118.119.121* @new_num()
  %0 = load i32, i32* undef, align 4
  %add.ptr22 = getelementptr inbounds %struct.bc_struct.0.2.4.6.8.15.24.27.29.32.38.46.48.92.93.94.95.97.99.100.102.105.107.111.118.119.121, %struct.bc_struct.0.2.4.6.8.15.24.27.29.32.38.46.48.92.93.94.95.97.99.100.102.105.107.111.118.119.121* %n2, i64 0, i32 4, i64 0
  %add.ptr24 = getelementptr inbounds i8, i8* %add.ptr22, i64 0
  %add.ptr25 = getelementptr inbounds i8, i8* %add.ptr24, i64 -1
  %add.ptr29 = getelementptr inbounds %struct.bc_struct.0.2.4.6.8.15.24.27.29.32.38.46.48.92.93.94.95.97.99.100.102.105.107.111.118.119.121, %struct.bc_struct.0.2.4.6.8.15.24.27.29.32.38.46.48.92.93.94.95.97.99.100.102.105.107.111.118.119.121* %call, i64 0, i32 4, i64 0
  %add.ptr31 = getelementptr inbounds i8, i8* %add.ptr29, i64 0
  %add.ptr32 = getelementptr inbounds i8, i8* %add.ptr31, i64 -1
  br i1 undef, label %if.end.45, label %if.then

if.then:                                          ; preds = %entry
  br i1 undef, label %while.cond.preheader, label %while.cond.38.preheader

while.cond.38.preheader:                          ; preds = %if.then
  %cmp39.39 = icmp sgt i32 %0, 0
  br i1 %cmp39.39, label %while.body.40.lr.ph, label %if.end.45

while.body.40.lr.ph:                              ; preds = %while.cond.38.preheader
  br label %while.body.40

while.cond.preheader:                             ; preds = %if.then
  br i1 undef, label %while.body.lr.ph, label %if.end.45

while.body.lr.ph:                                 ; preds = %while.cond.preheader
  br label %while.body

while.body:                                       ; preds = %while.body, %while.body.lr.ph
  br label %while.body

while.body.40:                                    ; preds = %while.body.40, %while.body.40.lr.ph
  %sumptr.141 = phi i8* [ %add.ptr32, %while.body.40.lr.ph ], [ %incdec.ptr42, %while.body.40 ]
  %n2ptr.040 = phi i8* [ %add.ptr25, %while.body.40.lr.ph ], [ %incdec.ptr41, %while.body.40 ]
  %incdec.ptr41 = getelementptr inbounds i8, i8* %n2ptr.040, i64 -1
  %1 = load i8, i8* %n2ptr.040, align 1
  %incdec.ptr42 = getelementptr inbounds i8, i8* %sumptr.141, i64 -1
  store i8 %1, i8* %sumptr.141, align 1
  br i1 false, label %while.body.40, label %while.cond.38.if.end.45.loopexit9_crit_edge

while.cond.38.if.end.45.loopexit9_crit_edge:      ; preds = %while.body.40
  br label %if.end.45

if.end.45:                                        ; preds = %while.cond.38.if.end.45.loopexit9_crit_edge, %while.cond.preheader, %while.cond.38.preheader, %entry
  %n2ptr.2 = phi i8* [ %add.ptr25, %entry ], [ %add.ptr25, %while.cond.preheader ], [ undef, %while.cond.38.if.end.45.loopexit9_crit_edge ], [ %add.ptr25, %while.cond.38.preheader ]
  ret void
}
