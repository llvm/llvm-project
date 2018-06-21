; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-scops -analyze \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s
; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-codegen -analyze \
; RUN: -polly-invariant-load-hoisting=true < %s

; The loop for.body is a scop with invariant load hoisting, but does not
; terminate predictably for ScalarEvolution. The scalar %1 therefore is not
; synthesizable using SCEVExpander. We therefore must have Stmt_for_end_loopexit
; to catch the induction variable at loop exit. We also check for not crashing
; at codegen because SCEVExpander would use the original induction variable in
; generated code.

%struct.bit_stream_struc.3.43.51.71.83.91.99.107.154 = type { i8*, i32, %struct._IO_FILE.1.41.49.69.81.89.97.105.153*, i8*, i32, i64, i32, i32 }
%struct._IO_FILE.1.41.49.69.81.89.97.105.153 = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker.0.40.48.68.80.88.96.104.152*, %struct._IO_FILE.1.41.49.69.81.89.97.105.153*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker.0.40.48.68.80.88.96.104.152 = type { %struct._IO_marker.0.40.48.68.80.88.96.104.152*, %struct._IO_FILE.1.41.49.69.81.89.97.105.153*, i32 }

define i32 @copy_buffer(%struct.bit_stream_struc.3.43.51.71.83.91.99.107.154* nocapture %bs) {
entry:
  %buf_byte_idx5.phi.trans.insert = getelementptr inbounds %struct.bit_stream_struc.3.43.51.71.83.91.99.107.154, %struct.bit_stream_struc.3.43.51.71.83.91.99.107.154* %bs, i64 0, i32 6
  br i1 undef, label %for.body, label %cleanup

for.body:
  %indvars.iv28 = phi i64 [ %indvars.iv.next29, %for.body ], [ 0, %entry ]
  %indvars.iv.next29 = add nuw nsw i64 %indvars.iv28, 1
  %0 = load i32, i32* %buf_byte_idx5.phi.trans.insert, align 8
  %cmp6 = icmp sgt i32 0, %0
  br i1 %cmp6, label %for.body, label %for.end.loopexit

for.end.loopexit:
  %var1 = trunc i64 %indvars.iv.next29 to i32
  br label %cleanup

cleanup:
  %retval.0 = phi i32 [ 0, %entry ], [ %var1, %for.end.loopexit ]
  ret i32 %retval.0
}


; CHECK:      Invariant Accesses: {
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [p_0_loaded_from_bs] -> { Stmt_for_body[i0] -> MemRef_bs[11] };
; CHECK-NEXT:         Execution Context: [p_0_loaded_from_bs] -> {  :  }
; CHECK-NEXT: }
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_body
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [p_0_loaded_from_bs] -> { Stmt_for_body[0] : p_0_loaded_from_bs >= 0 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [p_0_loaded_from_bs] -> { Stmt_for_body[i0] -> [0, 0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [p_0_loaded_from_bs] -> { Stmt_for_body[i0] -> MemRef_indvars_iv_next29[] };
; CHECK-NEXT:     Stmt_for_end_loopexit
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [p_0_loaded_from_bs] -> { Stmt_for_end_loopexit[] : p_0_loaded_from_bs >= 0 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [p_0_loaded_from_bs] -> { Stmt_for_end_loopexit[] -> [1, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [p_0_loaded_from_bs] -> { Stmt_for_end_loopexit[] -> MemRef_indvars_iv_next29[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [p_0_loaded_from_bs] -> { Stmt_for_end_loopexit[] -> MemRef_var1[] };
; CHECK-NEXT: }
