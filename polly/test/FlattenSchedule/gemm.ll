; RUN: opt %loadPolly -polly-print-flatten-schedule -disable-output < %s | FileCheck %s
;
; dgemm kernel
; C := alpha*A*B + beta*C
; C[ni][nj]
; A[ni][nk]
; B[nk][nj]

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"

define void @gemm(i32 %ni, i32 %nj, i32 %nk, double %alpha, double %beta, ptr noalias nonnull %C, ptr noalias nonnull %A, ptr noalias nonnull %B) {
entry:
  br label %ni.for

ni.for:
  %i = phi i32 [0, %entry], [%i.inc, %ni.inc]
  %i.cmp = icmp slt i32 %i, 3
  br i1 %i.cmp, label %nj.for, label %ni.exit

  nj.for:
    %j = phi i32 [0, %ni.for], [%j.inc, %nj.inc]
    %j.cmp = icmp slt i32 %j, 7
    br i1 %j.cmp, label %nj_beta, label %nj.exit

    nj_beta:
     %c_stride = mul nsw i32 %i, 3; %nj
     %c_idx_i = getelementptr inbounds double, ptr %C, i32 %c_stride
     %c_idx_ij = getelementptr inbounds double, ptr %c_idx_i, i32 %j

     ; C[i]ptr= beta
     %c = load double, ptr %c_idx_ij
     %c_beta = fmul double %c, %beta
     store double %c_beta, ptr %c_idx_ij

     br label %nk.for

    nk.for:
      %k = phi i32 [0, %nj_beta], [%k.inc, %nk.inc]
      %k.cmp = icmp slt i32 %k, 3 ; %nk
      br i1 %k.cmp, label %nk_alpha, label %nk.exit

      nk_alpha:
        %a_stride = mul nsw i32 %i, 3; %nk
        %a_idx_i = getelementptr inbounds double, ptr %A, i32 %a_stride
        %a_idx_ik = getelementptr inbounds double, ptr %a_idx_i, i32 %k

        %b_stride = mul nsw i32 %k, 3; %nj
        %b_idx_k = getelementptr inbounds double, ptr %B, i32 %b_stride
        %b_idx_kj = getelementptr inbounds double, ptr %b_idx_k, i32 %j

        ; C[i][j] += alpha * A[i]ptr B[k][j]
        %a = load double, ptr %a_idx_ik
        %b = load double, ptr %b_idx_kj
        %beta_c = load double, ptr %c_idx_ij

        %alpha_a = fmul double %a, %alpha
        %alpha_a_b = fmul double %alpha_a, %b
        %beta_c_alpha_a_b = fadd double %beta_c, %alpha_a_b

        store double %beta_c_alpha_a_b, ptr %c_idx_ij

        br label %nk.inc

    nk.inc:
      %k.inc = add nuw nsw i32 %k, 1
      br label %nk.for

    nk.exit:
      ; store double %c, ptr %c_idx_ij
      br label %nj.inc

  nj.inc:
    %j.inc = add nuw nsw i32 %j, 1
    br label %nj.for

  nj.exit:
    br label %ni.inc

ni.inc:
  %i.inc = add nuw nsw i32 %i, 1
  br label %ni.for

ni.exit:
  br label %return

return:
  ret void
}


; CHECK:      Schedule before flattening {
; CHECK-NEXT:     { Stmt_nk_alpha[i0, i1, i2] -> [i0, i1, 1, i2] }
; CHECK-NEXT:     { Stmt_nj_beta[i0, i1] -> [i0, i1, 0, 0] }
; CHECK-NEXT: }
; CHECK:      Schedule after flattening {
; CHECK-NEXT:     { Stmt_nj_beta[i0, i1] -> [28i0 + 4i1] }
; CHECK-NEXT:     { Stmt_nk_alpha[i0, i1, i2] -> [1 + 28i0 + 4i1 + i2] }
; CHECK-NEXT: }
