; Test the MSA ctcmsa and cfcmsa intrinsics (which are encoded with the ELM
; instruction format).

; RUN: llc -march=mips -mattr=+msa,+fp64 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 -verify-machineinstrs < %s | FileCheck %s

define i32 @msa_ir_cfcmsa_test() nounwind {
entry:
  %0 = tail call i32 @llvm.mips.cfcmsa(i32 0)
  ret i32 %0
}

; CHECK: msa_ir_cfcmsa_test:
; CHECK: cfcmsa $[[R1:[0-9]+]], $0
; CHECK: .size msa_ir_cfcmsa_test
;
define i32 @msa_csr_cfcmsa_test() nounwind {
entry:
  %0 = tail call i32 @llvm.mips.cfcmsa(i32 1)
  ret i32 %0
}

; CHECK: msa_csr_cfcmsa_test:
; CHECK: cfcmsa $[[R1:[0-9]+]], $1
; CHECK: .size msa_csr_cfcmsa_test
;
define i32 @msa_access_cfcmsa_test() nounwind {
entry:
  %0 = tail call i32 @llvm.mips.cfcmsa(i32 2)
  ret i32 %0
}

; CHECK: msa_access_cfcmsa_test:
; CHECK: cfcmsa $[[R1:[0-9]+]], $2
; CHECK: .size msa_access_cfcmsa_test
;
define i32 @msa_save_cfcmsa_test() nounwind {
entry:
  %0 = tail call i32 @llvm.mips.cfcmsa(i32 3)
  ret i32 %0
}

; CHECK: msa_save_cfcmsa_test:
; CHECK: cfcmsa $[[R1:[0-9]+]], $3
; CHECK: .size msa_save_cfcmsa_test
;
define i32 @msa_modify_cfcmsa_test() nounwind {
entry:
  %0 = tail call i32 @llvm.mips.cfcmsa(i32 4)
  ret i32 %0
}

; CHECK: msa_modify_cfcmsa_test:
; CHECK: cfcmsa $[[R1:[0-9]+]], $4
; CHECK: .size msa_modify_cfcmsa_test
;
define i32 @msa_request_cfcmsa_test() nounwind {
entry:
  %0 = tail call i32 @llvm.mips.cfcmsa(i32 5)
  ret i32 %0
}

; CHECK: msa_request_cfcmsa_test:
; CHECK: cfcmsa $[[R1:[0-9]+]], $5
; CHECK: .size msa_request_cfcmsa_test
;
define i32 @msa_map_cfcmsa_test() nounwind {
entry:
  %0 = tail call i32 @llvm.mips.cfcmsa(i32 6)
  ret i32 %0
}

; CHECK: msa_map_cfcmsa_test:
; CHECK: cfcmsa $[[R1:[0-9]+]], $6
; CHECK: .size msa_map_cfcmsa_test
;
define i32 @msa_unmap_cfcmsa_test() nounwind {
entry:
  %0 = tail call i32 @llvm.mips.cfcmsa(i32 7)
  ret i32 %0
}

; CHECK: msa_unmap_cfcmsa_test:
; CHECK: cfcmsa $[[R1:[0-9]+]], $7
; CHECK: .size msa_unmap_cfcmsa_test
;
define i32 @msa_invalid_reg_cfcmsa_test() nounwind {
entry:
  %0 = tail call i32 @llvm.mips.cfcmsa(i32 8)
  ret i32 %0
}

; CHECK-LABEL: msa_invalid_reg_cfcmsa_test:
; CHECK: cfcmsa ${{[0-9]+}}, $8
;
define void @msa_ir_ctcmsa_test() nounwind {
entry:
  tail call void @llvm.mips.ctcmsa(i32 0, i32 1)
  ret void
}

; CHECK: msa_ir_ctcmsa_test:
; CHECK: ctcmsa $0
; CHECK: .size msa_ir_ctcmsa_test
;
define void @msa_csr_ctcmsa_test() nounwind {
entry:
  tail call void @llvm.mips.ctcmsa(i32 1, i32 1)
  ret void
}

; CHECK: msa_csr_ctcmsa_test:
; CHECK: ctcmsa $1
; CHECK: .size msa_csr_ctcmsa_test
;
define void @msa_access_ctcmsa_test() nounwind {
entry:
  tail call void @llvm.mips.ctcmsa(i32 2, i32 1)
  ret void
}

; CHECK: msa_access_ctcmsa_test:
; CHECK: ctcmsa $2
; CHECK: .size msa_access_ctcmsa_test
;
define void @msa_save_ctcmsa_test() nounwind {
entry:
  tail call void @llvm.mips.ctcmsa(i32 3, i32 1)
  ret void
}

; CHECK: msa_save_ctcmsa_test:
; CHECK: ctcmsa $3
; CHECK: .size msa_save_ctcmsa_test
;
define void @msa_modify_ctcmsa_test() nounwind {
entry:
  tail call void @llvm.mips.ctcmsa(i32 4, i32 1)
  ret void
}

; CHECK: msa_modify_ctcmsa_test:
; CHECK: ctcmsa $4
; CHECK: .size msa_modify_ctcmsa_test
;
define void @msa_request_ctcmsa_test() nounwind {
entry:
  tail call void @llvm.mips.ctcmsa(i32 5, i32 1)
  ret void
}

; CHECK: msa_request_ctcmsa_test:
; CHECK: ctcmsa $5
; CHECK: .size msa_request_ctcmsa_test
;
define void @msa_map_ctcmsa_test() nounwind {
entry:
  tail call void @llvm.mips.ctcmsa(i32 6, i32 1)
  ret void
}

; CHECK: msa_map_ctcmsa_test:
; CHECK: ctcmsa $6
; CHECK: .size msa_map_ctcmsa_test
;
define void @msa_unmap_ctcmsa_test() nounwind {
entry:
  tail call void @llvm.mips.ctcmsa(i32 7, i32 1)
  ret void
}

; CHECK: msa_unmap_ctcmsa_test:
; CHECK: ctcmsa $7
; CHECK: .size msa_unmap_ctcmsa_test
;
define void @msa_invalid_reg_ctcmsa_test() nounwind {
entry:
  tail call void @llvm.mips.ctcmsa(i32 8, i32 1)
  ret void
}

; CHECK: msa_invalid_reg_ctcmsa_test:
; CHECK: ctcmsa $8
;
declare i32 @llvm.mips.cfcmsa(i32) nounwind
declare void @llvm.mips.ctcmsa(i32, i32) nounwind
