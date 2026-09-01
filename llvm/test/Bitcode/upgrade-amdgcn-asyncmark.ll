; RUN: split-file %s %t
; RUN: llvm-as < %t/legacy.ll | llvm-dis | FileCheck %s --check-prefix=LEGACY
; RUN: llvm-as < %t/staged.ll | llvm-dis | FileCheck %s --check-prefix=STAGED

; The asyncmark intrinsics originally had no stage operand. Upgrade them to the
; ALL stage (16), which is the behavior they had.

;--- legacy.ll
define void @legacy() {
; LEGACY-LABEL: define void @legacy(
; LEGACY-NEXT:    call void @llvm.amdgcn.asyncmark(i32 16)
; LEGACY-NEXT:    call void @llvm.amdgcn.wait.asyncmark(i16 0, i32 16)
; LEGACY-NEXT:    call void @llvm.amdgcn.wait.asyncmark(i16 3, i32 16)
; LEGACY-NEXT:    ret void
;
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
  call void @llvm.amdgcn.wait.asyncmark(i16 3)
  ret void
}

;--- staged.ll
; Calls that already carry a stage operand are left alone.
define void @staged() {
; STAGED-LABEL: define void @staged(
; STAGED-NEXT:    call void @llvm.amdgcn.asyncmark(i32 0)
; STAGED-NEXT:    call void @llvm.amdgcn.wait.asyncmark(i16 1, i32 0)
; STAGED-NEXT:    ret void
;
  call void @llvm.amdgcn.asyncmark(i32 0)
  call void @llvm.amdgcn.wait.asyncmark(i16 1, i32 0)
  ret void
}
