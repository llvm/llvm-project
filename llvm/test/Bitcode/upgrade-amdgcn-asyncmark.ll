; RUN: split-file %s %t
; RUN: llvm-as < %t/legacy.ll | llvm-dis | FileCheck %s --check-prefix=LEGACY
; RUN: llvm-as < %t/legacy-declared.ll | llvm-dis | FileCheck %s --check-prefix=DECLARED
; RUN: llvm-as < %t/staged.ll | llvm-dis | FileCheck %s --check-prefix=STAGED

; The asyncmark intrinsics originally had no stage mask operand. Upgrade them to
; a zero mask, which names every stage and so is the behavior they had.

;--- legacy.ll
define void @legacy() {
; LEGACY-LABEL: define void @legacy(
; LEGACY-NEXT:    call void @llvm.amdgcn.asyncmark(i32 0)
; LEGACY-NEXT:    call void @llvm.amdgcn.wait.asyncmark(i16 0, i32 0)
; LEGACY-NEXT:    call void @llvm.amdgcn.wait.asyncmark(i16 3, i32 0)
; LEGACY-NEXT:    ret void
;
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
  call void @llvm.amdgcn.wait.asyncmark(i16 3)
  ret void
}

;--- legacy-declared.ll
; The old declarations are spelled out and each is called more than once. The
; intrinsics are not overloaded, so the upgraded declaration has the same name
; as the old one.
declare void @llvm.amdgcn.asyncmark()
declare void @llvm.amdgcn.wait.asyncmark(i16)

define void @legacy_declared() {
; DECLARED-LABEL: define void @legacy_declared(
; DECLARED-NEXT:    call void @llvm.amdgcn.asyncmark(i32 0)
; DECLARED-NEXT:    call void @llvm.amdgcn.asyncmark(i32 0)
; DECLARED-NEXT:    call void @llvm.amdgcn.wait.asyncmark(i16 1, i32 0)
; DECLARED-NEXT:    call void @llvm.amdgcn.wait.asyncmark(i16 0, i32 0)
; DECLARED-NEXT:    ret void
;
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.wait.asyncmark(i16 1)
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
  ret void
}
; DECLARED-NOT: .old

;--- staged.ll
; Calls that already carry a mask are left alone. The masks here are non-empty
; so that leaving them alone is distinguishable from upgrading them.
define void @staged() {
; STAGED-LABEL: define void @staged(
; STAGED-NEXT:    call void @llvm.amdgcn.asyncmark(i32 1)
; STAGED-NEXT:    call void @llvm.amdgcn.wait.asyncmark(i16 1, i32 1)
; STAGED-NEXT:    ret void
;
  call void @llvm.amdgcn.asyncmark(i32 1)
  call void @llvm.amdgcn.wait.asyncmark(i16 1, i32 1)
  ret void
}
