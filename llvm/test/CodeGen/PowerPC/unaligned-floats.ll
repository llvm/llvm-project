; RUN: llc -mcpu=pwr7 -ppc-asm-full-reg-names -mtriple=powerpc-- \
; RUN:   -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mattr=allow-unaligned-fp-access -ppc-asm-full-reg-names \
; RUN:   -mtriple=powerpc-- -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -ppc-asm-full-reg-names -mtriple=powerpc-- \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefix=UNALIGN

; Test case as provided by author in https://bugs.llvm.org/show_bug.cgi?id=40554
%struct.anon = type { i32, [5 x i8] }

@s = dso_local local_unnamed_addr global %struct.anon { i32 0, [5 x i8] c"\00B\F6\E9y" }, align 4
@.str = private unnamed_addr constant [4 x i8] c"%g\0A\00", align 1
; Function Attrs: nofree nounwind
define dso_local i32 @main() local_unnamed_addr {
; CHECK-LABEL: main:
; CHECK:       lfs f1, 5(r3)
; CHECK:       blr
;
; UNALIGN-LABEL: main:
; UNALIGN:       lfs f1, 12(r1)
; UNALIGN:       blr
entry:
  %0 = load i8, ptr getelementptr inbounds (%struct.anon, ptr @s, i32 0, i32 1, i32 1), align 1
  %conv = zext i8 %0 to i32
  %shl = shl nuw i32 %conv, 24
  %1 = load i8, ptr getelementptr inbounds (%struct.anon, ptr @s, i32 0, i32 1, i32 2), align 2
  %conv1 = zext i8 %1 to i32
  %shl2 = shl nuw nsw i32 %conv1, 16
  %add = or i32 %shl2, %shl
  %2 = load i8, ptr getelementptr inbounds (%struct.anon, ptr @s, i32 0, i32 1, i32 3), align 1
  %conv3 = zext i8 %2 to i32
  %shl4 = shl nuw nsw i32 %conv3, 8
  %add5 = or i32 %add, %shl4
  %3 = load i8, ptr getelementptr inbounds (%struct.anon, ptr @s, i32 0, i32 1, i32 4), align 4
  %conv6 = zext i8 %3 to i32
  %add7 = or i32 %add5, %conv6
  %4 = bitcast i32 %add7 to float
  %conv8 = fpext float %4 to double
  %call = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @.str, double %conv8)
  ret i32 0
}
; Function Attrs: nofree nounwind
declare i32 @printf(ptr nocapture readonly, ...) local_unnamed_addr
