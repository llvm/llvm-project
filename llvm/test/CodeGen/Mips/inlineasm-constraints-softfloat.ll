; RUN: llc  -march=mips < %s | FileCheck %s --check-prefix=MIPS32
; RUN: llc  -march=mips64 < %s | FileCheck %s --check-prefix=MIPS64

define dso_local void @read_double(ptr nocapture noundef readonly %0) local_unnamed_addr #0 {
  %2 = load double, ptr %0, align 8
; MIPS32-LABEL: read_double:
; MIPS32: lw      $2, 4($4)
; MIPS32-NEXT: lw      $3, 0($4)
; MIPS64-LABEL: read_double:
; MIPS64: ld      $2, 0($4)
  tail call void asm sideeffect "", "r,~{$1}"(double %2)
  ret void
}

define dso_local void @read_float(ptr nocapture noundef readonly %0) local_unnamed_addr #0 {
  %2 = load float, ptr %0, align 8
; MIPS32-LABEL: read_float:
; MIPS32: lw      $2, 0($4)
; MIPS64-LABEL: read_float:
; MIPS64: lw      $2, 0($4)
  tail call void asm sideeffect "", "r,~{$1}"(float %2)
  ret void
}

attributes #0 = { "target-features"="+soft-float" "use-soft-float"="true" }
