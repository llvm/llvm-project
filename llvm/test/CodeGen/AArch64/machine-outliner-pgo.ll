; RUN: rm -rf %t && split-file %s %t

; RUN: llvm-profdata merge %t/a.proftext -o %t/a.profdata
; RUN: opt %t/a.ll -passes=pgo-instr-use -pgo-test-profile-file=%t/a.profdata -S -o %t/a2.ll

; RUN: llc < %t/a2.ll -enable-machine-outliner=conservative-pgo -mtriple=aarch64-linux-gnu -profile-summary-cold-count=0 | FileCheck %s --check-prefixes=CHECK,CONSERVATIVE
; RUN: llc < %t/a2.ll -enable-machine-outliner=optimistic-pgo -mtriple=aarch64-linux-gnu -profile-summary-cold-count=0 | FileCheck %s --check-prefixes=CHECK,OPTIMISTIC

;--- a.ll
declare void @z(i32, i32, i32, i32)

; CHECK-LABEL: always_outline:
define void @always_outline() cold {
entry:
; CHECK: [[OUTLINED:OUTLINED_FUNCTION_[0-9]+]]
  tail call void @z(i32 1, i32 2, i32 3, i32 4)
  ret void
; CHECK: .cfi_endproc
}

; CHECK-LABEL: cold:
define void @cold() {
entry:
; CHECK: [[OUTLINED]]
  tail call void @z(i32 1, i32 2, i32 3, i32 4)
  ret void
; CHECK: .cfi_endproc
}

; CHECK-LABEL: hot:
define void @hot() minsize {
entry:
; CHECK-NOT: [[OUTLINED]]
  tail call void @z(i32 1, i32 2, i32 3, i32 4)
  ret void
; CHECK: .cfi_endproc
}

; CHECK-LABEL: no_profile_minsize:
define void @no_profile_minsize() minsize {
entry:
; CONSERVATIVE-NOT: [[OUTLINED]]
; OPTIMISTIC: [[OUTLINED]]
  tail call void @z(i32 1, i32 2, i32 3, i32 4)
  ret void
; CHECK: .cfi_endproc
}

; CHECK-LABEL: no_profile_optsize:
define void @no_profile_optsize() optsize {
entry:
; CHECK-NOT: [[OUTLINED]]
  tail call void @z(i32 1, i32 2, i32 3, i32 4)
  ret void
; CHECK: .cfi_endproc
}

; CHECK: [[OUTLINED]]:
; CHECK-SAME: // @{{.*}} Tail Call
; CHECK:      mov     w0, #1
; CHECK-NEXT: mov     w1, #2
; CHECK-NEXT: mov     w2, #3
; CHECK-NEXT: mov     w3, #4
; CHECK-NEXT: b       z

;--- a.proftext
:ir

cold
# Func Hash:
742261418966908927
# Num Counters:
1
# Counter Values:
0

hot
# Func Hash:
742261418966908927
# Num Counters:
1
# Counter Values:
100
