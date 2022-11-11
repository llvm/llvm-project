; RUN: opt -passes=pgo-instr-gen -S 2>&1 < %s | FileCheck %s
;
; RUN: llvm-profdata merge %S/Inputs/callbr.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE


define i32 @a() {
entry:
; CHECK-NOT: ptrtoint ptr asm sideeffect
; CHECK: callbr void asm sideeffect
  %retval = alloca i32, align 4
  callbr void asm sideeffect "", "!i,~{dirflag},~{fpsr},~{flags}"() #1
          to label %asm.fallthrough [label %b]

asm.fallthrough:
  br label %b

b:
  %0 = load i32, ptr %retval, align 4
  ret i32 %0
}
; USE-LABEL: @a
; USE-SAME: !prof ![[BW_ENTRY:[0-9]+]]
; USE: callbr void asm sideeffect
; USE: to label
; USE-SAME: !prof ![[BW_CALLBR:[0-9]+]]
; USE ![[BW_ENTRY]] = !{!"function_entry_count", i64 100}
; USE ![[BW_CALLBR]] = !{!"branch_weights", i32 80, i32 20}

