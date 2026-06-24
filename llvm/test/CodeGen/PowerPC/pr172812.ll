; RUN: not opt %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: The only supported target OS's are AIX and ELF-based OS's
target triple = "powerpc-apple-darwin7.2"

@a = dso_local local_unnamed_addr global i32 0, align 4,!type!0
@g = internal dso_local global ptr null, align 4,!type!2

define void @_Z4testv() {
entry:
  store ptr inttoptr (i32 42 to ptr), ptr @a, align 4
  %0 = load ptr, ptr @a, align 4
  %1 = load i32, ptr %0, align 4
  ret void
}

!0 =!{i32 0,!"_ZTS1A1B"}
!2 =!{i32 0,!"_ZTS1A1D"}
