; RUN: not llc -mtriple=x86_64-unknown-windows-msvc -mattr=+egpr -o /dev/null %s 2>&1 | FileCheck %s
; CHECK: error: {{.*}}: in function func {{.*}}: EGPR (R16-R31) requires V3 unwind info on Windows x64

; EGPR enabled without V3 unwind (default V1) should produce a recoverable
; backend diagnostic (no crash, no stack trace).
; The uwtable attribute and the call site force a stack frame and SEH unwind
; info emission so the V3 pass runs regardless of the host platform's default
; (matters for cross-compilation on Linux/macOS hosts targeting Windows).

declare void @other()

define dso_local void @func() uwtable {
entry:
  call void @other()
  ret void
}
