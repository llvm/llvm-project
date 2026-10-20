; RUN: not llc -march=arst -o /dev/null %s 2>&1 | FileCheck -check-prefix=MARCH %s
; RUN: not llc -mtriple=arst-- -o /dev/null %s 2>&1 | FileCheck -check-prefix=MTRIPLE %s

; Check the error message doesn't say error twice.

; MARCH: {{.*}}llc{{.*}}: error: invalid target 'arst'.
; MARCH-EMPTY:
;
; MTRIPLE: {{.*}}llc{{.*}}: error: unable to get target for 'arst-unknown-unknown', see --version and --triple.
; MTRIPLE-EMPTY:

define void @func() {
  ret void
}
