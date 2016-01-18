; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/syntax.prof 2>&1 | FileCheck -check-prefix=NO-DEBUG %s
; RUN: not opt < %s -sample-profile -sample-profile-file=missing.prof 2>&1 | FileCheck -check-prefix=MISSING-FILE %s
; RUN: not opt < %s -sample-profile -sample-profile-file=%S/Inputs/bad_fn_header.prof 2>&1 | FileCheck -check-prefix=BAD-FN-HEADER %s
; RUN: not opt < %s -sample-profile -sample-profile-file=%S/Inputs/bad_sample_line.prof 2>&1 | FileCheck -check-prefix=BAD-SAMPLE-LINE %s
; RUN: not opt < %s -sample-profile -sample-profile-file=%S/Inputs/bad_line_values.prof 2>&1 | FileCheck -check-prefix=BAD-LINE-VALUES %s
; RUN: not opt < %s -sample-profile -sample-profile-file=%S/Inputs/bad_discriminator_value.prof 2>&1 | FileCheck -check-prefix=BAD-DISCRIMINATOR-VALUE %s
; RUN: not opt < %s -sample-profile -sample-profile-file=%S/Inputs/bad_samples.prof 2>&1 | FileCheck -check-prefix=BAD-SAMPLES %s
; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/bad_mangle.prof 2>&1 >/dev/null

define void @empty() {
entry:
  ret void
}
; NO-DEBUG: warning: No debug information found in function empty: Function profile not used
; MISSING-FILE: missing.prof: Could not open profile:
; BAD-FN-HEADER: error: {{.*}}bad_fn_header.prof:1: Expected 'mangled_name:NUM:NUM', found 3empty:100:BAD
; BAD-SAMPLE-LINE: error: {{.*}}bad_sample_line.prof:3: Expected 'NUM[.NUM]: NUM[ mangled_name:NUM]*', found 1: BAD
; BAD-LINE-VALUES: error: {{.*}}bad_line_values.prof:2: Expected 'mangled_name:NUM:NUM', found -1: 10
; BAD-DISCRIMINATOR-VALUE: error: {{.*}}bad_discriminator_value.prof:2: Expected 'NUM[.NUM]: NUM[ mangled_name:NUM]*', found 1.-3: 10
; BAD-SAMPLES: error: {{.*}}bad_samples.prof:2: Expected 'NUM[.NUM]: NUM[ mangled_name:NUM]*', found 1.3: -10
