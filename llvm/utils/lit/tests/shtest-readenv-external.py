## Tests the readenv substitution.

# UNSUPPORTED: system-windows

# RUN: env LIT_USE_INTERNAL_SHELL=0 PATH=%{system-path} %{lit} -a %{inputs}/shtest-readenv | FileCheck -match-full-lines %s

# CHECK: -- Testing: 7 tests{{.*}}

# CHECK-LABEL: PASS: shtest-readenv :: command-from-env.txt ({{[^)]*}})
# CHECK: Command Output (stdout):
# CHECK: SUCCESS
# CHECK: Command Output (stderr):
# CHECK: export CMD="echo" # {{R}}UN: at line 1
# CHECK: + export CMD=echo
# CHECK: + CMD=echo
# CHECK: ${CMD-'Environment variable specified in readenv subsitution is not set: CMD'} SUCCESS # {{R}}UN: at line 2
# CHECK: + echo SUCCESS

# CHECK-LABEL: PASS: shtest-readenv :: empty-string.txt ({{[^)]*}})
# CHECK: Command Output (stdout):
# CHECK: PRE POST
# CHECK: Command Output (stderr):
# CHECK: export ASDF="" # {{R}}UN: at line 1
# CHECK: + export ASDF=
# CHECK: + ASDF=
# CHECK: echo PRE ${ASDF-'Environment variable specified in readenv subsitution is not set: ASDF'} POST # {{R}}UN: at line 2
# CHECK: + echo PRE POST

# CHECK-LABEL: PASS: shtest-readenv :: env.txt ({{[^)]*}})
# CHECK: Command Output (stdout):
# CHECK: PRE1 foo POST1
# CHECK: PRE2 foo POST2
# CHECK: Command Output (stderr):
# CHECK: export FOO=foo # {{R}}UN: at line 4
# CHECK: + export FOO=foo
# CHECK: + FOO=foo
# CHECK: env FOO=bar [[ECHO:.*]] PRE1 ${FOO-'Environment variable specified in readenv subsitution is not set: FOO'} POST1 # {{R}}UN: at line 6
# CHECK: + env FOO=bar [[ECHO]] PRE1 foo POST1
# CHECK: echo PRE2 ${FOO-'Environment variable specified in readenv subsitution is not set: FOO'} POST2 # {{R}}UN: at line 7
# CHECK: + echo PRE2 foo POST2

# CHECK-LABEL: PASS: shtest-readenv :: export.txt ({{[^)]*}})
# CHECK: Command Output (stdout):
# CHECK: PRE VAR CONTENT POST
# CHECK: Command Output (stderr):
# CHECK: export VAR="VAR CONTENT" # {{R}}UN: at line 1
# CHECK: + export 'VAR=VAR CONTENT'
# CHECK: + VAR='VAR CONTENT'
# CHECK: echo PRE ${VAR-'Environment variable specified in readenv subsitution is not set: VAR'} POST # {{R}}UN: at line 2
# CHECK: + echo PRE VAR CONTENT POST

# CHECK-LABEL: PASS: shtest-readenv :: pipeline.txt ({{[^)]*}})
# CHECK: Command Output (stdout):
# CHECK: PRE1 FIRST POST1
# CHECK: PRE2 SECOND POST2
# CHECK: PRE3 SECOND POST3
# CHECK: Command Output (stderr):
# CHECK: export A=INIT # {{R}}UN: at line 2
# CHECK: + export A=INIT
# CHECK: + A=INIT
# CHECK: export A=FIRST && echo PRE1 ${A-'Environment variable specified in readenv subsitution is not set: A'} POST1 ; export A=SECOND ; echo PRE2 ${A-'Environment variable specified in readenv subsitution is not set: A'} POST2 # {{R}}UN: at line 3
# CHECK: + export A=FIRST
# CHECK: + A=FIRST
# CHECK: + echo PRE1 FIRST POST1
# CHECK: + export A=SECOND
# CHECK: + A=SECOND
# CHECK: + echo PRE2 SECOND POST2
# CHECK: echo PRE3 ${A-'Environment variable specified in readenv subsitution is not set: A'} POST3 # {{R}}UN: at line 4
# CHECK: + echo PRE3 SECOND POST3

# CHECK-LABEL: PASS: shtest-readenv :: reexport.txt ({{[^)]*}})
# CHECK: Command Output (stdout):
# CHECK: PRE FIRST:SECOND POST
# CHECK: Command Output (stderr):
# CHECK: export VAR="FIRST" # {{R}}UN: at line 1
# CHECK: + export VAR=FIRST
# CHECK: + VAR=FIRST
# CHECK: export VAR="${VAR-'Environment variable specified in readenv subsitution is not set: VAR'}:SECOND" # {{R}}UN: at line 2
# CHECK: + export VAR=FIRST:SECOND
# CHECK: + VAR=FIRST:SECOND
# CHECK: echo PRE ${VAR-'Environment variable specified in readenv subsitution is not set: VAR'} POST # {{R}}UN: at line 3
# CHECK: + echo PRE FIRST:SECOND POST

# CHECK-LABEL: PASS: shtest-readenv :: unset.txt ({{[^)]*}})
# CHECK: Command Output (stdout):
# CHECK-NEXT: --
# CHECK-NEXT: PRE Environment variable specified in readenv subsitution is not set: SOME_UNSET_ENV_VAR POST
# CHECK: Command Output (stderr):
# CHECK-NEXT: --
# CHECK-NEXT: echo PRE ${SOME_UNSET_ENV_VAR-'Environment variable specified in readenv subsitution is not set: SOME_UNSET_ENV_VAR'} POST # {{R}}UN: at line 1
# CHECK-NEXT: + echo PRE 'Environment variable specified in readenv subsitution is not set: SOME_UNSET_ENV_VAR' POST
