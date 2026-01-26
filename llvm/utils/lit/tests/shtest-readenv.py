## Tests the readenv substitution.

# RUN: env LIT_USE_INTERNAL_SHELL=1 not %{lit} -a %{inputs}/shtest-readenv | FileCheck -match-full-lines %s

# CHECK: -- Testing: 7 tests{{.*}}

# CHECK-LABEL: PASS: shtest-readenv :: command-from-env.txt ({{[^)]*}})
# CHECK: export CMD="echo"
# CHECK: # executed command: export CMD=echo
# CHECK: $CMD SUCCESS
# CHECK: # executed command: '%{readenv:CMD}' SUCCESS
# CHECK: # | SUCCESS

# CHECK-LABEL: PASS: shtest-readenv :: empty-string.txt ({{[^)]*}})
# CHECK: export ASDF=""
# CHECK: # executed command: export ASDF=
# CHECK: echo PRE $ASDF POST
# CHECK: # executed command: echo PRE '%{readenv:ASDF}' POST
# CHECK: # | PRE POST

# CHECK-LABEL: PASS: shtest-readenv :: env.txt ({{[^)]*}})
# CHECK: export FOO=foo
# CHECK: # executed command: export FOO=foo
# CHECK: env FOO=bar [[ECHO:.*]] PRE1 $FOO POST1
# CHECK: # executed command: env FOO=bar [[ECHO]] PRE1 '%{readenv:FOO}' POST1
# CHECK: # | PRE1 foo POST1
# CHECK: echo PRE2 $FOO POST2
# CHECK: # executed command: echo PRE2 '%{readenv:FOO}' POST2
# CHECK: # | PRE2 foo POST2

# CHECK-LABEL: PASS: shtest-readenv :: export.txt ({{[^)]*}})
# CHECK: export VAR="VAR CONTENT"
# CHECK: # executed command: export 'VAR=VAR CONTENT'
# CHECK: echo PRE $VAR POST
# CHECK: # executed command: echo PRE '%{readenv:VAR}' POST
# CHECK: # | PRE VAR CONTENT POST

# CHECK-LABEL: PASS: shtest-readenv :: pipeline.txt ({{[^)]*}})
# CHECK: export A=INIT
# CHECK: # executed command: export A=INIT
# CHECK: export A=FIRST && echo PRE1 $A POST1 ; export A=SECOND ; echo PRE2 $A POST2
# CHECK: # executed command: export A=FIRST
# CHECK: # executed command: echo PRE1 '%{readenv:A}' POST1
# CHECK: # | PRE1 FIRST POST1
# CHECK: # executed command: export A=SECOND
# CHECK: # executed command: echo PRE2 '%{readenv:A}' POST2
# CHECK: # | PRE2 SECOND POST2
# CHECK: echo PRE3 $A POST3
# CHECK: # executed command: echo PRE3 '%{readenv:A}' POST3
# CHECK: # | PRE3 SECOND POST3

# CHECK-LABEL: PASS: shtest-readenv :: reexport.txt ({{[^)]*}})
# CHECK: export VAR="FIRST"
# CHECK: # executed command: export VAR=FIRST
# CHECK: export VAR="$VAR:SECOND"
# CHECK: # executed command: export 'VAR=%{readenv:VAR}:SECOND'
# CHECK: echo PRE $VAR POST
# CHECK: # executed command: echo PRE '%{readenv:VAR}' POST
# CHECK: # | PRE FIRST:SECOND POST

# CHECK-LABEL: FAIL: shtest-readenv :: unset.txt ({{[^)]*}})
# CHECK: echo PRE $SOME_UNSET_ENV_VAR POST
# CHECK: # executed command: echo PRE '%{readenv:SOME_UNSET_ENV_VAR}' POST
# CHECK: # | Environment variable specified in readenv subsitution is not set: SOME_UNSET_ENV_VAR
