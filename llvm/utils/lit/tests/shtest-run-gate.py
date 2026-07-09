# Tests for the per-command feature gate on run lines, written as "RUN"
# immediately followed by "(<expr>):", where <expr> is a lit boolean
# expression (the same grammar as REQUIRES/UNSUPPORTED/XFAIL).
#
# The fixture suite (Inputs/shtest-run-gate/lit.cfg) makes the feature
# 'present' available and deliberately never defines 'absent'.
#
# Note: directive keywords are escaped (e.g. written as {{RUN}} then a colon)
# in the CHECK lines below so that lit does not mistake them for directives of
# *this* test.

# An ungated run line always runs; a gated run line runs only when its boolean
# expression holds.  Here 'present' is available and 'absent' is not, so every
# SKIP_* command must be gated out while every RUNS_* command executes.  A
# continuation line (trailing '\') inherits the gate of the RUN line it
# continues: RUNS_CONT_A/RUNS_CONT_B run as one gated-in command, while the
# gated-out SKIP_CONT continuation is dropped whole.
#
# RUN: %{lit} -v --show-all %{inputs}/shtest-run-gate/gate.txt \
# RUN:   | FileCheck %s --check-prefix=GATE --implicit-check-not=SKIP_
#
# GATE: PASS: shtest-run-gate :: gate.txt
# GATE: echo RUNS_PLAIN
# GATE: echo RUNS_PRESENT
# GATE: echo RUNS_OR
# GATE: echo RUNS_NOTABSENT
# GATE: echo RUNS_GROUPED
# GATE: echo RUNS_CONT_A && echo RUNS_CONT_B

# When every run line is gated out, the test is unsupported (lit exits 0).
#
# RUN: %{lit} -v --show-all %{inputs}/shtest-run-gate/all-gated.txt \
# RUN:   | FileCheck %s --check-prefix=ALLGATED
#
# ALLGATED: {{UNSUPPORTED}}: shtest-run-gate :: all-gated.txt
# ALLGATED: no '{{RUN}}:' lines enabled for the available features

# A gate must be a syntactically valid boolean expression; a malformed one is
# reported by the same BooleanExpression check the other keywords use.
#
# RUN: not %{lit} -v --show-all %{inputs}/shtest-run-gate/bad-gate-expr.txt \
# RUN:   | FileCheck %s --check-prefix=BADEXPR
#
# BADEXPR: UNRESOLVED: shtest-run-gate :: bad-gate-expr.txt
# BADEXPR: expected: {{.*}}or identifier

# A gate is only supported on COMMAND directives (i.e. run lines).
#
# RUN: not %{lit} -v --show-all %{inputs}/shtest-run-gate/gate-on-non-command.txt \
# RUN:   | FileCheck %s --check-prefix=NONCMD
#
# NONCMD: UNRESOLVED: shtest-run-gate :: gate-on-non-command.txt
# NONCMD: Feature gate {{.*}} may only be used on COMMAND directives
