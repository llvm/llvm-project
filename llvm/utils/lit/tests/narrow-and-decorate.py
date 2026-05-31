# End-to-end check that --check, --param fn=, and --param decorate= compose
# correctly on a real opt+FileCheck pipeline. The inner sample has two RUN
# lines and a deliberately-wrong CHECK in @bar's section.

# --- No params: RUN 0 fails on @bar's FILTER_AWAY check ---
# RUN: not %{lit} -a %{inputs}/narrow-and-decorate/sample.ll 2>&1 \
# RUN:   | FileCheck --check-prefix=BASE_FAIL %s
#
# BASE_FAIL: FAIL:
# BASE_FAIL: FILTER_AWAY

# --- --check=1: skip RUN 0, run only the ONLYFOO prefix RUN, which passes ---
# RUN: %{lit} --check=1 %{inputs}/narrow-and-decorate/sample.ll \
# RUN:   | FileCheck --check-prefix=PASS %s
#
# PASS: Passed: 1

# --- --param fn=foo --check=0: extract drops @bar; filter drops its CHECK ---
# RUN: %{lit} -a --param fn=foo --check=0 %{inputs}/narrow-and-decorate/sample.ll \
# RUN:   | FileCheck --check-prefix=FN %s
#
# FN: llvm-extract --func=foo
# FN: --filter-label=foo
# FN: Passed:

# --- All three: --check=0 + --param fn=foo + --param decorate=time ---
# RUN: %{lit} -a --param fn=foo --param decorate=time --check=0 \
# RUN:   %{inputs}/narrow-and-decorate/sample.ll \
# RUN:   | FileCheck --check-prefix=ALL %s
#
# ALL: llvm-extract --func=foo
# ALL: time {{.*}}opt
# ALL: --filter-label=foo
# ALL: Passed:

# --- --param decorate=time alone: pipeline still has the wrong CHECK, fails ---
# RUN: not %{lit} -a --param decorate=time %{inputs}/narrow-and-decorate/sample.ll 2>&1 \
# RUN:   | FileCheck --check-prefix=DEC_FAIL %s
#
# DEC_FAIL: time {{.*}}opt
# DEC_FAIL: FILTER_AWAY
