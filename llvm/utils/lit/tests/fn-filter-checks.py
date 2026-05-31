# Verify the end-to-end workflow: with `--param fn=NAMES`, llvm-extract narrows
# the IR and `FileCheck --filter-label=NAMES` drops CHECKs for the removed
# functions, so a CHECK that would otherwise fail is hidden.

# With --param fn=foo: llvm-extract removes @bar;
# --filter-label=foo is added to FileCheck, so the test passes.
# RUN: %{lit} --param fn=foo %{inputs}/fn-filter-checks/sample.ll

# Without --param fn=foo: @bar fails, so the test fails.
# RUN: not %{lit} %{inputs}/fn-filter-checks/sample.ll

# With --param fn=foo --param fn-pass=1: select-function removes @bar,
# so the test passes.
# RUN: %{lit} --param fn=foo --param fn-pass=1 %{inputs}/fn-filter-checks/sample.ll
