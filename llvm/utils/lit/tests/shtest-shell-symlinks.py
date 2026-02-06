# Check that the internal shell builtins correctly handle cases involving
# symlinks.

# REQUIRES: symlinks
# RUN: echo test
# RUN: %{lit} -v %{inputs}/shtest-shell-symlinks | FileCheck %s

# CHECK: -- Testing: 1 test{{.*}}
# CHECK: PASS: shtest-shell :: rm-symlink-dir.txt
