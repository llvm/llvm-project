## Test the integration with LLVM daemon tools.
#
# REQUIRES: have-llvm-build
# RUN: %{lit} %{inputs}/llvm-daemon-integration \
# RUN: | FileCheck --match-full-lines %s
# END.

# CHECK: PASS: llvm-daemon-integration :: use-daemon.txt ({{.*}})
