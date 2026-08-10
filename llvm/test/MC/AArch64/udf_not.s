# RUN: not llvm-mc -assemble -show-encoding -triple=aarch64- %s 2>&1 | FileCheck %s
udf 65536
udf -1
udf -768
# CHECK:{{.*}} immediate must be an integer in range [0, 65535].
# CHECK:{{.*}} immediate must be an integer in range [0, 65535].
# CHECK:{{.*}} immediate must be an integer in range [0, 65535].
