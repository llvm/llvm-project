# Preserve explicit "." in test_sub_dirs.

# RUN: not %{lit} -v --no-gtest-sharding %{inputs}/googletest-dot-subdir | FileCheck %s

# CHECK: FAIL: googletest-dot-subdir :: ./device-test.py

# RUN: not %{lit} -v %{inputs}/googletest-dot-subdir | FileCheck %s --check-prefix=SHARD

# SHARD: FAIL: googletest-dot-subdir :: ./device-test.py/{{[0-9]+}}/{{[0-9]+}}
