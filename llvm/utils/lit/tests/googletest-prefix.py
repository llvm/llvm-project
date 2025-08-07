# RUN: env GTEST_TOTAL_SHARDS=1 GTEST_SHARD_INDEX=0 \
# RUN: %{lit} -v --order=random --no-gtest-sharding %{inputs}/googletest-prefix --show-tests > %t.out
# FIXME: Temporarily dump test output so we can debug failing tests on
# buildbots.
# RUN: cat %t.out
# RUN: FileCheck < %t.out %s
#
# END.

# CHECK:      -- Available Tests --
# CHECK-NEXT:   googletest-format :: DummySubDir/test_one.py