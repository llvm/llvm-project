# Create a directory with 20 files and check the number of pools and workers per pool that lit will use.

# RUN: rm -Rf %t.dir && mkdir -p %t.dir
# RUN: %{python} -c "for i in range(20): open(rf'%t.dir/file{i}.txt', 'w').write('RUN:')"

# RUN:  echo "import lit.formats" > %t.dir/lit.cfg
# RUN:  echo "config.name = \"top-level-suite\"" >> %t.dir/lit.cfg
# RUN:  echo "config.suffixes = [\".txt\"]" >> %t.dir/lit.cfg
# RUN:  echo "config.test_format = lit.formats.ShTest()" >> %t.dir/lit.cfg


# 15 workers per pool max, 100 workers total max: we expect lit to cap the workers to the number of files
# RUN: env "LIT_WINDOWS_MAX_WORKERS_PER_POOL=15" %{lit} -s %t.dir/ -j100 > %t.out 2>&1
# CHECK: Using 2 pools balancing 20 workers total distributed as [10, 10]
# CHECK: Passed: 20

# 5 workers per pool max, 17 workers total max
# RUN: env "LIT_WINDOWS_MAX_WORKERS_PER_POOL=5" %{lit} -s %t.dir/ -j17 >> %t.out 2>&1
# CHECK: Using 4 pools balancing 17 workers total distributed as [5, 4, 4, 4]
# CHECK: Passed: 20

# 19 workers per pool max, 19 workers total max
# RUN: env "LIT_WINDOWS_MAX_WORKERS_PER_POOL=19" %{lit} -s %t.dir/ -j19 >> %t.out 2>&1
# CHECK-NOT: workers total distributed as
# CHECK: Passed: 20

# RUN: cat %t.out | FileCheck %s
