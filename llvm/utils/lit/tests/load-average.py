# Create a directory with 20 files and check the limit message.

# RUN: rm -Rf %t.dir && mkdir -p %t.dir
# RUN: %{python} -c "for i in range(20): open(rf'%t.dir/file{i}.txt', 'w').write('RUN:')"

# RUN:  echo "import lit.formats" > %t.dir/lit.cfg
# RUN:  echo "config.name = \"top-level-suite\"" >> %t.dir/lit.cfg
# RUN:  echo "config.suffixes = [\".txt\"]" >> %t.dir/lit.cfg
# RUN:  echo "config.test_format = lit.formats.ShTest()" >> %t.dir/lit.cfg

# RUN: %{lit} -s %t.dir/ -j100 --load-limit 0.01 > %t.out 2>&1
# CHECK: load-limit 1%
# CHECK: Passed: 20

# RUN: %{lit} -s %t.dir/ -l0.2 >> %t.out 2>&1
# CHECK: load-limit 20%
# CHECK: Passed: 20

# RUN: %{lit} -s %t.dir/ -j5 -l0.6 >> %t.out 2>&1
# CHECK: load-limit 60%
# CHECK: Passed: 20

# RUN: %{lit} -s %t.dir/ -j10 -l0.8 >> %t.out 2>&1
# CHECK: load-limit 80%
# CHECK: Passed: 20

# RUN: %{lit} -s %t.dir/ -j100 -l 0 >> %t.out 2>&1 || true
# CHECK: lit: error: argument -l/--load-limit: requires number in (0, 2], but found '0'

# RUN: %{lit} -s %t.dir/ -j100 --load-limit 2.1 >> %t.out 2>&1 || true
# CHECK: lit: error: argument -l/--load-limit: requires number in (0, 2], but found '2.1'

# RUN: %{lit} -s %t.dir/ -j100 -l five >> %t.out 2>&1 || true
# CHECK: lit: error: argument -l/--load-limit: requires number in (0, 2], but found 'five'

# RUN: cat %t.out | FileCheck %s
