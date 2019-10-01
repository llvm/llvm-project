# Check the basic discovery process, including a sub-suite.
#
# RUN: %{lit} %{inputs}/discovery \
# RUN:   -j 1 --debug --show-tests --show-suites \
# RUN:   -v > %t.out 2> %t.err
# RUN: FileCheck --check-prefix=CHECK-BASIC-OUT < %t.out %s
# RUN: FileCheck --check-prefix=CHECK-BASIC-ERR < %t.err %s
#
# CHECK-BASIC-ERR: loading suite config '{{.*(/|\\\\)discovery(/|\\\\)lit.cfg}}'
# CHECK-BASIC-ERR-DAG: loading suite config '{{.*(/|\\\\)discovery(/|\\\\)subsuite(/|\\\\)lit.cfg}}'
# CHECK-BASIC-ERR-DAG: loading local config '{{.*(/|\\\\)discovery(/|\\\\)subdir(/|\\\\)lit.local.cfg}}'
#
# CHECK-BASIC-OUT: -- Test Suites --
# CHECK-BASIC-OUT:   sub-suite - 2 tests
# CHECK-BASIC-OUT:     Source Root: {{.*[/\\]discovery[/\\]subsuite$}}
# CHECK-BASIC-OUT:     Exec Root  : {{.*[/\\]discovery[/\\]subsuite$}}
# CHECK-BASIC-OUT:   top-level-suite - 3 tests
# CHECK-BASIC-OUT:     Source Root: {{.*[/\\]discovery$}}
# CHECK-BASIC-OUT:     Exec Root  : {{.*[/\\]discovery$}}
#
# CHECK-BASIC-OUT: -- Available Tests --
# CHECK-BASIC-OUT: sub-suite :: test-one
# CHECK-BASIC-OUT: sub-suite :: test-two
# CHECK-BASIC-OUT: top-level-suite :: subdir/test-three
# CHECK-BASIC-OUT: top-level-suite :: test-one
# CHECK-BASIC-OUT: top-level-suite :: test-two

# Check discovery when providing the special builtin 'config_map'
# RUN: %{python} %{inputs}/config-map-discovery/driver.py \
# RUN:           %{inputs}/config-map-discovery/main-config/lit.cfg \
# RUN:           %{inputs}/config-map-discovery/lit.alt.cfg \
# RUN:           --threads=1 --debug --show-tests --show-suites > %t.out 2> %t.err
# RUN: FileCheck --check-prefix=CHECK-CONFIG-MAP-OUT < %t.out %s
# RUN: FileCheck --check-prefix=CHECK-CONFIG-MAP-ERR < %t.err %s

# CHECK-CONFIG-MAP-OUT-NOT: ERROR: lit.cfg invoked
# CHECK-CONFIG-MAP-OUT: -- Test Suites --
# CHECK-CONFIG-MAP-OUT:   config-map - 2 tests
# CHECK-CONFIG-MAP-OUT:     Source Root: {{.*[/\\]config-map-discovery[/\\]tests}}
# CHECK-CONFIG-MAP-OUT:     Exec Root  : {{.*[/\\]tests[/\\]Inputs[/\\]config-map-discovery}}
# CHECK-CONFIG-MAP-OUT: -- Available Tests --
# CHECK-CONFIG-MAP-OUT-NOT: invalid-test.txt
# CHECK-CONFIG-MAP-OUT:   config-map :: test1.txt
# CHECK-CONFIG-MAP-OUT:   config-map :: test2.txt

# CHECK-CONFIG-MAP-ERR: loading suite config '{{.*}}lit.alt.cfg'
# CHECK-CONFIG-MAP-ERR: loaded config '{{.*}}lit.alt.cfg'
# CHECK-CONFIG-MAP-ERR: resolved input '{{.*(/|\\\\)config-map-discovery(/|\\\\)main-config}}' to 'config-map'::()


# Check discovery when exact test names are given.
#
# RUN: %{lit} \
# RUN:     %{inputs}/discovery/subdir/test-three.py \
# RUN:     %{inputs}/discovery/subsuite/test-one.txt \
# RUN:   -j 1 --show-tests --show-suites -v > %t.out
# RUN: FileCheck --check-prefix=CHECK-EXACT-TEST < %t.out %s
#
# CHECK-EXACT-TEST: -- Available Tests --
# CHECK-EXACT-TEST: sub-suite :: test-one
# CHECK-EXACT-TEST: top-level-suite :: subdir/test-three

# Check discovery when config files end in .py
# RUN: %{lit} %{inputs}/py-config-discovery \
# RUN:   -j 1 --debug --show-tests --show-suites \
# RUN:   -v > %t.out 2> %t.err
# RUN: FileCheck --check-prefix=CHECK-PYCONFIG-OUT < %t.out %s
# RUN: FileCheck --check-prefix=CHECK-PYCONFIG-ERR < %t.err %s
#
# CHECK-PYCONFIG-ERR: loading suite config '{{.*(/|\\\\)py-config-discovery(/|\\\\)lit.site.cfg.py}}'
# CHECK-PYCONFIG-ERR: load_config from '{{.*(/|\\\\)discovery(/|\\\\)lit.cfg}}'
# CHECK-PYCONFIG-ERR: loaded config '{{.*(/|\\\\)discovery(/|\\\\)lit.cfg}}'
# CHECK-PYCONFIG-ERR: loaded config '{{.*(/|\\\\)py-config-discovery(/|\\\\)lit.site.cfg.py}}'
# CHECK-PYCONFIG-ERR-DAG: loading suite config '{{.*(/|\\\\)discovery(/|\\\\)subsuite(/|\\\\)lit.cfg}}'
# CHECK-PYCONFIG-ERR-DAG: loading local config '{{.*(/|\\\\)discovery(/|\\\\)subdir(/|\\\\)lit.local.cfg}}'
#
# CHECK-PYCONFIG-OUT: -- Test Suites --
# CHECK-PYCONFIG-OUT:   sub-suite - 2 tests
# CHECK-PYCONFIG-OUT:     Source Root: {{.*[/\\]discovery[/\\]subsuite$}}
# CHECK-PYCONFIG-OUT:     Exec Root  : {{.*[/\\]discovery[/\\]subsuite$}}
# CHECK-PYCONFIG-OUT:   top-level-suite - 3 tests
# CHECK-PYCONFIG-OUT:     Source Root: {{.*[/\\]discovery$}}
# CHECK-PYCONFIG-OUT:     Exec Root  : {{.*[/\\]py-config-discovery$}}
#
# CHECK-PYCONFIG-OUT: -- Available Tests --
# CHECK-PYCONFIG-OUT: sub-suite :: test-one
# CHECK-PYCONFIG-OUT: sub-suite :: test-two
# CHECK-PYCONFIG-OUT: top-level-suite :: subdir/test-three
# CHECK-PYCONFIG-OUT: top-level-suite :: test-one
# CHECK-PYCONFIG-OUT: top-level-suite :: test-two

# Check discovery when using an exec path.
#
# RUN: %{lit} %{inputs}/exec-discovery \
# RUN:   -j 1 --debug --show-tests --show-suites \
# RUN:   -v > %t.out 2> %t.err
# RUN: FileCheck --check-prefix=CHECK-ASEXEC-OUT < %t.out %s
# RUN: FileCheck --check-prefix=CHECK-ASEXEC-ERR < %t.err %s
#
# CHECK-ASEXEC-ERR: loading suite config '{{.*(/|\\\\)exec-discovery(/|\\\\)lit.site.cfg}}'
# CHECK-ASEXEC-ERR: load_config from '{{.*(/|\\\\)discovery(/|\\\\)lit.cfg}}'
# CHECK-ASEXEC-ERR: loaded config '{{.*(/|\\\\)discovery(/|\\\\)lit.cfg}}'
# CHECK-ASEXEC-ERR: loaded config '{{.*(/|\\\\)exec-discovery(/|\\\\)lit.site.cfg}}'
# CHECK-ASEXEC-ERR-DAG: loading suite config '{{.*(/|\\\\)discovery(/|\\\\)subsuite(/|\\\\)lit.cfg}}'
# CHECK-ASEXEC-ERR-DAG: loading local config '{{.*(/|\\\\)discovery(/|\\\\)subdir(/|\\\\)lit.local.cfg}}'
#
# CHECK-ASEXEC-OUT: -- Test Suites --
# CHECK-ASEXEC-OUT:   sub-suite - 2 tests
# CHECK-ASEXEC-OUT:     Source Root: {{.*[/\\]discovery[/\\]subsuite$}}
# CHECK-ASEXEC-OUT:     Exec Root  : {{.*[/\\]discovery[/\\]subsuite$}}
# CHECK-ASEXEC-OUT:   top-level-suite - 3 tests
# CHECK-ASEXEC-OUT:     Source Root: {{.*[/\\]discovery$}}
# CHECK-ASEXEC-OUT:     Exec Root  : {{.*[/\\]exec-discovery$}}
#
# CHECK-ASEXEC-OUT: -- Available Tests --
# CHECK-ASEXEC-OUT: sub-suite :: test-one
# CHECK-ASEXEC-OUT: sub-suite :: test-two
# CHECK-ASEXEC-OUT: top-level-suite :: subdir/test-three
# CHECK-ASEXEC-OUT: top-level-suite :: test-one
# CHECK-ASEXEC-OUT: top-level-suite :: test-two

# Check discovery when exact test names are given.
#
# FIXME: Note that using a path into a subsuite doesn't work correctly here.
#
# RUN: %{lit} \
# RUN:     %{inputs}/exec-discovery/subdir/test-three.py \
# RUN:   -j 1 --show-tests --show-suites -v > %t.out
# RUN: FileCheck --check-prefix=CHECK-ASEXEC-EXACT-TEST < %t.out %s
#
# CHECK-ASEXEC-EXACT-TEST: -- Available Tests --
# CHECK-ASEXEC-EXACT-TEST: top-level-suite :: subdir/test-three


# Check that we don't recurse infinitely when loading an site specific test
# suite located inside the test source root.
#
# RUN: %{lit} \
# RUN:     %{inputs}/exec-discovery-in-tree/obj/ \
# RUN:   -j 1 --show-tests --show-suites -v > %t.out
# RUN: FileCheck --check-prefix=CHECK-ASEXEC-INTREE < %t.out %s
#
# Try it again after cd'ing into the test suite using a short relative path.
#
# RUN: cd %{inputs}/exec-discovery-in-tree/obj/
# RUN: %{lit} . \
# RUN:   -j 1 --show-tests --show-suites -v > %t.out
# RUN: FileCheck --check-prefix=CHECK-ASEXEC-INTREE < %t.out %s
#
#      CHECK-ASEXEC-INTREE:   exec-discovery-in-tree-suite - 1 tests
# CHECK-ASEXEC-INTREE-NEXT:     Source Root: {{.*[/\\]exec-discovery-in-tree$}}
# CHECK-ASEXEC-INTREE-NEXT:     Exec Root  : {{.*[/\\]exec-discovery-in-tree[/\\]obj$}}
# CHECK-ASEXEC-INTREE-NEXT: -- Available Tests --
# CHECK-ASEXEC-INTREE-NEXT: exec-discovery-in-tree-suite :: test-one
