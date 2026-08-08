# Check that .weak_definition is rejected at parse time on AIX/XCOFF.
#
# .weak_definition is a Mach-O-specific directive and is not supported on
# XCOFF. The documented AIX directive for weak symbols is .weak.
#
# RUN: not llvm-mc -triple powerpc-ibm-aix-xcoff %s 2>&1 | FileCheck %s

        .weak_definition foo
foo:
        blr

# CHECK: error: '.weak_definition' is not supported on XCOFF
