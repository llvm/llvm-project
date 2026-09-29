# Check that .weak_definition is rejected on AIX/XCOFF instead of crashing
# the streamer, matching ELF behavior.
#
# .weak_definition is a Mach-O-specific directive and is not supported on
# XCOFF. The documented AIX directive for weak symbols is .weak.
#
# RUN: not llvm-mc -triple powerpc-ibm-aix-xcoff %s -filetype=obj -o /dev/null 2>&1 | FileCheck %s

        .weak_definition foo
foo:
        blr

# CHECK: error: unable to emit symbol attribute
