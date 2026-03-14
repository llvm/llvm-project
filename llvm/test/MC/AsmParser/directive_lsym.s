# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# FIXME: This is currently unsupported. If it turns out no one uses it, we
# should just rip it out.
        
# XFAIL: *

# CHECK: TEST0:
# CHECK: .lsym bar,foo
# CHECK: .lsym baz,3
TEST0:  
        .lsym   bar, foo
        .lsym baz, 2+1
