# RUN: not --crash llvm-mc -triple=powerpc64le-unknown-linux-gnu -filetype=obj %s \
# RUN: 2>&1 | FileCheck %s
        
_stext:
ld %r5, p_end - _stext(%r5)

# CHECK: LLVM ERROR: Invalid PC-relative half16ds relocation
