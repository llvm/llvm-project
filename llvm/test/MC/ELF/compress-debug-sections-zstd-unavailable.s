# UNSUPPORTED: zstd
# RUN: not llvm-mc -filetype=obj -compress-debug-sections=zstd -triple x86_64 %s 2>&1 | FileCheck %s

# CHECK:     error: --compress-debug-sections: LLVM was not built with LLVM_ENABLE_ZSTD or did not find zstd at build time
# CHECK-NOT: {{.}}
