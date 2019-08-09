# RUN: llvm-mc -arch=hexagon -mhvx --filetype=asm %s -o - 2>&1 | FileCheck %s
# RUN: llvm-mc --no-warn -arch=hexagon -mhvx --filetype=obj %s -o - | llvm-objdump -d - | FileCheck --check-prefix=CHECK-NOWARN %s
# RUN: not llvm-mc --fatal-warnings -arch=hexagon -mhvx --filetype=asm %s 2>&1 | FileCheck --check-prefix=CHECK-FATAL-WARN %s

	.text
    .warning

{
  v7.tmp = vmem(r28 + #3)
  v7:6.w = vadd(v17:16.w, v17:16.w)
  v17:16.uw = vunpack(v8.uh)
}

# CHECK-NOWARN-NOT: warning
# CHECK-FATAL-WARN-NOT: warning
# CHECK-FATAL-WARN: error
# CHECK-FATAL-WARN: error
# CHECK: warning:
# CHECK: warning:
