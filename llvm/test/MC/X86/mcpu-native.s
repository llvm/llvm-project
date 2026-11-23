# REQUIRES: host=x86_64-{{.*}}
# RUN: llvm-mc -triple=x86_64 -filetype=obj -o %t -mcpu=native %s 2> %t.stderr
# RUN: FileCheck --allow-empty %s < %t.stderr

# CHECK-NOT: {{.+}}
