# RUN: llvm-mc -filetype=obj -o %t -mcpu=native %s 2> %t.stderr
# RUN: FileCheck --allow-empty %s < %t.stderr

# CHECK-NOT: {{.+}}
