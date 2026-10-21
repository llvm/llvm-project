# RUN: llvm-mc -triple bpfel -filetype=obj -o %t %s
# RUN: llvm-objdump --no-print-imm-hex -d -r %t | FileCheck %s
# RUN: llvm-mc -triple bpfeb -filetype=obj -o %t %s
# RUN: llvm-objdump --no-print-imm-hex -d -r %t | FileCheck --check-prefix=CHECK-BE %s

// src_reg tells unwind_goto from may_goto and unwind_exit from exit.

  call 1
  unwind_goto Lpad    // BPF_JCOND | BPF_K, src_reg = 1
  may_goto Lpad       // BPF_JCOND | BPF_K, src_reg = 0
  exit                // BPF_EXIT
Lpad:
  unwind_exit         // BPF_EXIT, src_reg = 1

// CHECK: e5 10 02 00 00 00 00 00	unwind_goto +2
// CHECK: e5 00 01 00 00 00 00 00	may_goto +1
// CHECK: 95 00 00 00 00 00 00 00	exit
// CHECK: 95 10 00 00 00 00 00 00	unwind_exit

// CHECK-BE: e5 01 00 02 00 00 00 00	unwind_goto +2
// CHECK-BE: e5 00 00 01 00 00 00 00	may_goto +1
// CHECK-BE: 95 00 00 00 00 00 00 00	exit
// CHECK-BE: 95 01 00 00 00 00 00 00	unwind_exit
