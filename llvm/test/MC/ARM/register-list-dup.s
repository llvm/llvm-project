// RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -show-encoding < %s 2>&1 | FileCheck -strict-whitespace %s

clrm {r0, r0}
// CHECK: warning: duplicated register (r0) in register list
// CHECK-NEXT: {{^clrm {r0, r0}}}
// CHECK-NEXT: {{^          \^}}

clrm {r0, r0, r1}
// CHECK: warning: duplicated register (r0) in register list
// CHECK-NEXT: {{^clrm {r0, r0, r1}}}
// CHECK-NEXT: {{^          \^}}

clrm {r0, r1, r0}
// CHECK: warning: duplicated register (r0) in register list
// CHECK-NEXT: {{^clrm {r0, r1, r0}}}
// CHECK-NEXT: {{^              \^}}

clrm {r0, r1, r1}
// CHECK: warning: duplicated register (r1) in register list
// CHECK-NEXT: {{^clrm {r0, r1, r1}}}
// CHECK-NEXT: {{^              \^}}

clrm {r1, r0, r1}
// CHECK: warning: duplicated register (r1) in register list
// CHECK-NEXT: {{^clrm {r1, r0, r1}}}
// CHECK-NEXT: {{^              \^}}

clrm {r1, r1, r0}
// CHECK: warning: duplicated register (r1) in register list
// CHECK-NEXT: {{^clrm {r1, r1, r0}}}
// CHECK-NEXT: {{^          \^}}

clrm {r0-r3, r0}
// CHECK: warning: duplicated register (r0) in register list
// CHECK-NEXT: {{^clrm {r0-r3, r0}}}
// CHECK-NEXT: {{^             \^}}

clrm {r2, r0-r3}
// CHECK: warning: duplicated register (r2) in register list
// CHECK-NEXT: {{^clrm {r2, r0-r3}}}
// CHECK-NEXT: {{^             \^}}

vscclrm {s0, s0, s1, vpr}
// CHECK: error: non-contiguous register range
// CHECK: {{^vscclrm {s0, s0, s1, vpr}}}
// CHECK: {{^             \^}}

vscclrm {s0-s3, vpr, s4}
// CHECK: error: register list not in ascending order
// CHECK-NEXT: {{^vscclrm {s0-s3, vpr, s4}}}
// CHECK-NEXT: {{^                     \^}}

vscclrm {s0-s3, vpr, vpr}
// CHECK: warning: duplicated register (vpr) in register list
// CHECK-NEXT: {{^vscclrm {s0-s3, vpr, vpr}}}
// CHECK-NEXT: {{^                     \^}}

vscclrm {q2, d4, vpr}
// CHECK: error: register list not in ascending order
// CHECK-NEXT: {{^vscclrm {q2, d4, vpr}}}
// CHECK-NEXT: {{^             \^}}

vscclrm {q2, d5, vpr}
// CHECK: error: non-contiguous register range
// CHECK-NEXT: {{^vscclrm {q2, d5, vpr}}}
// CHECK-NEXT: {{^             \^}}
