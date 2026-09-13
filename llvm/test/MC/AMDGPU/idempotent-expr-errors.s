// RUN: split-file %s %t
// RUN: not llvm-mc -triple=amdgpu9.00-amd-amdhsa -filetype=obj %t/unresolved.s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNRESOLVED
// RUN: not llvm-mc -triple=amdgpu9.00-amd-amdhsa -filetype=obj %t/relocatable.s -o /dev/null 2>&1 | FileCheck %s --check-prefix=RELOCATABLE
// RUN: not llvm-mc -triple=amdgpu9.00-amd-amdhsa -filetype=obj %t/relocatable-alias.s -o /dev/null 2>&1 | FileCheck %s --check-prefix=RELOCATABLE --implicit-check-not="cyclic dependency"
// RUN: not llvm-mc -triple=amdgpu9.00-amd-amdhsa -filetype=obj %t/max-unresolved-first.s -o /dev/null 2>&1 | FileCheck %s --check-prefix=EARLY-FAILURE --implicit-check-not="cyclic dependency"
// RUN: not llvm-mc -triple=amdgpu9.00-amd-amdhsa -filetype=obj %t/max-relocatable-first.s -o /dev/null 2>&1 | FileCheck %s --check-prefix=EARLY-FAILURE --implicit-check-not="cyclic dependency"
// RUN: not llvm-mc -triple=amdgpu9.00-amd-amdhsa -filetype=obj %t/or-unresolved-first.s -o /dev/null 2>&1 | FileCheck %s --check-prefix=EARLY-FAILURE --implicit-check-not="cyclic dependency"
// RUN: not llvm-mc -triple=amdgpu9.00-amd-amdhsa -filetype=obj %t/or-relocatable-first.s -o /dev/null 2>&1 | FileCheck %s --check-prefix=EARLY-FAILURE --implicit-check-not="cyclic dependency"
// RUN: not llvm-mc -triple=amdgpu9.00-amd-amdhsa -filetype=obj %t/max-cycle.s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CYCLE
// RUN: not llvm-mc -triple=amdgpu9.00-amd-amdhsa -filetype=obj %t/or-cycle.s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CYCLE
// RUN: not llvm-mc -triple=amdgpu9.00-amd-amdhsa -filetype=obj %t/alias-cycle.s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CYCLE
// RUN: not llvm-mc -triple=amdgpu9.00-amd-amdhsa -filetype=obj %t/mixed-cycle.s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CYCLE
// RUN: not llvm-mc -triple=amdgpu9.00-amd-amdhsa -filetype=obj %t/binary-cycle.s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CYCLE

// A known all-ones OR operand does not make an unresolved or relocatable
// operand absolute. Iterative evaluation must preserve the existing rejection.
// UNRESOLVED: error: expected relocatable expression
// RELOCATABLE: error: expected relocatable expression

// Evaluate operands in order and stop on the first nonabsolute operand.
// A later cycle must not be visited after the target expression has failed.
// EARLY-FAILURE: error: expected relocatable expression

// A completed shared expression is reusable; an active symbol cycle is not.
// CYCLE: error: cyclic dependency detected for symbol

//--- unresolved.s
.data
.long or(-1, missing)

//--- relocatable.s
.data
label:
.byte 0
.long or(-1, label)

//--- max-cycle.s
.set a, max(b, 1)
.set b, max(a, 2)
.data
.long a

//--- or-cycle.s
.set a, or(b, 1)
.set b, or(a, 2)
.data
.long a

//--- alias-cycle.s
.set a, alias
.set alias, b
.set b, max(a, 2)
.data
.long a

//--- mixed-cycle.s
.set a, max(b, 1)
.set b, or(a, 2)
.data
.long a

//--- binary-cycle.s
.set a, max(b, 1)
.set b, a + 1
.data
.long a

//--- relocatable-alias.s
.data
label:
.byte 0
.set alias, label + 1
.long or(-1, alias)

//--- max-unresolved-first.s
.data
.long max(missing, .Lcycle_a)
.set .Lcycle_a, max(.Lcycle_b, 1)
.set .Lcycle_b, max(.Lcycle_a, 2)

//--- max-relocatable-first.s
.data
label:
.byte 0
.long max(label, .Lcycle_a)
.set .Lcycle_a, max(.Lcycle_b, 1)
.set .Lcycle_b, max(.Lcycle_a, 2)

//--- or-unresolved-first.s
.data
.long or(missing, .Lcycle_a)
.set .Lcycle_a, max(.Lcycle_b, 1)
.set .Lcycle_b, max(.Lcycle_a, 2)

//--- or-relocatable-first.s
.data
label:
.byte 0
.long or(label, .Lcycle_a)
.set .Lcycle_a, max(.Lcycle_b, 1)
.set .Lcycle_b, max(.Lcycle_a, 2)
