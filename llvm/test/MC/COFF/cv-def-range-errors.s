# RUN: not llvm-mc < %s -filetype=obj --triple=x86_64-windows -o /dev/null 2>&1 | FileCheck %s

.cv_def_range  .Ltmp1 .Ltmp2
# CHECK: error: expected comma before def_range type in .cv_def_range directive
# CHECK: error: expected def_range type in directive

.cv_def_range  .Ltmp1 .Ltmp2,
# CHECK: error: expected def_range type in directive

.cv_def_range  .Ltmp1 .Ltmp2, subfield_reg
# CHECK: error: expected comma before register number in .cv_def_range directive
# CHECK: error: expected register number

.cv_def_range  .Ltmp1 .Ltmp2, subfield_reg,
# CHECK: error: unknown token in expression
# CHECK: error: expected register number

.cv_def_range  .Ltmp1 .Ltmp2, subfield_reg, 25
# CHECK: error: expected comma before offset in .cv_def_range directive
# CHECK: error: expected offset value

.cv_def_range  .Ltmp1 .Ltmp2, subfield_reg, 25,
# CHECK: error: unknown token in expression
# CHECK: error: expected offset value




.cv_def_range    .Ltmp1 .Ltmp2
# CHECK: error: expected comma before def_range type in .cv_def_range directive
# CHECK: error: expected def_range type in directive

.cv_def_range    .Ltmp1 .Ltmp2, 
# CHECK: error: expected def_range type in directive

.cv_def_range    .Ltmp1 .Ltmp2, reg
# CHECK: error: expected comma before register number in .cv_def_range directive
# CHECK: error: expected register number

.cv_def_range    .Ltmp1 .Ltmp2, reg,
# CHECK: error: unknown token in expression
# CHECK: error: expected register number




.cv_def_range    .Ltmp1 .Ltmp2
# CHECK: error: expected comma before def_range type in .cv_def_range directive
# CHECK: error: expected def_range type in directive

.cv_def_range    .Ltmp1 .Ltmp2, 
# CHECK: error: expected def_range type in directive

.cv_def_range    .Ltmp1 .Ltmp2, frame_ptr_rel
# CHECK: error: expected comma before offset in .cv_def_range directive
# CHECK: error: expected offset value

.cv_def_range    .Ltmp1 .Ltmp2, frame_ptr_rel,
# CHECK: error: unknown token in expression
# CHECK: error: expected offset value





.cv_def_range    .Ltmp1 .Ltmp2
# CHECK: error: expected comma before def_range type in .cv_def_range directive
# CHECK: error: expected def_range type in directive

.cv_def_range    .Ltmp1 .Ltmp2, 
# CHECK: error: expected def_range type in directive

.cv_def_range    .Ltmp1 .Ltmp2, reg_rel
# CHECK: error: expected comma before register number in .cv_def_range directive
# CHECK: error: expected register value

.cv_def_range    .Ltmp1 .Ltmp2, reg_rel, 
# CHECK: error: unknown token in expression
# CHECK: error: expected register value

.cv_def_range    .Ltmp1 .Ltmp2, reg_rel, 330
# CHECK: error: expected comma before flag value in .cv_def_range directive
# CHECK: error: expected flag value

.cv_def_range    .Ltmp1 .Ltmp2, reg_rel, 330, 
# CHECK: error: unknown token in expression
# CHECK: error: expected flag value

.cv_def_range    .Ltmp1 .Ltmp2, reg_rel, 330, 0
# CHECK: error: expected comma before base pointer offset in .cv_def_range directive
# CHECK: error: expected base pointer offset value

.cv_def_range    .Ltmp1 .Ltmp2, reg_rel, 330, 0,
# CHECK: error: unknown token in expression
# CHECK: error: expected base pointer offset value
